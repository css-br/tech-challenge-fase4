"""Detecção de alterações vocais a partir de características acústicas.

O classificador é treinado com o dataset VOICED (PhysioNet): gravações da
vogal "a" sustentada, de pessoas saudáveis e de pessoas com patologias da voz.
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.modelos import Evento

# Faixa de frequência fundamental procurada (cobre vozes graves e agudas)
F0_MINIMA_HZ = 60
F0_MAXIMA_HZ = 500
TAMANHO_QUADRO = 1024
PASSO = 256
N_MFCC = 13

# Limite de decisão usado quando nenhum outro é informado
LIMITE_PADRAO = 0.5
# Fração máxima de vozes saudáveis que aceitamos apontar como alteradas
MAX_FALSO_ALARME = 0.2


def _variacao_relativa(valores: np.ndarray) -> float:
    """Média da diferença entre valores vizinhos, dividida pela média dos valores."""
    if len(valores) < 2 or np.mean(valores) == 0:
        return 0.0
    return float(np.mean(np.abs(np.diff(valores))) / np.mean(valores))


def extrair_caracteristicas(sinal: np.ndarray, sr: int) -> dict[str, float]:
    """Resume um trecho de voz em números que descrevem sua qualidade.

    - f0_media, f0_desvio: altura da voz (pitch) e o quanto ela varia.
    - jitter: instabilidade da frequência de um instante para o seguinte.
    - shimmer: instabilidade do volume de um instante para o seguinte.
    - proporcao_vozeada: fração do tempo em que há vibração das cordas vocais.
    - rms_media, zcr_media: energia e "ruidosidade" do sinal.
    - mfcc_1..13: resumo do timbre.

    Jitter e shimmer são calculados entre quadros vizinhos (cerca de 32 ms),
    uma aproximação da definição clínica, que mede ciclo a ciclo.
    """
    sinal = np.asarray(sinal, dtype=float)

    f0, vozeado, _ = librosa.pyin(
        sinal, fmin=F0_MINIMA_HZ, fmax=F0_MAXIMA_HZ, sr=sr,
        frame_length=TAMANHO_QUADRO, hop_length=PASSO,
    )
    rms = librosa.feature.rms(y=sinal, frame_length=TAMANHO_QUADRO, hop_length=PASSO)[0]
    zcr = librosa.feature.zero_crossing_rate(sinal, frame_length=TAMANHO_QUADRO, hop_length=PASSO)[0]
    mfcc = librosa.feature.mfcc(y=sinal, sr=sr, n_mfcc=N_MFCC, n_fft=TAMANHO_QUADRO, hop_length=PASSO)

    vozeado = vozeado & np.isfinite(f0)
    f0_vozeado = f0[vozeado]
    tem_voz = len(f0_vozeado) > 0

    caracteristicas = {
        "f0_media": float(np.mean(f0_vozeado)) if tem_voz else 0.0,
        "f0_desvio": float(np.std(f0_vozeado)) if tem_voz else 0.0,
        "jitter": _variacao_relativa(1.0 / f0_vozeado) if tem_voz else 0.0,
        "shimmer": _variacao_relativa(rms[: len(vozeado)][vozeado]) if tem_voz else 0.0,
        "proporcao_vozeada": float(np.mean(vozeado)),
        "rms_media": float(np.mean(rms)),
        "zcr_media": float(np.mean(zcr)),
    }
    for i in range(N_MFCC):
        caracteristicas[f"mfcc_{i + 1}"] = float(np.mean(mfcc[i]))
    return caracteristicas


def carregar_voiced(pasta: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Lê o VOICED e devolve (características, rótulo, informações do registro).

    Rótulo: 0 = voz saudável, 1 = voz patológica.
    """
    import wfdb

    pasta = Path(pasta)
    linhas, rotulos, infos = [], [], []
    for registro in (pasta / "RECORDS").read_text().split():
        gravacao = wfdb.rdrecord(str(pasta / registro))
        comentario = " ".join(gravacao.comments)
        diagnostico = comentario.split("<diagnoses>:")[1].split("<medications>")[0].strip()
        linhas.append(extrair_caracteristicas(gravacao.p_signal[:, 0], gravacao.fs))
        rotulos.append(0 if diagnostico.lower() == "healthy" else 1)
        infos.append({"registro": registro, "diagnostico": diagnostico})
    return pd.DataFrame(linhas), pd.Series(rotulos, name="patologica"), pd.DataFrame(infos)


def escolher_limite(y, probabilidades, max_falso_alarme: float = MAX_FALSO_ALARME) -> float:
    """Menor limite de decisão que mantém os falsos alarmes dentro do máximo.

    Falso alarme = voz saudável com probabilidade acima do limite. Quanto
    menor o limite, mais vozes alteradas são encontradas, mas mais vozes
    saudáveis são apontadas por engano. O limite nunca fica abaixo de 0.5.
    """
    y = np.asarray(y)
    probabilidades = np.asarray(probabilidades, dtype=float)
    saudaveis = probabilidades[y == 0]
    candidatos = sorted({LIMITE_PADRAO, *(float(p) for p in probabilidades if p > LIMITE_PADRAO)})
    for limite in candidatos:
        if np.mean(saudaveis > limite) <= max_falso_alarme:
            return limite
    return candidatos[-1]


def treinar_classificador(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestClassifier, dict]:
    """Treina o classificador de voz saudável (0) contra patológica (1).

    As métricas vêm de validação cruzada: cada gravação é avaliada por um
    modelo que não a viu no treino. O modelo devolvido é treinado com tudo.

    O limite de decisão é escolhido para limitar os falsos alarmes, e as
    métricas de recall, precisão e F1 são calculadas nesse limite.
    """
    modelo = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    divisao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilidades = cross_val_predict(modelo, X, y, cv=divisao, method="predict_proba")[:, 1]

    limite = escolher_limite(y, probabilidades)
    previsto = (probabilidades > limite).astype(int)
    matriz = confusion_matrix(y, previsto)

    metricas = {
        "auc": float(roc_auc_score(y, probabilidades)),
        "limite": limite,
        "recall": float(recall_score(y, previsto)),
        "precisao": float(precision_score(y, previsto, zero_division=0)),
        "f1": float(f1_score(y, previsto)),
        "falso_alarme": float(matriz[0, 1] / matriz[0].sum()),
        "matriz_confusao": matriz.tolist(),
        "probabilidades": probabilidades.tolist(),
    }
    modelo.fit(X, y)
    return modelo, metricas


def avaliar_voz(sinal: np.ndarray, sr: int, modelo, limite: float = LIMITE_PADRAO) -> list[Evento]:
    """Gera um evento da fonte "voz" quando o modelo indica voz alterada.

    A severidade é 1 (leve): a voz sozinha é um indício fraco, que só pesa
    no risco quando aparece junto com eventos de outras fontes.
    """
    caracteristicas = pd.DataFrame([extrair_caracteristicas(sinal, sr)])
    probabilidade = float(modelo.predict_proba(caracteristicas)[0][1])
    if probabilidade <= limite:
        return []
    return [
        Evento(
            fonte="voz",
            tipo="voz_alterada",
            severidade=1,
            instante=0.0,
            descricao=f"Características da voz compatíveis com alteração vocal ({probabilidade:.0%})",
        )
    ]
