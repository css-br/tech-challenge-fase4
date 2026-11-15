"""Detecção de anomalias em séries temporais de sinais vitais.

Combina duas verificações:
1. Limites clínicos: valores perigosos por si só (ex.: oxigenação abaixo de 88%).
2. Isolation Forest: um modelo por sinal, que aprende o padrão normal e
   acusa valores ou mudanças bruscas incomuns, mesmo dentro dos limites.

O detector funciona amostra a amostra, como em um monitor de leito.
"""

import math
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.modelos import Evento

SINAIS = ["fc", "spo2", "fr", "pas"]  # batimentos, oxigenação, respiração, pressão sistólica
UNIDADES = {"fc": "bpm", "spo2": "%", "fr": "rpm", "pas": "mmHg"}

JANELA_S = 120               # histórico usado como referência do paciente (1 amostra por segundo)
CONTAMINACAO = 0.005         # fração de pontos que cada modelo trata como anômalos no treino
SUAVIZACAO_PRESSAO = 30      # suavização da pressão simulada, em amostras
AMOSTRAS_PARA_ENCERRAR = 5   # amostras normais seguidas que encerram um episódio

# Sinal -> (mínimo, máximo) aceitável. Fora disso o evento é grave.
LIMITES_CLINICOS = {
    "spo2": (88.0, math.inf),
    "fc": (40.0, 140.0),
    "pas": (0.0, 180.0),
}


def carregar_sinais(caminho_csv: Path) -> pd.DataFrame:
    """Lê um arquivo *_Numerics.csv do BIDMC.

    Colunas devolvidas: tempo_s, fc, spo2, fr, pas.
    O BIDMC não traz pressão arterial, então a pressão sistólica (pas) é
    SIMULADA: oscila suavemente em torno de 120 mmHg. A simulação é
    reprodutível (a semente vem do nome do arquivo).
    """
    caminho_csv = Path(caminho_csv)
    bruto = pd.read_csv(caminho_csv)
    bruto.columns = [coluna.strip() for coluna in bruto.columns]
    sinais = bruto.rename(columns={"Time [s]": "tempo_s", "HR": "fc", "SpO2": "spo2", "RESP": "fr"})

    gerador = np.random.default_rng(zlib.crc32(caminho_csv.name.encode()))
    ruido = gerador.normal(0, 8, len(sinais) + SUAVIZACAO_PRESSAO)
    suave = np.convolve(ruido, np.ones(SUAVIZACAO_PRESSAO) / SUAVIZACAO_PRESSAO, mode="valid")[: len(sinais)]
    sinais["pas"] = np.clip(120 + suave * 3 + gerador.normal(0, 1, len(sinais)), 80, 160).round(1)

    return sinais[["tempo_s", *SINAIS]]


def injetar_anomalias(df: pd.DataFrame, semente: int = 42) -> pd.DataFrame:
    """Devolve uma cópia da série com três anomalias artificiais e a coluna `anomalia_real`.

    Uma anomalia em cada terço da série, com 20 a 40 s de duração:
    queda de oxigenação, taquicardia e pico de pressão. Como sabemos onde
    elas estão, podemos medir precisão e recall do detector.
    """
    gerador = np.random.default_rng(semente)
    resultado = df.copy()
    resultado["anomalia_real"] = False
    for sinal in ("fc", "spo2", "pas"):
        resultado[sinal] = resultado[sinal].astype(float)

    alteracoes = [("spo2", -15.0, -12.0), ("fc", 45.0, 60.0), ("pas", 55.0, 70.0)]
    terco = len(resultado) // 3
    for i, (sinal, menor, maior) in enumerate(alteracoes):
        duracao = int(gerador.integers(20, 41))
        inicio = i * terco + int(gerador.integers(0, terco - duracao))
        trecho = resultado.index[inicio : inicio + duracao]
        resultado.loc[trecho, sinal] += gerador.uniform(menor, maior)
        resultado.loc[trecho, "anomalia_real"] = True
    return resultado


def _valida(amostra: dict) -> bool:
    """Leitura utilizável: todos os sinais presentes, numéricos e maiores que zero."""
    for sinal in SINAIS:
        valor = amostra.get(sinal)
        if valor is None or not np.isfinite(valor) or valor <= 0:
            return False
    return True


class DetectorSinaisVitais:
    """Detector de anomalias que avalia uma amostra por vez.

    Há um Isolation Forest para cada sinal. Cada um olha duas coisas:
    o valor atual e o quanto ele se afasta da mediana dos últimos 120 s
    do próprio paciente. Assim o detector pega tanto valores raros quanto
    mudanças bruscas, e sabe dizer qual sinal disparou.

    (Um único modelo com todos os sinais juntos foi testado e detectou
    bem menos: a alteração em um sinal ficava diluída entre os demais.)
    """

    def __init__(self):
        self.modelos = {
            sinal: IsolationForest(contamination=CONTAMINACAO, random_state=42) for sinal in SINAIS
        }
        self.reiniciar()

    def reiniciar(self) -> None:
        """Esquece o histórico. Usar ao começar a monitorar outro paciente."""
        self._janela: deque[dict] = deque(maxlen=JANELA_S)
        self._severidade_episodio = 0   # 0 = fora de episódio
        self._normais_seguidas = 0
        self._sensor_em_falha = False

    # ---------- características ----------

    @staticmethod
    def _caracteristicas(validos: pd.DataFrame) -> pd.DataFrame:
        """Para cada sinal: o valor e o quanto ele se afasta da mediana dos últimos 120 s."""
        mediana_movel = validos[SINAIS].rolling(JANELA_S, min_periods=1).median()
        desvios = (validos[SINAIS] - mediana_movel).add_suffix("_desvio")
        return pd.concat([validos[SINAIS], desvios], axis=1)

    @staticmethod
    def _linhas_validas(df: pd.DataFrame) -> pd.Series:
        return df[SINAIS].notna().all(axis=1) & (df[SINAIS] > 0).all(axis=1)

    @staticmethod
    def _fora_dos_limites(valores: pd.DataFrame) -> pd.DataFrame:
        """Tabela booleana: para cada linha, quais sinais violam o limite clínico."""
        fora = pd.DataFrame(False, index=valores.index, columns=SINAIS)
        for sinal, (minimo, maximo) in LIMITES_CLINICOS.items():
            fora[sinal] = (valores[sinal] < minimo) | (valores[sinal] > maximo)
        return fora

    def _anomalos_pelo_modelo(self, caracteristicas: pd.DataFrame) -> pd.DataFrame:
        """Tabela booleana: para cada linha, quais sinais o modelo considera anômalos."""
        anomalos = pd.DataFrame(False, index=caracteristicas.index, columns=SINAIS)
        for sinal, modelo in self.modelos.items():
            anomalos[sinal] = modelo.predict(caracteristicas[[sinal, f"{sinal}_desvio"]]) == -1
        return anomalos

    # ---------- treino ----------

    def treinar(self, series: pd.DataFrame | list[pd.DataFrame]) -> None:
        """Aprende o padrão normal a partir de uma ou mais séries (uma por paciente)."""
        if isinstance(series, pd.DataFrame):
            series = [series]
        partes = []
        for serie in series:
            validos = serie[self._linhas_validas(serie)]
            partes.append(self._caracteristicas(validos))
        treino = pd.concat(partes, ignore_index=True)
        for sinal, modelo in self.modelos.items():
            modelo.fit(treino[[sinal, f"{sinal}_desvio"]])

    # ---------- avaliação de uma série inteira ----------

    def avaliar_serie(self, df: pd.DataFrame) -> pd.Series:
        """True em cada linha considerada anômala. Leituras inválidas ficam como False."""
        marcado = pd.Series(False, index=df.index)
        validos = df[self._linhas_validas(df)]
        if validos.empty:
            return marcado
        pelo_modelo = self._anomalos_pelo_modelo(self._caracteristicas(validos)).any(axis=1)
        pelos_limites = self._fora_dos_limites(validos).any(axis=1)
        marcado.loc[validos.index] = (pelo_modelo | pelos_limites).to_numpy()
        return marcado

    # ---------- avaliação amostra a amostra ----------

    def avaliar_amostra(self, amostra: dict, tempo_s: float) -> list[Evento]:
        """Avalia a leitura mais recente e devolve os eventos novos (em geral, nenhum).

        Um episódio contínuo gera um único evento, no instante em que começa
        (ou um segundo evento se a gravidade aumentar), para não inundar a
        equipe com alertas repetidos.
        """
        if not _valida(amostra):
            if self._sensor_em_falha:
                return []
            self._sensor_em_falha = True
            return [Evento("sensor", "leitura_invalida", 1, tempo_s,
                           "Leitura de sinais vitais ausente ou zerada (verificar sensor)")]
        self._sensor_em_falha = False

        self._janela.append({sinal: float(amostra[sinal]) for sinal in SINAIS})
        janela = pd.DataFrame(self._janela)
        atual = janela.iloc[[-1]]

        fora_dos_limites = self._fora_dos_limites(atual).iloc[0]
        pelo_modelo = self._anomalos_pelo_modelo(self._caracteristicas(janela).iloc[[-1]]).iloc[0]
        alterados = [sinal for sinal in SINAIS if fora_dos_limites[sinal] or pelo_modelo[sinal]]

        if not alterados:
            self._normais_seguidas += 1
            if self._normais_seguidas >= AMOSTRAS_PARA_ENCERRAR:
                self._severidade_episodio = 0
            return []

        self._normais_seguidas = 0
        grave = bool(fora_dos_limites.any())
        severidade = 3 if grave else 2
        if severidade <= self._severidade_episodio:
            return []  # episódio já comunicado
        self._severidade_episodio = severidade

        valores = atual.iloc[0]
        leituras = ", ".join(f"{sinal.upper()} {valores[sinal]:.0f} {UNIDADES[sinal]}" for sinal in SINAIS)
        return [Evento(
            fonte="sinais_vitais",
            tipo="limite_clinico" if grave else "padrao_anomalo",
            severidade=severidade,
            instante=tempo_s,
            descricao=(
                f"Sinais vitais fora do padrão ({', '.join(s.upper() for s in alterados)}). "
                f"Leituras: {leituras}"
            ),
        )]
