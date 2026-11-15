"""Detecção de movimentos fora do padrão a partir dos ângulos das articulações.

Ideia: vídeos de execuções corretas definem a faixa esperada de cada
ângulo. Em um vídeo novo, um ângulo que fica fora dessa faixa por tempo
suficiente é registrado como desvio.
"""

import numpy as np
import pandas as pd

from src.modelos import Evento

COLUNAS_DE_TEMPO = ("frame", "tempo_s")

# Desvio mais longo que isto é considerado grave
DURACAO_GRAVE_S = 2.0


def aprender_faixas(dfs_corretos: list[pd.DataFrame], margem: float = 10.0) -> dict[str, tuple[float, float]]:
    """Faixa esperada de cada ângulo, aprendida de vídeos de execução correta.

    Para cada articulação: do percentil 1 ao 99 dos valores observados, com
    uma margem de tolerância em graus. Os percentis (em vez de mínimo e
    máximo) descartam erros pontuais do detector de pose.
    """
    todos = pd.concat(dfs_corretos, ignore_index=True)
    faixas = {}
    for coluna in todos.columns:
        valores = todos[coluna].dropna()
        if coluna in COLUNAS_DE_TEMPO or valores.empty:
            continue
        minimo = max(0.0, float(np.percentile(valores, 1)) - margem)
        maximo = min(180.0, float(np.percentile(valores, 99)) + margem)
        faixas[coluna] = (minimo, maximo)
    return faixas


def _trechos(mascara: np.ndarray) -> list[tuple[int, int]]:
    """Posições (início, fim inclusive) de cada sequência de valores True."""
    trechos, inicio = [], None
    for i, valor in enumerate(mascara):
        if valor and inicio is None:
            inicio = i
        elif not valor and inicio is not None:
            trechos.append((inicio, i - 1))
            inicio = None
    if inicio is not None:
        trechos.append((inicio, len(mascara) - 1))
    return trechos


def detectar_desvios(
    df: pd.DataFrame,
    faixas: dict[str, tuple[float, float]],
    min_duracao_s: float = 0.5,
) -> list[Evento]:
    """Eventos da fonte "video" para cada trecho com ângulo fora da faixa.

    Trechos mais curtos que min_duracao_s são ignorados (ruído do detector).
    Quadros sem pessoa (ângulo NaN) nunca contam como desvio.
    """
    if len(df) < 2:
        return []
    tempo = df["tempo_s"].to_numpy()
    duracao_quadro = float(np.median(np.diff(tempo)))

    eventos = []
    for articulacao, (minimo, maximo) in faixas.items():
        if articulacao not in df.columns:
            continue
        valores = df[articulacao].to_numpy(dtype=float)
        # comparações com NaN dão False, então quadros sem pessoa ficam de fora
        fora = (valores < minimo) | (valores > maximo)

        for inicio, fim in _trechos(fora):
            duracao = tempo[fim] - tempo[inicio] + duracao_quadro
            if duracao < min_duracao_s:
                continue
            trecho = valores[inicio : fim + 1]
            extremo = trecho.max() if trecho.mean() > maximo else trecho.min()
            eventos.append(
                Evento(
                    fonte="video",
                    tipo="angulo_fora_da_faixa",
                    severidade=3 if duracao > DURACAO_GRAVE_S else 2,
                    instante=float(tempo[inicio]),
                    descricao=(
                        f"{articulacao} fora da faixa esperada ({minimo:.0f}° a {maximo:.0f}°) "
                        f"por {duracao:.1f} s, chegando a {extremo:.0f}°"
                    ),
                )
            )
    return sorted(eventos, key=lambda evento: evento.instante)


# ---------------------------------------------------------------------------
# Amplitude do movimento e padrão individual do paciente
# ---------------------------------------------------------------------------


def amplitude_esperada(
    dfs_corretos: list[pd.DataFrame], articulacao: str, tolerancia: float = 15.0
) -> tuple[float, float]:
    """Faixa esperada para o pico do movimento (o ângulo máximo atingido).

    Vai do menor ao maior pico das execuções corretas, com uma tolerância
    em graus para cada lado.
    """
    picos = [float(df[articulacao].max()) for df in dfs_corretos if df[articulacao].notna().any()]
    return (min(picos) - tolerancia, max(picos) + tolerancia)


def detectar_amplitude(df: pd.DataFrame, articulacao: str, faixa: tuple[float, float]) -> list[Evento]:
    """Evento quando o pico do movimento fica abaixo ou acima do esperado."""
    valores = df[articulacao]
    if valores.isna().all():
        return []
    pico = float(valores.max())
    minimo, maximo = faixa
    if minimo <= pico <= maximo:
        return []
    sentido = "abaixo" if pico < minimo else "acima"
    return [
        Evento(
            fonte="video",
            tipo="amplitude_fora_do_esperado",
            severidade=2,
            instante=float(df.loc[valores.idxmax(), "tempo_s"]),
            descricao=(
                f"Amplitude de {articulacao} {sentido} do esperado: pico de {pico:.0f}° "
                f"(esperado entre {minimo:.0f}° e {maximo:.0f}°)"
            ),
        )
    ]


def aprender_padrao(
    dfs_corretos: list[pd.DataFrame],
    articulacao_principal: str,
    margem: float = 10.0,
    tolerancia: float = 15.0,
) -> dict:
    """Padrão de execução do PRÓPRIO paciente, aprendido de repetições corretas.

    Cada pessoa executa o exercício com amplitude e postura próprias, então
    o padrão é individual: vem das repetições feitas sob supervisão do
    fisioterapeuta. O resultado é um dicionário simples, que pode ser
    gravado em JSON.
    """
    faixas = aprender_faixas(dfs_corretos, margem=margem)
    return {
        "articulacao_principal": articulacao_principal,
        "faixas": {nome: list(faixa) for nome, faixa in faixas.items()},
        "amplitude": list(amplitude_esperada(dfs_corretos, articulacao_principal, tolerancia)),
    }


def avaliar_execucao(df: pd.DataFrame, padrao: dict, min_duracao_s: float = 0.3) -> list[Evento]:
    """Compara uma execução com o padrão do paciente e devolve os desvios encontrados."""
    faixas = {nome: tuple(faixa) for nome, faixa in padrao["faixas"].items()}
    eventos = detectar_desvios(df, faixas, min_duracao_s=min_duracao_s)
    eventos += detectar_amplitude(df, padrao["articulacao_principal"], tuple(padrao["amplitude"]))
    return sorted(eventos, key=lambda evento: evento.instante)
