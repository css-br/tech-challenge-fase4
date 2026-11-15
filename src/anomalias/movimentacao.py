"""Detecção de padrões anômalos de movimentação do paciente.

Trabalha sobre a trajetória produzida por src/video/detector.py (posição
da pessoa em cada quadro, vinda do YOLOv8) e procura três situações:
imobilidade prolongada, saída da área esperada e ausência do paciente.
"""

import numpy as np
import pandas as pd

from src.modelos import Evento

TOLERANCIA_PARADO = 0.02   # deslocamento (fração do quadro) ainda considerado "parado"
MIN_FORA_DA_AREA_S = 1.0


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


def _imobilidade(tempo, x, y, duracao_quadro, imobilidade_s) -> list[Evento]:
    eventos, i, n = [], 0, len(tempo)
    while i < n:
        if np.isnan(x[i]):
            i += 1
            continue
        # avança enquanto a pessoa permanece perto do ponto onde estava no quadro i
        j = i
        while j + 1 < n and np.hypot(x[j + 1] - x[i], y[j + 1] - y[i]) <= TOLERANCIA_PARADO:
            j += 1
        duracao = tempo[j] - tempo[i] + duracao_quadro
        if duracao >= imobilidade_s:
            eventos.append(Evento(
                fonte="movimentacao",
                tipo="imobilidade",
                severidade=2,
                instante=float(tempo[i]),
                descricao=f"Paciente sem se mover por {duracao:.0f} s",
            ))
            i = j + 1
        else:
            i += 1
    return eventos


def detectar_anomalias_movimentacao(
    df_traj: pd.DataFrame,
    imobilidade_s: float = 10.0,
    zona: tuple[float, float, float, float] = (0.05, 0.05, 0.95, 0.95),
    ausencia_s: float = 5.0,
) -> list[Evento]:
    """Eventos da fonte "movimentacao".

    - zona: (x mínimo, y mínimo, x máximo, y máximo) da área esperada, de 0 a 1.
    - imobilidade_s: tempo parado a partir do qual há evento.
    - ausencia_s: tempo sem ninguém no quadro a partir do qual há evento.
    """
    if len(df_traj) < 2:
        return []

    tempo = df_traj["tempo_s"].to_numpy(dtype=float)
    x = df_traj["x_centro"].to_numpy(dtype=float)
    y = df_traj["y_centro"].to_numpy(dtype=float)
    duracao_quadro = float(np.median(np.diff(tempo)))

    eventos = _imobilidade(tempo, x, y, duracao_quadro, imobilidade_s)

    x_min, y_min, x_max, y_max = zona
    fora = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)  # NaN dá False
    for inicio, fim in _trechos(fora):
        duracao = tempo[fim] - tempo[inicio] + duracao_quadro
        if duracao >= MIN_FORA_DA_AREA_S:
            eventos.append(Evento("movimentacao", "fora_da_area", 2, float(tempo[inicio]),
                                  f"Paciente fora da área esperada por {duracao:.0f} s"))

    ausente = df_traj["n_pessoas"].to_numpy() == 0
    for inicio, fim in _trechos(ausente):
        duracao = tempo[fim] - tempo[inicio] + duracao_quadro
        if duracao > ausencia_s:
            eventos.append(Evento("movimentacao", "paciente_ausente", 2, float(tempo[inicio]),
                                  f"Nenhuma pessoa detectada por {duracao:.0f} s"))

    return sorted(eventos, key=lambda evento: evento.instante)
