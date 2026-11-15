"""Detecção da pessoa no vídeo com YOLOv8.

Enquanto o MediaPipe descreve a postura (onde está cada articulação), o
YOLOv8 responde onde a pessoa está no quadro e quantas pessoas aparecem.
Essa trajetória alimenta a análise de movimentação do paciente.
"""

import math
from pathlib import Path

import cv2
import pandas as pd

CAMINHO_MODELO = Path("models/yolov8n.pt")
CLASSE_PESSOA = 0  # índice da classe "person" no YOLOv8 pré-treinado
CONFIANCA_MINIMA = 0.4

COLUNAS = ["frame", "tempo_s", "x_centro", "y_centro", "largura", "altura", "n_pessoas"]


def escolher_pessoa(caixas: list[tuple[float, float, float, float]]) -> int | None:
    """Índice da maior caixa (x1, y1, x2, y2), ou None se não houver caixas.

    Quando há mais de uma pessoa no quadro (por exemplo, o fisioterapeuta
    ao fundo), o paciente é quem ocupa a maior área.
    """
    if not caixas:
        return None
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in caixas]
    return areas.index(max(areas))


def detectar_pessoa(
    caminho_video: Path,
    intervalo: tuple[int, int] | None = None,
    passo: int = 1,
) -> pd.DataFrame:
    """Devolve a posição da pessoa principal em cada quadro analisado.

    `intervalo` limita a um trecho (quadro inicial, quadro final) e `passo`
    analisa um a cada `passo` quadros.

    Colunas: frame, tempo_s, x_centro, y_centro, largura, altura (todas de
    0 a 1, relativas ao tamanho do quadro) e n_pessoas. Quadros sem pessoa
    ficam com as posições em NaN e n_pessoas = 0.
    """
    from ultralytics import YOLO

    captura = cv2.VideoCapture(str(caminho_video))
    if not captura.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho_video}")

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    largura = captura.get(cv2.CAP_PROP_FRAME_WIDTH)
    altura = captura.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    inicio, fim = intervalo if intervalo else (0, total)
    captura.set(cv2.CAP_PROP_POS_FRAMES, inicio)

    CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(str(CAMINHO_MODELO))  # baixa os pesos na primeira execução

    linhas = []
    for frame in range(inicio, fim):
        lido, quadro = captura.read()
        if not lido:
            break
        if (frame - inicio) % passo:
            continue

        resultado = modelo(quadro, classes=[CLASSE_PESSOA], conf=CONFIANCA_MINIMA, verbose=False)[0]
        caixas = [tuple(caixa) for caixa in resultado.boxes.xyxy.tolist()]
        escolhida = escolher_pessoa(caixas)

        linha = {"frame": frame, "tempo_s": (frame - inicio) / fps, "n_pessoas": len(caixas)}
        if escolhida is None:
            linha.update({nome: math.nan for nome in ("x_centro", "y_centro", "largura", "altura")})
        else:
            x1, y1, x2, y2 = caixas[escolhida]
            linha.update({
                "x_centro": (x1 + x2) / 2 / largura,
                "y_centro": (y1 + y2) / 2 / altura,
                "largura": (x2 - x1) / largura,
                "altura": (y2 - y1) / altura,
            })
        linhas.append(linha)

    captura.release()
    return pd.DataFrame(linhas, columns=COLUNAS)
