"""Análise postural com MediaPipe Pose.

O MediaPipe localiza 33 pontos do corpo em cada quadro do vídeo. A partir
deles calculamos os ângulos das principais articulações, que são a base
para detectar movimentos fora do padrão.

Observação: as aulas usam a interface antiga (mp.solutions.pose), que foi
removida das versões atuais do MediaPipe. Aqui usamos a interface atual
(PoseLandmarker). O modelo e os 33 pontos são os mesmos.
"""

import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CAMINHO_MODELO = Path("models/pose_landmarker_full.task")
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# Abaixo desta visibilidade (0 a 1) o ponto é considerado encoberto
VISIBILIDADE_MINIMA = 0.5

# Nome do ângulo -> índices dos três pontos (o do meio é o vértice).
# Índices do MediaPipe: 11/12 ombros, 13/14 cotovelos, 15/16 punhos,
# 23/24 quadris, 25/26 joelhos, 27/28 tornozelos (ímpar = esquerdo).
ANGULOS = {
    "cotovelo_esq": (11, 13, 15),
    "cotovelo_dir": (12, 14, 16),
    "ombro_esq": (23, 11, 13),
    "ombro_dir": (24, 12, 14),
    "quadril_esq": (11, 23, 25),
    "quadril_dir": (12, 24, 26),
    "joelho_esq": (23, 25, 27),
    "joelho_dir": (24, 26, 28),
}

# Ligações desenhadas no vídeo anotado
LIGACOES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]


def calcular_angulo(a, b, c) -> float:
    """Ângulo em graus formado pelos pontos a-b-c, com vértice em b.

    Exemplo: quadril-joelho-tornozelo dá o ângulo do joelho
    (180 = perna estendida, 90 = joelho dobrado em ângulo reto).
    """
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    v1, v2 = a - b, c - b
    tamanho = np.linalg.norm(v1) * np.linalg.norm(v2)
    if tamanho == 0:
        return math.nan
    cosseno = np.clip(np.dot(v1, v2) / tamanho, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosseno)))


def angulos_do_corpo(pontos: np.ndarray) -> dict[str, float]:
    """Calcula os ângulos das articulações a partir dos 33 pontos de um quadro.

    `pontos` tem uma linha por ponto: (x, y, visibilidade), com x e y em pixels.
    Se algum dos três pontos de um ângulo está encoberto, o ângulo fica NaN.
    """
    angulos = {}
    for nome, (i, j, k) in ANGULOS.items():
        if min(pontos[i, 2], pontos[j, 2], pontos[k, 2]) < VISIBILIDADE_MINIMA:
            angulos[nome] = math.nan
        else:
            angulos[nome] = calcular_angulo(pontos[i, :2], pontos[j, :2], pontos[k, :2])
    return angulos


def garantir_modelo() -> Path:
    """Baixa o arquivo do modelo de pose na primeira execução."""
    if not CAMINHO_MODELO.exists():
        CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL_MODELO, timeout=120) as resposta:
            CAMINHO_MODELO.write_bytes(resposta.read())
    return CAMINHO_MODELO


def _desenhar(quadro: np.ndarray, pontos: np.ndarray) -> None:
    for i, j in LIGACOES:
        if min(pontos[i, 2], pontos[j, 2]) >= VISIBILIDADE_MINIMA:
            p1 = (int(pontos[i, 0]), int(pontos[i, 1]))
            p2 = (int(pontos[j, 0]), int(pontos[j, 1]))
            cv2.line(quadro, p1, p2, (0, 255, 0), 2)
    for i in {indice for par in LIGACOES for indice in par}:
        if pontos[i, 2] >= VISIBILIDADE_MINIMA:
            cv2.circle(quadro, (int(pontos[i, 0]), int(pontos[i, 1])), 4, (0, 0, 255), -1)


def extrair_poses(
    caminho_video: Path,
    saida_anotada: Path | None = None,
    intervalo: tuple[int, int] | None = None,
    passo: int = 1,
) -> pd.DataFrame:
    """Percorre o vídeo e devolve os ângulos das articulações em cada quadro.

    - saida_anotada: se informado, grava um vídeo com o esqueleto desenhado.
    - intervalo: (quadro inicial, quadro final) para analisar só um trecho.
    - passo: analisa um a cada `passo` quadros (3 = três vezes mais rápido).

    Colunas: frame, tempo_s e um ângulo por articulação (ver ANGULOS).
    Quadros sem pessoa detectada ficam com os ângulos em NaN.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    captura = cv2.VideoCapture(str(caminho_video))
    if not captura.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho_video}")

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    inicio, fim = intervalo if intervalo else (0, total)
    captura.set(cv2.CAP_PROP_POS_FRAMES, inicio)

    gravador = None
    if saida_anotada:
        Path(saida_anotada).parent.mkdir(parents=True, exist_ok=True)
        gravador = cv2.VideoWriter(
            str(saida_anotada), cv2.VideoWriter_fourcc(*"mp4v"), fps / passo, (largura, altura)
        )

    opcoes = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(garantir_modelo())),
        running_mode=vision.RunningMode.VIDEO,
    )
    linhas = []
    with vision.PoseLandmarker.create_from_options(opcoes) as detector:
        for frame in range(inicio, fim):
            lido, quadro = captura.read()
            if not lido:
                break
            if (frame - inicio) % passo:
                continue

            imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
            resultado = detector.detect_for_video(imagem, int((frame - inicio) * 1000 / fps))

            linha = {"frame": frame, "tempo_s": (frame - inicio) / fps}
            if resultado.pose_landmarks:
                pontos = np.array([
                    (p.x * largura, p.y * altura, p.visibility) for p in resultado.pose_landmarks[0]
                ])
                linha.update(angulos_do_corpo(pontos))
                if gravador:
                    _desenhar(quadro, pontos)
            else:
                linha.update({nome: math.nan for nome in ANGULOS})
            linhas.append(linha)

            if gravador:
                gravador.write(quadro)

    captura.release()
    if gravador:
        gravador.release()
    return pd.DataFrame(linhas, columns=["frame", "tempo_s", *ANGULOS])
