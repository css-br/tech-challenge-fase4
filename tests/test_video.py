import json
import math

import numpy as np
import pandas as pd
import pytest

from src.video.desvios import (
    amplitude_esperada,
    aprender_faixas,
    aprender_padrao,
    avaliar_execucao,
    detectar_amplitude,
    detectar_desvios,
)
from src.video.detector import detectar_pessoa, escolher_pessoa
from src.video.pose import ANGULOS, angulos_do_corpo, calcular_angulo, extrair_poses

FPS = 30


def _serie(valores) -> pd.DataFrame:
    """Monta a tabela de um vídeo a 30 quadros por segundo com um único ângulo."""
    valores = np.asarray(valores, dtype=float)
    return pd.DataFrame({
        "frame": np.arange(len(valores)),
        "tempo_s": np.arange(len(valores)) / FPS,
        "joelho_esq": valores,
    })


# ---------- calcular_angulo ----------

def test_angulo_de_um_canto_reto():
    assert calcular_angulo((0, 1), (0, 0), (1, 0)) == pytest.approx(90)


def test_angulo_de_pontos_alinhados():
    assert calcular_angulo((-1, 0), (0, 0), (1, 0)) == pytest.approx(180)


def test_angulo_indefinido_quando_dois_pontos_coincidem():
    assert math.isnan(calcular_angulo((0, 0), (0, 0), (1, 0)))


# ---------- angulos_do_corpo ----------

def _pontos_em_pe() -> np.ndarray:
    """33 pontos (x, y, visibilidade) de uma pessoa em pé, de frente, braços ao lado do corpo."""
    pontos = np.zeros((33, 3))
    pontos[:, 2] = 1.0
    posicoes = {
        11: (60, 20), 12: (40, 20),   # ombros (esquerdo, direito)
        13: (60, 40), 14: (40, 40),   # cotovelos
        15: (60, 60), 16: (40, 60),   # punhos
        23: (60, 70), 24: (40, 70),   # quadris
        25: (60, 100), 26: (40, 100), # joelhos
        27: (60, 130), 28: (40, 130), # tornozelos
    }
    for indice, (x, y) in posicoes.items():
        pontos[indice, :2] = (x, y)
    return pontos


def test_pessoa_em_pe_tem_joelhos_e_cotovelos_estendidos():
    angulos = angulos_do_corpo(_pontos_em_pe())

    assert set(angulos) == set(ANGULOS)
    assert angulos["joelho_esq"] == pytest.approx(180)
    assert angulos["cotovelo_dir"] == pytest.approx(180)
    assert angulos["ombro_esq"] == pytest.approx(0, abs=1e-6)  # braço junto ao corpo


def test_articulacao_pouco_visivel_fica_sem_angulo():
    pontos = _pontos_em_pe()
    pontos[25, 2] = 0.1  # joelho esquerdo encoberto

    angulos = angulos_do_corpo(pontos)

    assert math.isnan(angulos["joelho_esq"])
    assert angulos["joelho_dir"] == pytest.approx(180)


# ---------- escolher_pessoa ----------

def test_escolhe_a_pessoa_de_maior_caixa():
    caixas = [(0, 0, 10, 10), (20, 20, 80, 90), (5, 5, 30, 30)]

    assert escolher_pessoa(caixas) == 1


def test_sem_caixas_nao_ha_pessoa():
    assert escolher_pessoa([]) is None


# ---------- aprender_faixas ----------

def test_faixa_vai_do_percentil_1_ao_99_com_margem():
    correto = _serie(np.arange(50, 151))  # 50, 51, ..., 150

    faixas = aprender_faixas([correto], margem=10)

    assert faixas["joelho_esq"] == pytest.approx((41, 159))


def test_faixa_ignora_quadros_sem_angulo_e_junta_varios_videos():
    video_a = _serie([60, 70, np.nan, 80])
    video_b = _serie([np.nan, 100, 120])

    minimo, maximo = aprender_faixas([video_a, video_b], margem=0)["joelho_esq"]

    assert 60 <= minimo <= 61
    assert 119 <= maximo <= 120


def test_faixa_fica_dentro_de_0_a_180_graus():
    correto = _serie([2, 90, 178])

    assert aprender_faixas([correto], margem=10)["joelho_esq"] == (0, 180)


# ---------- detectar_desvios ----------

FAIXAS = {"joelho_esq": (80.0, 120.0)}


def test_trecho_de_um_segundo_fora_da_faixa_gera_um_evento_moderado():
    valores = [100] * 60 + [150] * 30 + [100] * 60  # fora da faixa entre 2 s e 3 s

    eventos = detectar_desvios(_serie(valores), FAIXAS)

    assert len(eventos) == 1
    assert eventos[0].fonte == "video"
    assert eventos[0].tipo == "angulo_fora_da_faixa"
    assert eventos[0].severidade == 2
    assert eventos[0].instante == pytest.approx(2.0)
    assert "joelho_esq" in eventos[0].descricao


def test_pico_de_um_decimo_de_segundo_e_ignorado():
    valores = [100] * 60 + [150] * 3 + [100] * 60

    assert detectar_desvios(_serie(valores), FAIXAS) == []


def test_trecho_longo_fora_da_faixa_e_grave():
    valores = [100] * 30 + [40] * 90 + [100] * 30  # 3 s abaixo do mínimo

    eventos = detectar_desvios(_serie(valores), FAIXAS)

    assert [e.severidade for e in eventos] == [3]


def test_quadros_sem_pessoa_nao_geram_evento_nem_erro():
    valores = [100] * 30 + [np.nan] * 90 + [100] * 30

    assert detectar_desvios(_serie(valores), FAIXAS) == []


def test_dois_trechos_separados_geram_dois_eventos():
    valores = [100] * 30 + [150] * 30 + [100] * 60 + [150] * 30 + [100] * 30

    eventos = detectar_desvios(_serie(valores), FAIXAS)

    assert [round(e.instante) for e in eventos] == [1, 4]


def test_video_inteiro_sem_pessoa_devolve_lista_vazia():
    assert detectar_desvios(_serie([np.nan] * 90), FAIXAS) == []


# ---------- amplitude do movimento ----------

def test_amplitude_esperada_vem_dos_picos_das_execucoes_corretas():
    corretas = [_serie([20, 90, 150, 90, 20]), _serie([20, 80, 140, 80, 20])]

    assert amplitude_esperada(corretas, "joelho_esq", tolerancia=15) == (125, 165)


def test_movimento_com_amplitude_menor_que_a_esperada_gera_evento():
    execucao = _serie([20] * 30 + [110] * 30 + [20] * 30)  # pico de 110 graus a partir de 1 s

    eventos = detectar_amplitude(execucao, "joelho_esq", (125, 165))

    assert [(e.fonte, e.tipo, e.severidade) for e in eventos] == [("video", "amplitude_fora_do_esperado", 2)]
    assert eventos[0].instante == pytest.approx(1.0)
    assert "abaixo" in eventos[0].descricao


def test_movimento_com_amplitude_maior_que_a_esperada_gera_evento():
    eventos = detectar_amplitude(_serie([20, 100, 178, 100, 20]), "joelho_esq", (125, 165))

    assert len(eventos) == 1
    assert "acima" in eventos[0].descricao


def test_amplitude_dentro_do_esperado_nao_gera_evento():
    assert detectar_amplitude(_serie([20, 100, 150, 100, 20]), "joelho_esq", (125, 165)) == []


def test_amplitude_de_video_sem_pessoa_nao_gera_evento():
    assert detectar_amplitude(_serie([np.nan] * 30), "joelho_esq", (125, 165)) == []


# ---------- padrão individual do paciente ----------

def test_padrao_pode_ser_gravado_em_json_e_usado_depois():
    corretas = [_serie([100] * 30 + [150] * 30 + [100] * 30), _serie([100] * 30 + [145] * 30 + [100] * 30)]

    padrao = json.loads(json.dumps(aprender_padrao(corretas, "joelho_esq", margem=10, tolerancia=15)))

    assert padrao["articulacao_principal"] == "joelho_esq"
    assert avaliar_execucao(corretas[0], padrao) == []


def test_execucao_fora_do_padrao_do_paciente_gera_eventos_de_faixa_e_de_amplitude():
    corretas = [_serie([100] * 30 + [150] * 30 + [100] * 30), _serie([100] * 30 + [145] * 30 + [100] * 30)]
    padrao = aprender_padrao(corretas, "joelho_esq", margem=10, tolerancia=15)
    # começa dobrado demais (60 graus por 1 s) e não chega à amplitude habitual (pico de 110)
    execucao = _serie([60] * 30 + [110] * 30 + [100] * 30)

    tipos = sorted(e.tipo for e in avaliar_execucao(execucao, padrao))

    assert tipos == ["amplitude_fora_do_esperado", "angulo_fora_da_faixa"]


# ---------- vídeo real (clipe de amostra do REHAB24-6) ----------

CLIPE = "data/amostras/fisio_correta.mp4"


def test_extrai_angulos_de_um_video_real():
    angulos = extrair_poses(CLIPE, passo=10)

    assert list(angulos.columns) == ["frame", "tempo_s", *ANGULOS]
    assert angulos["frame"].tolist()[:3] == [0, 10, 20]
    assert angulos["ombro_dir"].notna().mean() > 0.9
    assert angulos["ombro_dir"].max() > 150  # o braço direito sobe bem acima da horizontal


def test_detecta_a_pessoa_em_um_video_real():
    trajetoria = detectar_pessoa(CLIPE, passo=15)

    assert (trajetoria["n_pessoas"] >= 1).all()
    assert trajetoria["x_centro"].between(0, 1).all()
    assert trajetoria["altura"].min() > 0.4  # a pessoa ocupa boa parte da altura do quadro


def test_video_inexistente_gera_erro_claro():
    with pytest.raises(FileNotFoundError):
        extrair_poses("data/amostras/nao_existe.mp4")
