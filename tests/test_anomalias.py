import numpy as np
import pandas as pd
import pytest

from src.anomalias.movimentacao import detectar_anomalias_movimentacao
from src.anomalias.prescricoes import detectar_anomalias_prescricao
from src.anomalias.sinais_vitais import DetectorSinaisVitais, carregar_sinais, injetar_anomalias

# ====================== sinais vitais ======================


def _sinais_normais(segundos: int = 600, semente: int = 0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    return pd.DataFrame({
        "tempo_s": np.arange(segundos, dtype=float),
        "fc": 80 + gerador.normal(0, 2, segundos),
        "spo2": 97 + gerador.normal(0, 0.5, segundos),
        "fr": 16 + gerador.normal(0, 1, segundos),
        "pas": 120 + gerador.normal(0, 3, segundos),
    })


@pytest.fixture(scope="module")
def detector() -> DetectorSinaisVitais:
    detector = DetectorSinaisVitais()
    detector.treinar([_sinais_normais(semente=0), _sinais_normais(semente=1)])
    return detector


NORMAL = {"fc": 80.0, "spo2": 97.0, "fr": 16.0, "pas": 120.0}


def test_amostra_normal_nao_gera_evento(detector):
    detector.reiniciar()

    eventos = [e for t in range(40) for e in detector.avaliar_amostra(NORMAL, float(t))]

    assert eventos == []


def test_queda_de_oxigenacao_gera_evento_grave(detector):
    detector.reiniciar()
    for t in range(40):
        detector.avaliar_amostra(NORMAL, float(t))

    eventos = detector.avaliar_amostra({**NORMAL, "spo2": 82.0}, 40.0)

    assert len(eventos) == 1
    assert eventos[0].fonte == "sinais_vitais"
    assert eventos[0].severidade == 3
    assert eventos[0].instante == 40.0
    assert "spo2" in eventos[0].descricao.lower()


def test_alteracao_moderada_e_detectada_pelo_modelo(detector):
    # FC de 115 está dentro dos limites clínicos, mas é muito diferente do padrão aprendido
    detector.reiniciar()
    for t in range(40):
        detector.avaliar_amostra(NORMAL, float(t))

    eventos = [e for t in range(40, 50) for e in detector.avaliar_amostra({**NORMAL, "fc": 115.0}, float(t))]

    assert len(eventos) >= 1
    assert eventos[0].severidade == 2


def test_episodio_continuo_gera_um_unico_evento(detector):
    detector.reiniciar()
    for t in range(40):
        detector.avaliar_amostra(NORMAL, float(t))

    eventos = [e for t in range(40, 70) for e in detector.avaliar_amostra({**NORMAL, "spo2": 82.0}, float(t))]

    assert len(eventos) == 1


@pytest.mark.parametrize("leitura", [{"spo2": 0.0}, {"fc": float("nan")}, {"pas": None}])
def test_leitura_invalida_vira_evento_de_sensor(detector, leitura):
    detector.reiniciar()

    eventos = detector.avaliar_amostra({**NORMAL, **leitura}, 5.0)

    assert [(e.fonte, e.tipo) for e in eventos] == [("sensor", "leitura_invalida")]


def test_leitura_invalida_nao_contamina_as_amostras_seguintes(detector):
    detector.reiniciar()
    for t in range(40):
        detector.avaliar_amostra(NORMAL, float(t))
    detector.avaliar_amostra({**NORMAL, "spo2": 0.0}, 40.0)

    eventos = [e for t in range(41, 60) for e in detector.avaliar_amostra(NORMAL, float(t))]

    assert eventos == []


def test_avaliar_serie_marca_o_trecho_anomalo(detector):
    serie = _sinais_normais(segundos=200, semente=5)
    serie.loc[100:129, "spo2"] = 80.0

    marcado = detector.avaliar_serie(serie)

    assert marcado.loc[100:129].mean() > 0.9
    assert marcado.loc[0:89].mean() < 0.1


def test_injecao_marca_os_trechos_alterados_e_e_reprodutivel():
    original = _sinais_normais(segundos=480, semente=3)

    a = injetar_anomalias(original, semente=42)
    b = injetar_anomalias(original, semente=42)

    pd.testing.assert_frame_equal(a, b)
    assert a["anomalia_real"].dtype == bool
    assert 60 <= a["anomalia_real"].sum() <= 120  # três trechos de 20 a 40 s
    alterado = (a[["fc", "spo2", "pas"]] != original[["fc", "spo2", "pas"]]).any(axis=1)
    assert (alterado == a["anomalia_real"]).all()
    assert "anomalia_real" not in original.columns  # não altera a tabela recebida


def test_carrega_csv_do_bidmc_com_pressao_simulada(tmp_path):
    caminho = tmp_path / "bidmc_01_Numerics.csv"
    caminho.write_text("Time [s], HR, PULSE, RESP, SpO2\n0,94,93,25,97\n1,95,93,24,\n2,96,94,25,98\n")

    sinais = carregar_sinais(caminho)

    assert list(sinais.columns) == ["tempo_s", "fc", "spo2", "fr", "pas"]
    assert sinais["fc"].tolist() == [94, 95, 96]
    assert np.isnan(sinais.loc[1, "spo2"])
    assert sinais["pas"].between(80, 160).all()
    pd.testing.assert_frame_equal(sinais, carregar_sinais(caminho))  # pressão simulada é reprodutível


# ====================== prescrições ======================


def _prescricoes(linhas) -> pd.DataFrame:
    df = pd.DataFrame(linhas, columns=["paciente_id", "data_hora", "medicamento", "dose_mg"])
    df["data_hora"] = pd.to_datetime(df["data_hora"])
    return df


def test_tratamento_estavel_nao_gera_evento():
    df = _prescricoes([
        ("P1", "2026-01-01 08:00", "losartana", 50),
        ("P1", "2026-01-02 08:00", "losartana", 50),
        ("P1", "2026-01-03 08:00", "losartana", 50),
    ])

    assert detectar_anomalias_prescricao(df) == []


def test_aumento_de_dose_acima_de_50_por_cento_gera_evento():
    df = _prescricoes([
        ("P1", "2026-01-01 08:00", "losartana", 50),
        ("P1", "2026-01-02 08:00", "losartana", 80),
    ])

    eventos = detectar_anomalias_prescricao(df)

    assert [(e.fonte, e.tipo, e.severidade) for e in eventos] == [("prescricao", "mudanca_brusca_de_dose", 2)]
    assert eventos[0].instante == 24 * 3600
    assert "losartana" in eventos[0].descricao


def test_dose_mais_que_dobrada_e_grave():
    df = _prescricoes([
        ("P1", "2026-01-01 08:00", "morfina", 10),
        ("P1", "2026-01-01 20:00", "morfina", 30),
    ])

    assert [e.severidade for e in detectar_anomalias_prescricao(df)] == [3]


def test_reducao_brusca_tambem_gera_evento():
    df = _prescricoes([
        ("P1", "2026-01-01 08:00", "prednisona", 40),
        ("P1", "2026-01-02 08:00", "prednisona", 10),
    ])

    assert [e.tipo for e in detectar_anomalias_prescricao(df)] == ["mudanca_brusca_de_dose"]


def test_ajuste_pequeno_de_dose_nao_gera_evento():
    df = _prescricoes([
        ("P1", "2026-01-01 08:00", "losartana", 50),
        ("P1", "2026-01-02 08:00", "losartana", 60),
    ])

    assert detectar_anomalias_prescricao(df) == []


def test_dois_medicamentos_novos_em_menos_de_24_horas_geram_evento():
    df = _prescricoes([
        ("P1", "2026-01-01 08:00", "losartana", 50),
        ("P1", "2026-01-03 08:00", "amiodarona", 200),
        ("P1", "2026-01-03 18:00", "varfarina", 5),
    ])

    eventos = detectar_anomalias_prescricao(df)

    assert [(e.tipo, e.severidade) for e in eventos] == [("troca_inesperada_de_tratamento", 2)]
    assert "amiodarona" in eventos[0].descricao and "varfarina" in eventos[0].descricao


def test_medicamentos_da_primeira_prescricao_nao_contam_como_novos():
    df = _prescricoes([
        ("P1", "2026-01-01 08:00", "losartana", 50),
        ("P1", "2026-01-01 08:00", "sinvastatina", 20),
        ("P1", "2026-01-01 09:00", "omeprazol", 20),
    ])

    assert detectar_anomalias_prescricao(df) == []


def test_pacientes_sao_analisados_separadamente():
    df = _prescricoes([
        ("P1", "2026-01-01 08:00", "losartana", 50),
        ("P2", "2026-01-02 08:00", "losartana", 100),
    ])

    assert detectar_anomalias_prescricao(df) == []


# ====================== movimentação ======================

FPS = 10


def _trajetoria(x, y=None, pessoas=None) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    y = np.full(len(x), 0.5) if y is None else np.asarray(y, dtype=float)
    pessoas = np.ones(len(x), dtype=int) if pessoas is None else np.asarray(pessoas)
    return pd.DataFrame({
        "frame": np.arange(len(x)),
        "tempo_s": np.arange(len(x)) / FPS,
        "x_centro": np.where(pessoas > 0, x, np.nan),
        "y_centro": np.where(pessoas > 0, y, np.nan),
        "largura": 0.2,
        "altura": 0.6,
        "n_pessoas": pessoas,
    })


def _andando(segundos: float) -> np.ndarray:
    """Vai e volta entre x = 0.3 e x = 0.7."""
    t = np.arange(int(segundos * FPS)) / FPS
    return 0.5 + 0.2 * np.sin(2 * np.pi * t / 4)


def test_paciente_em_movimento_nao_gera_evento():
    assert detectar_anomalias_movimentacao(_trajetoria(_andando(30))) == []


def test_quinze_segundos_parado_gera_imobilidade():
    x = np.concatenate([_andando(8), np.full(15 * FPS, 0.5), _andando(8)])

    eventos = detectar_anomalias_movimentacao(_trajetoria(x), imobilidade_s=10)

    assert [(e.fonte, e.tipo) for e in eventos] == [("movimentacao", "imobilidade")]
    assert eventos[0].instante == pytest.approx(8.0, abs=0.5)


def test_parada_curta_nao_e_imobilidade():
    x = np.concatenate([_andando(8), np.full(5 * FPS, 0.5), _andando(8)])

    assert detectar_anomalias_movimentacao(_trajetoria(x), imobilidade_s=10) == []


def test_centro_fora_da_zona_gera_evento():
    x = np.concatenate([_andando(8), np.linspace(0.7, 0.99, 10), np.full(3 * FPS, 0.99)])

    eventos = detectar_anomalias_movimentacao(_trajetoria(x), imobilidade_s=10)

    assert [e.tipo for e in eventos] == ["fora_da_area"]


def test_mais_de_cinco_segundos_sem_pessoa_gera_paciente_ausente():
    x = _andando(20)
    pessoas = np.ones(len(x), dtype=int)
    pessoas[50:120] = 0  # 7 segundos sem ninguém

    eventos = detectar_anomalias_movimentacao(_trajetoria(x, pessoas=pessoas))

    assert [e.tipo for e in eventos] == ["paciente_ausente"]
    assert eventos[0].instante == pytest.approx(5.0)


def test_ausencia_curta_nao_gera_evento():
    x = _andando(20)
    pessoas = np.ones(len(x), dtype=int)
    pessoas[50:70] = 0  # 2 segundos

    assert detectar_anomalias_movimentacao(_trajetoria(x, pessoas=pessoas)) == []


def test_trajetoria_vazia_devolve_lista_vazia():
    assert detectar_anomalias_movimentacao(_trajetoria([])) == []
