import numpy as np
import pandas as pd

from src.audio.voz import avaliar_voz, escolher_limite, extrair_caracteristicas, treinar_classificador

SR = 8000


def _seno(frequencia: float, segundos: float = 2.0) -> np.ndarray:
    t = np.arange(int(SR * segundos)) / SR
    return 0.5 * np.sin(2 * np.pi * frequencia * t)


def _seno_instavel(frequencia: float, segundos: float = 2.0) -> np.ndarray:
    """Tom cuja frequência e volume oscilam, como em uma voz trêmula."""
    gerador = np.random.default_rng(0)
    t = np.arange(int(SR * segundos)) / SR
    # a frequência muda a cada 50 ms
    passos = gerador.uniform(0.85, 1.15, size=int(segundos / 0.05) + 1)
    frequencias = frequencia * np.repeat(passos, int(SR * 0.05))[: len(t)]
    volumes = np.repeat(gerador.uniform(0.2, 0.6, size=len(passos)), int(SR * 0.05))[: len(t)]
    fase = 2 * np.pi * np.cumsum(frequencias) / SR
    return volumes * np.sin(fase)


class ModeloFixo:
    """Modelo de mentira que sempre devolve a mesma probabilidade de voz patológica."""

    def __init__(self, probabilidade: float):
        self.probabilidade = probabilidade

    def predict_proba(self, X):
        return np.array([[1 - self.probabilidade, self.probabilidade]] * len(X))


def test_frequencia_fundamental_de_um_tom_puro():
    caracteristicas = extrair_caracteristicas(_seno(220), SR)

    assert 210 <= caracteristicas["f0_media"] <= 230


def test_tom_puro_tem_jitter_e_shimmer_quase_nulos():
    caracteristicas = extrair_caracteristicas(_seno(220), SR)

    assert caracteristicas["jitter"] < 0.01
    assert caracteristicas["shimmer"] < 0.05


def test_tom_instavel_tem_mais_jitter_e_shimmer_que_tom_puro():
    puro = extrair_caracteristicas(_seno(220), SR)
    instavel = extrair_caracteristicas(_seno_instavel(220), SR)

    assert instavel["jitter"] > puro["jitter"]
    assert instavel["shimmer"] > puro["shimmer"]


def test_silencio_devolve_todos_os_campos_sem_valores_invalidos():
    referencia = extrair_caracteristicas(_seno(220), SR)

    caracteristicas = extrair_caracteristicas(np.zeros(SR * 2), SR)

    assert caracteristicas.keys() == referencia.keys()
    assert all(np.isfinite(valor) for valor in caracteristicas.values())
    assert caracteristicas["f0_media"] == 0.0


def test_treino_devolve_metricas_da_classe_patologica():
    gerador = np.random.default_rng(1)
    saudaveis = pd.DataFrame({"a": gerador.normal(0, 1, 40), "b": gerador.normal(0, 1, 40)})
    patologicas = pd.DataFrame({"a": gerador.normal(5, 1, 40), "b": gerador.normal(5, 1, 40)})
    X = pd.concat([saudaveis, patologicas], ignore_index=True)
    y = pd.Series([0] * 40 + [1] * 40)

    modelo, metricas = treinar_classificador(X, y)

    assert {"recall", "precisao", "f1", "auc", "matriz_confusao", "limite", "falso_alarme"} <= metricas.keys()
    assert metricas["recall"] > 0.9
    assert metricas["auc"] > 0.9
    assert metricas["falso_alarme"] <= 0.2
    assert modelo.predict(pd.DataFrame({"a": [5.0], "b": [5.0]}))[0] == 1


def test_limite_e_o_menor_que_respeita_o_maximo_de_falsos_alarmes():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    probabilidades = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.5, 0.7, 0.95])

    # Com limite 0.6, só uma das cinco vozes saudáveis (0.9) fica acima: 20%.
    # Com limite 0.3 seriam duas (40%), acima do máximo permitido.
    assert escolher_limite(y, probabilidades, max_falso_alarme=0.2) == 0.6


def test_limite_nunca_fica_abaixo_de_meio():
    y = np.array([0, 0, 1, 1])
    probabilidades = np.array([0.05, 0.1, 0.8, 0.9])

    assert escolher_limite(y, probabilidades, max_falso_alarme=0.2) == 0.5


def test_voz_acima_do_limite_gera_evento_leve():
    eventos = avaliar_voz(_seno(220), SR, ModeloFixo(0.9), limite=0.7)

    assert len(eventos) == 1
    assert eventos[0].fonte == "voz"
    assert eventos[0].tipo == "voz_alterada"
    assert eventos[0].severidade == 1


def test_voz_abaixo_do_limite_nao_gera_evento():
    assert avaliar_voz(_seno(220), SR, ModeloFixo(0.6), limite=0.7) == []
