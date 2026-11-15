import numpy as np
import pandas as pd
import pytest

from src.anomalias.sinais_vitais import DetectorSinaisVitais
from src.fusao.monitoramento import monitorar_sinais
from src.modelos import Evento


def _serie(segundos: int = 200, semente: int = 0) -> pd.DataFrame:
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
    detector.treinar([_serie(600, semente=1), _serie(600, semente=2)])
    return detector


class RegistroDeAlertas:
    """Substitui a emissão real: só guarda o que seria alertado."""

    def __init__(self):
        self.alertas = []

    def __call__(self, paciente_id, risco, eventos):
        alerta = {"paciente_id": paciente_id, "nivel": risco.nivel, "n_eventos": len(eventos)}
        self.alertas.append(alerta)
        return alerta


def test_serie_normal_nao_gera_alerta(detector):
    registro = RegistroDeAlertas()
    serie = pd.DataFrame({"tempo_s": np.arange(100.0), "fc": 80.0, "spo2": 97.0, "fr": 16.0, "pas": 120.0})

    resultado = monitorar_sinais("P001", [], serie, detector, ao_alertar=registro)

    assert registro.alertas == []
    assert resultado.risco.nivel == "baixo"


def test_queda_de_oxigenacao_gera_um_unico_alerta_alto(detector):
    registro = RegistroDeAlertas()
    serie = pd.DataFrame({"tempo_s": np.arange(150.0), "fc": 80.0, "spo2": 97.0, "fr": 16.0, "pas": 120.0})
    serie.loc[100:129, "spo2"] = 80.0

    resultado = monitorar_sinais("P003", [], serie, detector, ao_alertar=registro)

    assert [a["nivel"] for a in registro.alertas] == ["alto"]
    assert resultado.risco.nivel == "alto"
    assert any(e.fonte == "sinais_vitais" for e in resultado.eventos)


def test_eventos_anteriores_entram_na_conta_do_risco(detector):
    # Vídeo e texto moderados já somam risco médio (4). Com a queda de oxigenação, sobe para alto.
    registro = RegistroDeAlertas()
    anteriores = [Evento("video", "teste", 2, 0.0, "desvio"), Evento("texto", "teste", 2, 0.0, "termo")]
    serie = pd.DataFrame({"tempo_s": np.arange(150.0), "fc": 80.0, "spo2": 97.0, "fr": 16.0, "pas": 120.0})
    serie.loc[100:129, "spo2"] = 80.0

    resultado = monitorar_sinais("P003", anteriores, serie, detector, nivel_ja_alertado="medio", ao_alertar=registro)

    assert [a["nivel"] for a in registro.alertas] == ["alto"]
    assert registro.alertas[0]["n_eventos"] == 3
    assert len(anteriores) == 2  # a lista recebida não é alterada


def test_nao_repete_alerta_quando_o_nivel_nao_sobe(detector):
    registro = RegistroDeAlertas()
    serie = pd.DataFrame({"tempo_s": np.arange(300.0), "fc": 80.0, "spo2": 97.0, "fr": 16.0, "pas": 120.0})
    serie.loc[100:129, "spo2"] = 80.0   # primeiro episódio grave
    serie.loc[200:229, "fc"] = 160.0    # segundo episódio grave

    monitorar_sinais("P003", [], serie, detector, ao_alertar=registro)

    assert len(registro.alertas) == 1


def test_leitura_invalida_nao_gera_alerta_nem_interrompe(detector):
    registro = RegistroDeAlertas()
    serie = pd.DataFrame({"tempo_s": np.arange(100.0), "fc": 80.0, "spo2": 97.0, "fr": 16.0, "pas": 120.0})
    serie.loc[40:49, "spo2"] = np.nan

    resultado = monitorar_sinais("P001", [], serie, detector, ao_alertar=registro)

    assert registro.alertas == []
    assert [e.fonte for e in resultado.eventos] == ["sensor"]
