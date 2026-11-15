from src.modelos import Evento, Risco


def test_evento_guarda_os_campos_informados():
    evento = Evento(
        fonte="sinais_vitais",
        tipo="queda_oxigenacao",
        severidade=3,
        instante=12.5,
        descricao="SpO2 em 82%",
    )

    assert evento.fonte == "sinais_vitais"
    assert evento.severidade == 3
    assert evento.instante == 12.5


def test_risco_guarda_pontuacao_nivel_e_fontes():
    risco = Risco(pontuacao=5, nivel="alto", fontes=["video", "texto"])

    assert risco.nivel == "alto"
    assert risco.fontes == ["video", "texto"]
