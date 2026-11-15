import pytest

from src.config import VARIAVEIS, carregar_config


def _ambiente_completo() -> dict[str, str]:
    return {nome: f"valor-{nome}" for nome in VARIAVEIS}


def test_carrega_todas_as_variaveis():
    config = carregar_config(_ambiente_completo())

    assert config.speech_key == "valor-AZURE_SPEECH_KEY"
    assert config.speech_region == "valor-AZURE_SPEECH_REGION"
    assert config.language_endpoint == "valor-AZURE_LANGUAGE_ENDPOINT"
    assert config.openai_deployment == "valor-AZURE_OPENAI_DEPLOYMENT"


def test_erro_informa_qual_variavel_falta():
    ambiente = _ambiente_completo()
    del ambiente["AZURE_LANGUAGE_KEY"]

    with pytest.raises(RuntimeError, match="AZURE_LANGUAGE_KEY"):
        carregar_config(ambiente)


def test_variavel_vazia_conta_como_faltando():
    ambiente = _ambiente_completo()
    ambiente["AZURE_SPEECH_KEY"] = "  "

    with pytest.raises(RuntimeError, match="AZURE_SPEECH_KEY"):
        carregar_config(ambiente)
