"""Testes de integração com o Azure Text Analytics e o Azure OpenAI.

Fazem chamadas reais. São pulados quando não há arquivo .env.
Existem porque os dois módulos têm plano B para quando o serviço falha,
e sem estes testes uma falha de conexão passaria despercebida.
"""

from pathlib import Path

import pytest

from src.audio.texto import consultar_azure
from src.fusao.alerta import montar_relatorio_padrao, redigir_com_gpt
from src.fusao.risco import calcular_risco
from src.modelos import Evento

pytestmark = pytest.mark.skipif(
    not Path(".env").exists(), reason="sem .env com as chaves do Azure"
)


def test_text_analytics_responde_com_sentimento_e_frases_chave():
    analise = consultar_azure("Estou com muita dor no peito e falta de ar, estou com medo.")

    assert analise is not None, "o Azure Text Analytics não respondeu"
    assert analise.sentimento == "negative"
    assert analise.pontuacao_negativa > 0.6
    assert analise.frases_chave


def test_gpt_redige_um_relatorio_proprio():
    eventos = [Evento("sinais_vitais", "limite_clinico", 3, 118.0, "SpO2 em 84%")]
    risco = calcular_risco(eventos)

    texto = redigir_com_gpt("P003", risco, eventos)

    assert len(texto) > 50
    assert texto != montar_relatorio_padrao("P003", risco, eventos)
    assert "84" in texto  # usa o dado recebido
