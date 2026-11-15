"""Testes de integração com o Azure Speech.

Fazem chamadas reais ao serviço. São pulados quando não há arquivo .env.
"""

import wave
from pathlib import Path

import pytest

from src.audio.sintese import sintetizar
from src.audio.transcricao import transcrever

pytestmark = pytest.mark.skipif(
    not Path(".env").exists(), reason="sem .env com as chaves do Azure"
)


def _gravar_silencio(caminho: Path, segundos: float = 2.0) -> None:
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(16000)
        arquivo.writeframes(b"\x00\x00" * int(16000 * segundos))


def test_audio_sintetizado_e_transcrito_de_volta(tmp_path):
    caminho = tmp_path / "fala.wav"

    sintetizar("O paciente relata dor no peito.", caminho)
    texto = transcrever(caminho)

    assert caminho.stat().st_size > 1000
    assert "dor no peito" in texto.lower()


def test_transcreve_audio_com_mais_de_uma_frase(tmp_path):
    caminho = tmp_path / "longo.wav"
    frases = "Primeiro eu senti tontura. Depois de alguns minutos veio a falta de ar. No fim da tarde tive febre."

    sintetizar(frases, caminho)
    texto = transcrever(caminho).lower()

    assert "tontura" in texto
    assert "febre" in texto


def test_audio_em_silencio_devolve_texto_vazio(tmp_path):
    caminho = tmp_path / "silencio.wav"
    _gravar_silencio(caminho)

    assert transcrever(caminho) == ""


def test_arquivo_inexistente_devolve_texto_vazio(tmp_path):
    assert transcrever(tmp_path / "nao_existe.wav") == ""
