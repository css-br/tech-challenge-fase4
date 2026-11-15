"""Geração dos áudios de consulta com o Azure Text to Speech.

Os áudios de consulta deste projeto são sintéticos: partem de roteiros
escritos em data/roteiros/ e são convertidos em fala por este módulo.
"""

from pathlib import Path

import azure.cognitiveservices.speech as speechsdk

from src.config import carregar_config

VOZ_PADRAO = "pt-BR-FranciscaNeural"


def sintetizar(texto: str, caminho_wav: Path, voz: str = VOZ_PADRAO) -> Path:
    """Converte o texto em fala e grava um arquivo .wav. Devolve o caminho gravado."""
    config = carregar_config()
    caminho_wav = Path(caminho_wav)
    caminho_wav.parent.mkdir(parents=True, exist_ok=True)

    speech_config = speechsdk.SpeechConfig(subscription=config.speech_key, region=config.speech_region)
    speech_config.speech_synthesis_voice_name = voz
    saida = speechsdk.audio.AudioOutputConfig(filename=str(caminho_wav))

    sintetizador = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=saida)
    resultado = sintetizador.speak_text_async(texto).get()

    if resultado.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        detalhes = resultado.cancellation_details
        raise RuntimeError(f"Falha na síntese de fala: {detalhes.reason} {detalhes.error_details}")
    return caminho_wav
