"""Transcrição de áudio com o Azure Speech to Text."""

import logging
import threading
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk

from src.config import carregar_config

log = logging.getLogger(__name__)

IDIOMA = "pt-BR"
TEMPO_MAXIMO_S = 300


def transcrever(caminho_wav: Path) -> str:
    """Devolve o texto falado no arquivo .wav.

    Usa o reconhecimento contínuo, que percorre o áudio inteiro. (O modo
    simples, recognize_once, para na primeira pausa e perderia o resto
    de uma consulta com várias frases.)

    Devolve "" quando não há fala reconhecível ou quando o serviço falha,
    para que o restante do fluxo continue funcionando.
    """
    caminho_wav = Path(caminho_wav)
    if not caminho_wav.exists():
        log.warning("Áudio não encontrado: %s", caminho_wav)
        return ""

    try:
        config = carregar_config()
        speech_config = speechsdk.SpeechConfig(subscription=config.speech_key, region=config.speech_region)
        speech_config.speech_recognition_language = IDIOMA
        entrada = speechsdk.audio.AudioConfig(filename=str(caminho_wav))
        reconhecedor = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=entrada)

        trechos: list[str] = []
        terminou = threading.Event()

        def ao_reconhecer(evento):
            if evento.result.reason == speechsdk.ResultReason.RecognizedSpeech and evento.result.text:
                trechos.append(evento.result.text)

        def ao_cancelar(evento):
            if evento.reason == speechsdk.CancellationReason.Error:
                log.warning("Transcrição cancelada pelo serviço: %s", evento.error_details)
            terminou.set()

        reconhecedor.recognized.connect(ao_reconhecer)
        reconhecedor.canceled.connect(ao_cancelar)  # também dispara ao chegar no fim do arquivo
        reconhecedor.session_stopped.connect(lambda evento: terminou.set())

        reconhecedor.start_continuous_recognition()
        terminou.wait(timeout=TEMPO_MAXIMO_S)
        reconhecedor.stop_continuous_recognition()

        return " ".join(trechos)
    except Exception as erro:
        log.warning("Falha ao transcrever %s: %s", caminho_wav, erro)
        return ""
