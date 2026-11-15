"""Confere se os três serviços Azure do projeto estão respondendo.

Uso (a partir da raiz do projeto):
    python scripts/testar_azure.py

Saída esperada: uma linha "OK" para cada serviço.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from src.config import OPENAI_API_VERSION, carregar_config

FRASE = "Bom dia, este é um teste do sistema de monitoramento."


def testar_speech(config) -> str:
    """Gera um áudio com a frase de teste e transcreve de volta."""
    speech_config = speechsdk.SpeechConfig(subscription=config.speech_key, region=config.speech_region)
    speech_config.speech_synthesis_voice_name = "pt-BR-FranciscaNeural"
    speech_config.speech_recognition_language = "pt-BR"

    with tempfile.TemporaryDirectory() as pasta:
        caminho = str(Path(pasta) / "teste.wav")

        saida = speechsdk.audio.AudioOutputConfig(filename=caminho)
        sintetizador = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=saida)
        resultado = sintetizador.speak_text_async(FRASE).get()
        if resultado.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"síntese falhou: {resultado.reason}")
        del sintetizador  # libera o arquivo antes de ler

        entrada = speechsdk.audio.AudioConfig(filename=caminho)
        reconhecedor = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=entrada)
        transcricao = reconhecedor.recognize_once_async().get()
        if transcricao.reason != speechsdk.ResultReason.RecognizedSpeech:
            raise RuntimeError(f"transcrição falhou: {transcricao.reason}")
        del reconhecedor
        return transcricao.text


def testar_language(config) -> str:
    """Pede o sentimento de uma frase claramente negativa."""
    cliente = TextAnalyticsClient(config.language_endpoint, AzureKeyCredential(config.language_key))
    resposta = cliente.analyze_sentiment(["Estou com muita dor e falta de ar."], language="pt")[0]
    if resposta.is_error:
        raise RuntimeError(resposta.error.message)
    return resposta.sentiment


def testar_openai(config) -> str:
    """Pede uma resposta curta ao modelo GPT."""
    cliente = AzureOpenAI(
        api_key=config.openai_key,
        azure_endpoint=config.openai_endpoint,
        api_version=OPENAI_API_VERSION,
    )
    resposta = cliente.chat.completions.create(
        model=config.openai_deployment,
        messages=[{"role": "user", "content": "Responda apenas com a palavra: ok"}],
        max_tokens=5,
    )
    return resposta.choices[0].message.content.strip()


def main() -> int:
    config = carregar_config()
    testes = [
        ("Speech (síntese + transcrição)", testar_speech),
        ("Language (sentimento)", testar_language),
        ("OpenAI (GPT)", testar_openai),
    ]
    falhas = 0
    for nome, funcao in testes:
        try:
            print(f"OK    {nome}: {funcao(config)}")
        except Exception as erro:  # o script existe justamente para mostrar o erro
            falhas += 1
            print(f"FALHA {nome}: {erro}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
