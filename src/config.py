"""Leitura das chaves e endereços dos serviços Azure a partir do arquivo .env."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

# Nome da variável de ambiente -> nome do campo em Config
VARIAVEIS = {
    "AZURE_SPEECH_KEY": "speech_key",
    "AZURE_SPEECH_REGION": "speech_region",
    "AZURE_LANGUAGE_KEY": "language_key",
    "AZURE_LANGUAGE_ENDPOINT": "language_endpoint",
    "AZURE_OPENAI_KEY": "openai_key",
    "AZURE_OPENAI_ENDPOINT": "openai_endpoint",
    "AZURE_OPENAI_DEPLOYMENT": "openai_deployment",
}

# Versão da API do Azure OpenAI usada pelo projeto
OPENAI_API_VERSION = "2024-10-21"


@dataclass
class Config:
    speech_key: str
    speech_region: str
    language_key: str
    language_endpoint: str
    openai_key: str
    openai_endpoint: str
    openai_deployment: str


def carregar_config(ambiente: Mapping[str, str] | None = None) -> Config:
    """Monta a configuração a partir das variáveis de ambiente.

    Sem argumento, lê o arquivo .env da raiz do projeto. O parâmetro
    `ambiente` existe para os testes poderem passar um dicionário.
    """
    if ambiente is None:
        load_dotenv()
        ambiente = os.environ

    valores = {nome: ambiente.get(nome, "").strip() for nome in VARIAVEIS}
    faltando = [nome for nome, valor in valores.items() if not valor]
    if faltando:
        raise RuntimeError(
            "Variáveis ausentes no .env: " + ", ".join(faltando)
            + ". Copie env.template.txt para .env e preencha."
        )

    return Config(**{VARIAVEIS[nome]: valor for nome, valor in valores.items()})
