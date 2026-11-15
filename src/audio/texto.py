"""Análise do texto transcrito de uma consulta.

Duas análises complementares:
1. Termos críticos: lista própria de sintomas, procurada direto no texto.
2. Azure Text Analytics: sentimento e frases-chave.

A primeira funciona sem internet. Se o Azure falhar, a análise continua
só com os termos críticos.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass

from src.modelos import Evento

log = logging.getLogger(__name__)

# Termo -> severidade (3 = grave, 2 = moderada, 1 = leve)
TERMOS_CRITICOS = {
    "falta de ar": 3,
    "dor no peito": 3,
    "desmaio": 3,
    "sangramento": 3,
    "tontura": 2,
    "confusão": 2,
    "palpitação": 2,
    "febre": 2,
    "vômito": 2,
    "formigamento": 2,
    "fraqueza": 2,
    "cansaço": 1,
    "fadiga": 1,
    "insônia": 1,
    "dor de cabeça": 1,
}

# Acima desta pontuação negativa (0 a 1) o sentimento vira um evento
LIMITE_SENTIMENTO_NEGATIVO = 0.6


@dataclass
class AnaliseAzure:
    sentimento: str            # "positive", "neutral", "negative" ou "mixed"
    pontuacao_negativa: float  # 0 a 1
    frases_chave: list[str]


def _normalizar(texto: str) -> str:
    """Minúsculas e sem acentos, para comparar 'Confusão' com 'confusao'."""
    sem_acentos = unicodedata.normalize("NFD", texto)
    sem_acentos = "".join(c for c in sem_acentos if unicodedata.category(c) != "Mn")
    return sem_acentos.lower()


def encontrar_termos_criticos(texto: str) -> list[str]:
    """Devolve os termos críticos presentes no texto, na ordem em que aparecem."""
    normalizado = _normalizar(texto)
    encontrados = []
    for termo in TERMOS_CRITICOS:
        # \b garante palavra inteira: "desmaio" não casa com "desmaiozinho"
        achado = re.search(rf"\b{re.escape(_normalizar(termo))}\b", normalizado)
        if achado:
            encontrados.append((achado.start(), termo))
    return [termo for _, termo in sorted(encontrados)]


def criar_cliente():
    """Cria o cliente do Azure Text Analytics com as chaves do .env."""
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    from src.config import carregar_config

    config = carregar_config()
    return TextAnalyticsClient(config.language_endpoint, AzureKeyCredential(config.language_key))


def consultar_azure(texto: str, cliente=None) -> AnaliseAzure | None:
    """Pede sentimento e frases-chave ao Azure. Devolve None se o serviço falhar."""
    try:
        cliente = cliente or criar_cliente()
        sentimento = cliente.analyze_sentiment([texto], language="pt")[0]
        frases = cliente.extract_key_phrases([texto], language="pt")[0]
        if sentimento.is_error or frases.is_error:
            raise RuntimeError("o serviço devolveu erro para o documento")
        return AnaliseAzure(
            sentimento=sentimento.sentiment,
            pontuacao_negativa=sentimento.confidence_scores.negative,
            frases_chave=list(frases.key_phrases),
        )
    except Exception as erro:
        log.warning("Azure Text Analytics indisponível (%s). Seguindo só com termos críticos.", erro)
        return None


def analisar_texto(texto: str, cliente=None) -> list[Evento]:
    """Transforma o texto de uma consulta em eventos da fonte "texto"."""
    if not texto.strip():
        return []

    eventos = [
        Evento(
            fonte="texto",
            tipo="termo_critico",
            severidade=TERMOS_CRITICOS[termo],
            instante=0.0,
            descricao=f'Paciente mencionou "{termo}" na consulta',
        )
        for termo in encontrar_termos_criticos(texto)
    ]

    analise = consultar_azure(texto, cliente)
    if analise and analise.pontuacao_negativa > LIMITE_SENTIMENTO_NEGATIVO:
        eventos.append(
            Evento(
                fonte="texto",
                tipo="sentimento_negativo",
                severidade=1,
                instante=0.0,
                descricao=f"Sentimento negativo na fala do paciente ({analise.pontuacao_negativa:.0%})",
            )
        )
    return eventos
