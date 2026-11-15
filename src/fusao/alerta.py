"""Emissão do alerta para a equipe médica.

O alerta é impresso no terminal e gravado em results/alertas.jsonl (uma
linha por alerta). O texto do relatório é redigido pelo GPT (Azure
OpenAI). A DECISÃO de alertar e o nível de risco nunca vêm do GPT: vêm
das regras de src/fusao/risco.py. Se o GPT estiver indisponível, o
alerta sai com um texto padrão.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.modelos import Evento, Risco

log = logging.getLogger(__name__)

CAMINHO_ALERTAS = Path("results/alertas.jsonl")
NIVEIS_COM_ALERTA = {"medio", "alto"}
ROTULOS = {"baixo": "BAIXO", "medio": "MÉDIO", "alto": "ALTO"}

INSTRUCOES_GPT = (
    "Você redige comunicados curtos para a equipe médica de um hospital, em português do Brasil. "
    "Receberá o nível de risco de um paciente e a lista de eventos detectados por um sistema de "
    "monitoramento. Escreva um relatório objetivo de até 120 palavras: comece pelo nível de risco, "
    "resuma os achados do mais grave ao mais leve e termine sugerindo que a equipe avalie o paciente. "
    "Use apenas as informações recebidas. Não faça diagnóstico, não sugira medicamentos nem doses, "
    "e não invente dados."
)


def _ordenados(eventos: list[Evento]) -> list[Evento]:
    """Do mais grave ao mais leve, mantendo a ordem original entre os de mesma gravidade."""
    return sorted(eventos, key=lambda evento: -evento.severidade)


def montar_relatorio_padrao(paciente_id: str, risco: Risco, eventos: list[Evento]) -> str:
    """Relatório por modelo fixo de texto, usado quando o GPT não está disponível."""
    linhas = [
        f"Paciente {paciente_id}: risco {ROTULOS[risco.nivel]} (pontuação {risco.pontuacao}).",
        "Eventos detectados:",
    ]
    linhas += [f"- [{evento.fonte}, severidade {evento.severidade}] {evento.descricao}" for evento in _ordenados(eventos)]
    linhas.append("Recomenda-se avaliação do paciente pela equipe médica.")
    return "\n".join(linhas)


def redigir_com_gpt(paciente_id: str, risco: Risco, eventos: list[Evento]) -> str:
    """Pede ao GPT (Azure OpenAI) a redação do relatório."""
    from openai import AzureOpenAI

    from src.config import OPENAI_API_VERSION, carregar_config

    config = carregar_config()
    cliente = AzureOpenAI(
        api_key=config.openai_key,
        azure_endpoint=config.openai_endpoint,
        api_version=OPENAI_API_VERSION,
        timeout=30,
    )
    dados = montar_relatorio_padrao(paciente_id, risco, eventos)
    resposta = cliente.chat.completions.create(
        model=config.openai_deployment,
        messages=[
            {"role": "system", "content": INSTRUCOES_GPT},
            {"role": "user", "content": dados},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return resposta.choices[0].message.content.strip()


def redigir_relatorio(paciente_id: str, risco: Risco, eventos: list[Evento], redator=None) -> str:
    """Texto do relatório. Usa o GPT e, se ele falhar, o texto padrão.

    `redator` é a função que escreve o texto; existe para os testes
    poderem simular o GPT fora do ar.
    """
    redator = redator or redigir_com_gpt
    try:
        return redator(paciente_id, risco, eventos)
    except Exception as erro:
        log.warning("Redação pelo GPT indisponível (%s). Usando o texto padrão.", erro)
        return montar_relatorio_padrao(paciente_id, risco, eventos)


def emitir_alerta(
    paciente_id: str,
    risco: Risco,
    eventos: list[Evento],
    caminho: Path = CAMINHO_ALERTAS,
    redator=None,
) -> dict | None:
    """Emite o alerta se o risco for médio ou alto. Devolve o registro gravado, ou None."""
    if risco.nivel not in NIVEIS_COM_ALERTA:
        return None

    alerta = {
        "paciente_id": paciente_id,
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "nivel": risco.nivel,
        "pontuacao": risco.pontuacao,
        "fontes": risco.fontes,
        "eventos": [asdict(evento) for evento in _ordenados(eventos)],
        "relatorio": redigir_relatorio(paciente_id, risco, eventos, redator),
    }

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(alerta, ensure_ascii=False) + "\n")

    borda = "=" * 78
    print(f"\n{borda}")
    print(f"  ALERTA PARA A EQUIPE MÉDICA | paciente {paciente_id} | risco {ROTULOS[risco.nivel]} | {alerta['data_hora']}")
    print(borda)
    print(alerta["relatorio"])
    print(f"{borda}\n")
    return alerta
