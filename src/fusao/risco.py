"""Fusão multimodal: transforma os eventos de todas as fontes em um nível de risco.

A regra é simples de propósito, para que a equipe médica (e a banca)
consiga refazer a conta de cabeça:

1. De cada fonte vale só o evento mais grave (severidade de 1 a 3).
2. A pontuação é a soma dessas severidades.
3. Nível ALTO: pontuação 5 ou mais, ou qualquer evento grave (severidade 3).
   Nível MÉDIO: pontuação de 2 a 4.
   Nível BAIXO: pontuação 0 ou 1.

Consequência: um indício leve isolado (ex.: só a voz) não gera alerta,
mas vários indícios moderados em fontes diferentes somam e geram.
"""

from src.modelos import Evento, Risco

PONTUACAO_ALTO = 5
PONTUACAO_MEDIO = 2
SEVERIDADE_GRAVE = 3

# Falha de sensor é problema do equipamento, não do paciente: não entra na pontuação
FONTES_SEM_PESO_CLINICO = {"sensor"}


def calcular_risco(eventos: list[Evento]) -> Risco:
    maior_por_fonte: dict[str, int] = {}
    for evento in eventos:
        if evento.fonte in FONTES_SEM_PESO_CLINICO:
            continue
        maior_por_fonte[evento.fonte] = max(maior_por_fonte.get(evento.fonte, 0), evento.severidade)

    pontuacao = sum(maior_por_fonte.values())
    tem_grave = SEVERIDADE_GRAVE in maior_por_fonte.values()

    if pontuacao >= PONTUACAO_ALTO or tem_grave:
        nivel = "alto"
    elif pontuacao >= PONTUACAO_MEDIO:
        nivel = "medio"
    else:
        nivel = "baixo"

    return Risco(pontuacao=pontuacao, nivel=nivel, fontes=sorted(maior_por_fonte))
