"""Tipos compartilhados por todos os módulos do projeto."""

from dataclasses import dataclass


@dataclass
class Evento:
    """Algo fora do padrão detectado em uma das fontes de dados do paciente.

    Todos os módulos (vídeo, voz, texto, sinais vitais, prescrições e
    movimentação) devolvem listas de Evento. É esse formato comum que
    permite juntar tudo na etapa de fusão.
    """

    fonte: str        # "video", "voz", "texto", "sinais_vitais", "prescricao", "movimentacao" ou "sensor"
    tipo: str         # ex.: "angulo_fora_da_faixa", "termo_critico"
    severidade: int   # 1 (leve), 2 (moderada), 3 (grave)
    instante: float   # segundos desde o início do dado analisado
    descricao: str    # frase legível, usada no texto do alerta


@dataclass
class Risco:
    """Resultado da fusão dos eventos de um paciente."""

    pontuacao: int
    nivel: str         # "baixo", "medio" ou "alto"
    fontes: list[str]  # fontes que contribuíram para a pontuação
