"""Monitoramento contínuo: acompanha os sinais vitais amostra a amostra.

A cada leitura nova, o risco do paciente é recalculado juntando os
eventos que já existiam (vídeo, voz, texto, prescrições) com os eventos
novos dos sinais vitais. Um alerta é emitido apenas quando o nível de
risco SOBE, para não repetir o mesmo aviso.
"""

import time
from dataclasses import dataclass

import pandas as pd

from src.anomalias.sinais_vitais import SINAIS, DetectorSinaisVitais
from src.fusao.alerta import emitir_alerta
from src.fusao.risco import calcular_risco
from src.modelos import Evento, Risco

ORDEM_DOS_NIVEIS = {"baixo": 0, "medio": 1, "alto": 2}


@dataclass
class ResultadoMonitoramento:
    eventos: list[Evento]   # todos os eventos do paciente ao final
    risco: Risco            # risco ao final
    alertas: list[dict]     # alertas emitidos durante o monitoramento


def monitorar_sinais(
    paciente_id: str,
    eventos_anteriores: list[Evento],
    serie: pd.DataFrame,
    detector: DetectorSinaisVitais,
    nivel_ja_alertado: str = "baixo",
    ao_alertar=emitir_alerta,
    pausa_s: float = 0.0,
    ao_ler=None,
) -> ResultadoMonitoramento:
    """Percorre a série de sinais vitais como se as leituras chegassem em tempo real.

    - eventos_anteriores: eventos já detectados nas outras fontes.
    - nivel_ja_alertado: nível do último alerta já enviado para este paciente.
    - ao_alertar: função chamada para emitir o alerta (trocada nos testes).
    - pausa_s: espera entre leituras, para a demonstração (0 = sem espera).
    - ao_ler: função opcional chamada a cada leitura, para mostrar o andamento.
    """
    detector.reiniciar()
    eventos = list(eventos_anteriores)
    nivel_alertado = nivel_ja_alertado
    risco = calcular_risco(eventos)
    alertas = []

    for linha in serie.itertuples():
        amostra = {sinal: getattr(linha, sinal) for sinal in SINAIS}
        novos = detector.avaliar_amostra(amostra, float(linha.tempo_s))
        if ao_ler:
            ao_ler(linha.tempo_s, amostra, novos)

        if novos:
            eventos += novos
            risco = calcular_risco(eventos)
            if ORDEM_DOS_NIVEIS[risco.nivel] > ORDEM_DOS_NIVEIS[nivel_alertado]:
                alerta = ao_alertar(paciente_id, risco, eventos)
                if alerta:
                    alertas.append(alerta)
                nivel_alertado = risco.nivel

        if pausa_s:
            time.sleep(pausa_s)

    return ResultadoMonitoramento(eventos=eventos, risco=risco, alertas=alertas)
