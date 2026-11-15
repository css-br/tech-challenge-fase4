"""Detecção de alterações inesperadas na evolução das prescrições.

Aqui usamos regras explícitas em vez de um modelo: são poucas prescrições
por paciente (dado insuficiente para treinar) e a equipe médica precisa
entender exatamente por que um alerta foi gerado.
"""

import pandas as pd

from src.modelos import Evento

MUDANCA_BRUSCA = 0.5        # variação de dose acima de 50% entre prescrições seguidas
FATOR_GRAVE = 2.0           # dose mais que dobrada
JANELA_TROCA = pd.Timedelta(hours=24)


def _segundos(instante: pd.Timestamp, inicio: pd.Timestamp) -> float:
    return (instante - inicio).total_seconds()


def _mudancas_de_dose(paciente: str, prescricoes: pd.DataFrame, inicio: pd.Timestamp) -> list[Evento]:
    eventos = []
    for medicamento, historico in prescricoes.groupby("medicamento"):
        anterior = None
        for linha in historico.itertuples():
            if anterior is not None and anterior.dose_mg > 0:
                variacao = abs(linha.dose_mg - anterior.dose_mg) / anterior.dose_mg
                if variacao > MUDANCA_BRUSCA:
                    grave = linha.dose_mg > FATOR_GRAVE * anterior.dose_mg
                    eventos.append(Evento(
                        fonte="prescricao",
                        tipo="mudanca_brusca_de_dose",
                        severidade=3 if grave else 2,
                        instante=_segundos(linha.data_hora, inicio),
                        descricao=(
                            f"Paciente {paciente}: dose de {medicamento} passou de "
                            f"{anterior.dose_mg:g} mg para {linha.dose_mg:g} mg ({variacao:.0%} de variação)"
                        ),
                    ))
            anterior = linha
    return eventos


def _trocas_de_tratamento(paciente: str, prescricoes: pd.DataFrame, inicio: pd.Timestamp) -> list[Evento]:
    """Dois ou mais medicamentos novos iniciados em menos de 24 h.

    Os medicamentos prescritos nas primeiras 24 h (a prescrição de entrada)
    não contam como novos.
    """
    primeira_vez = prescricoes.groupby("medicamento")["data_hora"].min().sort_values()
    novos = primeira_vez[primeira_vez > inicio + JANELA_TROCA]

    eventos, grupo = [], []
    for medicamento, instante in novos.items():
        if grupo and instante - grupo[-1][1] >= JANELA_TROCA:
            grupo = []
        grupo.append((medicamento, instante))
        if len(grupo) == 2:  # evento no momento em que o segundo medicamento novo aparece
            eventos.append(Evento(
                fonte="prescricao",
                tipo="troca_inesperada_de_tratamento",
                severidade=2,
                instante=_segundos(instante, inicio),
                descricao=(
                    f"Paciente {paciente}: medicamentos novos iniciados em menos de 24 h: "
                    + " e ".join(nome for nome, _ in grupo)
                ),
            ))
    return eventos


def detectar_anomalias_prescricao(df: pd.DataFrame) -> list[Evento]:
    """Eventos da fonte "prescricao".

    Colunas esperadas: paciente_id, data_hora, medicamento, dose_mg.
    O instante de cada evento é contado em segundos desde a primeira
    prescrição do paciente.
    """
    df = df.copy()
    df["data_hora"] = pd.to_datetime(df["data_hora"])

    eventos = []
    for paciente, prescricoes in df.sort_values("data_hora").groupby("paciente_id"):
        inicio = prescricoes["data_hora"].min()
        eventos += _mudancas_de_dose(paciente, prescricoes, inicio)
        eventos += _trocas_de_tratamento(paciente, prescricoes, inicio)
    return sorted(eventos, key=lambda evento: evento.instante)
