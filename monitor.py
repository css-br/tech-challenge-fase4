"""Fluxo completo de monitoramento multimodal de um paciente.

Uso (a partir da raiz do projeto):
    python monitor.py --paciente P003
    python monitor.py --paciente P001 --velocidade 50
    python monitor.py --lista

Etapas:
    1. Vídeo      -> postura (MediaPipe) comparada ao padrão do paciente, e posição da pessoa (YOLOv8)
    2. Voz        -> classificador de alterações vocais
    3. Consulta   -> transcrição (Azure Speech) e análise do texto (Azure Text Analytics)
    4. Prescrições-> regras de mudança de tratamento
    5. Sinais vitais em tempo real (simulado) -> detector de anomalias

Depois das etapas 1 a 4 o risco é calculado e, se for médio ou alto, o
alerta é emitido. Na etapa 5 o risco é recalculado a cada evento novo e
um novo alerta sai quando o nível sobe.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import soundfile as sf

from src.anomalias.movimentacao import detectar_anomalias_movimentacao
from src.anomalias.prescricoes import detectar_anomalias_prescricao
from src.audio.texto import analisar_texto
from src.audio.transcricao import transcrever
from src.audio.voz import avaliar_voz
from src.fusao.alerta import emitir_alerta, montar_relatorio_padrao
from src.fusao.monitoramento import monitorar_sinais
from src.fusao.risco import calcular_risco
from src.modelos import Evento
from src.video.desvios import avaliar_execucao
from src.video.detector import detectar_pessoa
from src.video.pose import extrair_poses

PACIENTES = Path("data/pacientes_demo.json")
PRESCRICOES = Path("data/prescricoes.csv")
PADRAO_VIDEO = Path("models/padrao_video.json")  # padrão individual do paciente (notebook 03)
MODELO_VOZ = Path("models/voz.joblib")
MODELO_SINAIS = Path("models/sinais_vitais.joblib")
PASSO_VIDEO = 3  # analisa um a cada 3 quadros


def mostrar(titulo: str, eventos: list[Evento]) -> None:
    print(f"\n--- {titulo} ---")
    if not eventos:
        print("    nenhum evento")
    for evento in eventos:
        print(f"    [severidade {evento.severidade}] {evento.descricao}")


def analisar_video(paciente_id: str, caminho: Path) -> list[Evento]:
    padrao = json.loads(PADRAO_VIDEO.read_text(encoding="utf-8"))
    anotado = Path("results") / f"{paciente_id}_video_anotado.mp4"

    angulos = extrair_poses(caminho, saida_anotada=anotado, passo=PASSO_VIDEO)
    eventos = avaliar_execucao(angulos, padrao)

    trajetoria = detectar_pessoa(caminho, passo=PASSO_VIDEO)
    eventos += detectar_anomalias_movimentacao(trajetoria)

    print(f"    vídeo anotado: {anotado.as_posix()}")
    return eventos


def analisar_voz(caminho: Path) -> list[Evento]:
    salvo = joblib.load(MODELO_VOZ)
    sinal, sr = sf.read(caminho)
    return avaliar_voz(sinal, sr, salvo["modelo"], limite=salvo["limite"])


def analisar_consulta(caminho: Path) -> list[Evento]:
    texto = transcrever(caminho)
    print(f'    transcrição: "{texto}"' if texto else "    (nenhuma fala reconhecida)")
    return analisar_texto(texto)


def analisar_prescricoes(paciente_id: str) -> list[Evento]:
    prescricoes = pd.read_csv(PRESCRICOES)
    return detectar_anomalias_prescricao(prescricoes[prescricoes["paciente_id"] == paciente_id])


def acompanhar_leitura(tempo_s: float, amostra: dict, novos: list[Evento]) -> None:
    """Mostra uma leitura a cada 30 s e todos os eventos no momento em que acontecem."""
    if tempo_s % 30 == 0:
        print(f"    t={tempo_s:>4.0f}s  FC {amostra['fc']:.0f}  SpO2 {amostra['spo2']:.0f}  "
              f"FR {amostra['fr']:.0f}  PAS {amostra['pas']:.0f}")
    for evento in novos:
        print(f"    t={tempo_s:>4.0f}s  >>> EVENTO [severidade {evento.severidade}] {evento.descricao}")


def monitorar(paciente_id: str, velocidade: float, usar_gpt: bool) -> int:
    pacientes = json.loads(PACIENTES.read_text(encoding="utf-8"))
    if paciente_id not in pacientes:
        print(f"Paciente {paciente_id} não encontrado. Disponíveis: {', '.join(pacientes)}")
        return 1
    dados = pacientes[paciente_id]
    redator = None if usar_gpt else montar_relatorio_padrao

    def alertar(paciente, risco, eventos):
        return emitir_alerta(paciente, risco, eventos, redator=redator)

    print(f"\n{'#' * 78}\n# Paciente {paciente_id}: {dados['descricao']}\n{'#' * 78}")

    eventos: list[Evento] = []
    etapas = [
        ("1/5 Vídeo de fisioterapia", lambda: analisar_video(paciente_id, Path(dados["video"]))),
        ("2/5 Voz", lambda: analisar_voz(Path(dados["voz"]))),
        ("3/5 Áudio da consulta", lambda: analisar_consulta(Path(dados["consulta"]))),
        ("4/5 Prescrições", lambda: analisar_prescricoes(paciente_id)),
    ]
    for titulo, etapa in etapas:
        print(f"\n[{titulo}] processando...")
        try:
            novos = etapa()
        except Exception as erro:  # uma fonte com problema não pode derrubar o monitoramento
            print(f"    etapa indisponível: {erro}")
            novos = []
        mostrar(titulo, novos)
        eventos += novos

    risco = calcular_risco(eventos)
    print(f"\n>>> Risco antes dos sinais vitais: {risco.nivel.upper()} (pontuação {risco.pontuacao}, fontes: {', '.join(risco.fontes) or 'nenhuma'})")
    alertar(paciente_id, risco, eventos)

    print(f"\n[5/5 Sinais vitais em tempo real] {velocidade:g}x a velocidade real")
    serie = pd.read_csv(dados["sinais_vitais"])
    resultado = monitorar_sinais(
        paciente_id, eventos, serie, joblib.load(MODELO_SINAIS),
        nivel_ja_alertado=risco.nivel, ao_alertar=alertar,
        pausa_s=1 / velocidade if velocidade > 0 else 0, ao_ler=acompanhar_leitura,
    )

    final = resultado.risco
    print(f"\n{'#' * 78}")
    print(f"# Fim do monitoramento de {paciente_id}: risco {final.nivel.upper()} (pontuação {final.pontuacao})")
    print(f"# Eventos: {len(resultado.eventos)} | fontes: {', '.join(final.fontes) or 'nenhuma'}")
    print(f"# Alertas registrados em results/alertas.jsonl")
    print(f"{'#' * 78}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitoramento multimodal de um paciente de demonstração.")
    parser.add_argument("--paciente", default="P003")
    parser.add_argument("--velocidade", type=float, default=20, help="quantas vezes mais rápido que o tempo real (0 = sem pausa)")
    parser.add_argument("--sem-gpt", action="store_true", help="redige o alerta com o texto padrão, sem chamar o Azure OpenAI")
    parser.add_argument("--lista", action="store_true", help="mostra os pacientes disponíveis")
    argumentos = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="    aviso: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos no terminal do Windows

    if argumentos.lista:
        for paciente_id, dados in json.loads(PACIENTES.read_text(encoding="utf-8")).items():
            print(f"{paciente_id}: {dados['descricao']}")
        return 0
    return monitorar(argumentos.paciente, argumentos.velocidade, usar_gpt=not argumentos.sem_gpt)


if __name__ == "__main__":
    sys.exit(main())
