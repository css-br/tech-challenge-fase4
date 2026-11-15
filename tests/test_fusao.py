import json

from src.fusao.alerta import emitir_alerta, montar_relatorio_padrao, redigir_relatorio
from src.fusao.risco import calcular_risco
from src.modelos import Evento, Risco


def _evento(fonte: str, severidade: int, descricao: str = "algo fora do padrão") -> Evento:
    return Evento(fonte=fonte, tipo="teste", severidade=severidade, instante=0.0, descricao=descricao)


def _redator_fixo(texto: str):
    return lambda paciente_id, risco, eventos: texto


def _redator_que_falha(paciente_id, risco, eventos):
    raise ConnectionError("Azure OpenAI indisponível")


# ---------- calcular_risco ----------

def test_sem_eventos_o_risco_e_baixo():
    risco = calcular_risco([])

    assert risco == Risco(pontuacao=0, nivel="baixo", fontes=[])


def test_um_evento_leve_isolado_mantem_risco_baixo():
    risco = calcular_risco([_evento("voz", 1)])

    assert (risco.pontuacao, risco.nivel) == (1, "baixo")


def test_um_evento_grave_ja_torna_o_risco_alto():
    risco = calcular_risco([_evento("sinais_vitais", 3)])

    assert (risco.pontuacao, risco.nivel) == (3, "alto")


def test_duas_fontes_moderadas_dao_risco_medio():
    risco = calcular_risco([_evento("video", 2), _evento("texto", 2)])

    assert (risco.pontuacao, risco.nivel) == (4, "medio")
    assert risco.fontes == ["texto", "video"]


def test_tres_fontes_moderadas_dao_risco_alto():
    risco = calcular_risco([_evento("video", 2), _evento("texto", 2), _evento("prescricao", 2)])

    assert (risco.pontuacao, risco.nivel) == (6, "alto")


def test_varios_eventos_da_mesma_fonte_contam_uma_vez_pelo_mais_grave():
    risco = calcular_risco([_evento("texto", 1), _evento("texto", 2), _evento("texto", 1)])

    assert (risco.pontuacao, risco.nivel) == (2, "medio")
    assert risco.fontes == ["texto"]


def test_falha_de_sensor_nao_aumenta_o_risco_clinico():
    risco = calcular_risco([_evento("sensor", 1), _evento("voz", 1)])

    assert risco.pontuacao == 1
    assert risco.fontes == ["voz"]


# ---------- relatório ----------

def test_relatorio_padrao_lista_nivel_e_eventos_do_mais_grave_ao_mais_leve():
    eventos = [_evento("voz", 1, "voz alterada"), _evento("sinais_vitais", 3, "SpO2 em 82%")]
    risco = calcular_risco(eventos)

    texto = montar_relatorio_padrao("P003", risco, eventos)

    assert "P003" in texto
    assert "ALTO" in texto
    assert texto.index("SpO2 em 82%") < texto.index("voz alterada")


def test_redacao_usa_o_texto_padrao_quando_o_gpt_falha():
    eventos = [_evento("sinais_vitais", 3, "SpO2 em 82%")]
    risco = calcular_risco(eventos)

    texto = redigir_relatorio("P003", risco, eventos, redator=_redator_que_falha)

    assert texto == montar_relatorio_padrao("P003", risco, eventos)


# ---------- emitir_alerta ----------

def test_risco_baixo_nao_emite_alerta_nem_grava_arquivo(tmp_path):
    caminho = tmp_path / "alertas.jsonl"
    eventos = [_evento("voz", 1)]

    alerta = emitir_alerta("P001", calcular_risco(eventos), eventos, caminho, redator=_redator_fixo("x"))

    assert alerta is None
    assert not caminho.exists()


def test_alerta_e_gravado_com_todos_os_campos(tmp_path, capsys):
    caminho = tmp_path / "alertas.jsonl"
    eventos = [_evento("sinais_vitais", 3, "SpO2 em 82%"), _evento("texto", 2, 'mencionou "tontura"')]
    risco = calcular_risco(eventos)

    alerta = emitir_alerta("P003", risco, eventos, caminho, redator=_redator_fixo("Relatório de teste."))

    gravado = json.loads(caminho.read_text(encoding="utf-8").strip())
    assert gravado == alerta
    assert gravado["paciente_id"] == "P003"
    assert gravado["nivel"] == "alto"
    assert gravado["pontuacao"] == 5
    assert gravado["fontes"] == ["sinais_vitais", "texto"]
    assert gravado["relatorio"] == "Relatório de teste."
    assert [e["descricao"] for e in gravado["eventos"]] == ["SpO2 em 82%", 'mencionou "tontura"']
    assert "data_hora" in gravado
    assert "P003" in capsys.readouterr().out  # também aparece no terminal


def test_alertas_seguidos_sao_acrescentados_ao_arquivo(tmp_path):
    caminho = tmp_path / "alertas.jsonl"
    eventos = [_evento("sinais_vitais", 3)]
    risco = calcular_risco(eventos)

    emitir_alerta("P001", risco, eventos, caminho, redator=_redator_fixo("a"))
    emitir_alerta("P002", risco, eventos, caminho, redator=_redator_fixo("b"))

    linhas = caminho.read_text(encoding="utf-8").strip().split("\n")
    assert [json.loads(linha)["paciente_id"] for linha in linhas] == ["P001", "P002"]


def test_alerta_sai_mesmo_com_o_gpt_fora_do_ar(tmp_path):
    caminho = tmp_path / "alertas.jsonl"
    eventos = [_evento("sinais_vitais", 3, "SpO2 em 82%")]
    risco = calcular_risco(eventos)

    alerta = emitir_alerta("P003", risco, eventos, caminho, redator=_redator_que_falha)

    assert alerta["relatorio"] == montar_relatorio_padrao("P003", risco, eventos)
    assert caminho.exists()
