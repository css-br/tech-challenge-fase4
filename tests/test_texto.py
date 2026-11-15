from src.audio.texto import analisar_texto, encontrar_termos_criticos


class ClienteQueFalha:
    """Simula o serviço Azure fora do ar."""

    def analyze_sentiment(self, *args, **kwargs):
        raise ConnectionError("serviço indisponível")

    def extract_key_phrases(self, *args, **kwargs):
        raise ConnectionError("serviço indisponível")


def test_encontra_termos_ignorando_maiusculas_e_acentos():
    texto = "Doutor, sinto FALTA DE AR à noite e uma Dor no Peito quando subo escadas."

    termos = encontrar_termos_criticos(texto)

    assert "falta de ar" in termos
    assert "dor no peito" in termos


def test_encontra_termo_escrito_sem_acento():
    assert "confusão" in encontrar_termos_criticos("tive um pouco de confusao ontem")


def test_texto_sem_queixas_nao_tem_termos():
    texto = "Estou me sentindo bem, dormi bem e tenho feito minhas caminhadas."

    assert encontrar_termos_criticos(texto) == []


def test_texto_vazio_nao_tem_termos():
    assert encontrar_termos_criticos("") == []


def test_nao_confunde_pedaco_de_palavra_com_termo():
    # "desmaio" não deve ser encontrado dentro de outra palavra
    assert encontrar_termos_criticos("ele estava desmaiozinho de sono") == []


def test_analise_gera_um_evento_por_termo_critico():
    eventos = analisar_texto("Sinto falta de ar e dor no peito.", cliente=ClienteQueFalha())

    tipos = [(e.fonte, e.tipo) for e in eventos]
    assert tipos == [("texto", "termo_critico"), ("texto", "termo_critico")]
    assert {e.severidade for e in eventos} == {3}


def test_analise_continua_quando_o_azure_falha():
    eventos = analisar_texto("Estou com muita tontura.", cliente=ClienteQueFalha())

    assert len(eventos) == 1
    assert "tontura" in eventos[0].descricao


def test_analise_de_texto_vazio_devolve_lista_vazia():
    assert analisar_texto("", cliente=ClienteQueFalha()) == []
