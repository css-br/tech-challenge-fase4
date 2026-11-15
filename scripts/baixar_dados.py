"""Baixa os datasets públicos usados no projeto.

Uso (a partir da raiz do projeto):
    python scripts/baixar_dados.py voiced
    python scripts/baixar_dados.py voiced --destino F:/fase4-techchallenge/data

Os dados ficam fora do git. O destino padrão é a pasta data/raw/.
"""

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PHYSIONET = "https://physionet.org/files"


def baixar_arquivo(url: str, destino: Path) -> bool:
    """Baixa um arquivo, pulando se já existir. Devolve True se baixou agora."""
    if destino.exists() and destino.stat().st_size > 0:
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resposta:
        destino.write_bytes(resposta.read())
    return True


def baixar_varios(pares: list[tuple[str, Path]]) -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        resultados = list(executor.map(lambda par: baixar_arquivo(*par), pares))
    print(f"{sum(resultados)} arquivos baixados, {len(resultados) - sum(resultados)} já existiam")


def baixar_voiced(pasta: Path) -> None:
    """VOICED: 208 gravações de voz (vogal 'a'), saudáveis e patológicas."""
    base = f"{PHYSIONET}/voiced/1.0.0"
    pasta = pasta / "voiced"
    baixar_arquivo(f"{base}/RECORDS", pasta / "RECORDS")
    registros = (pasta / "RECORDS").read_text().split()
    pares = []
    for registro in registros:
        for sufixo in (".hea", ".dat", "-info.txt"):
            pares.append((f"{base}/{registro}{sufixo}", pasta / f"{registro}{sufixo}"))
    baixar_varios(pares)
    print(f"VOICED em {pasta} ({len(registros)} registros)")


DATASETS = {"voiced": baixar_voiced}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--destino", type=Path, default=Path("data/raw"))
    argumentos = parser.parse_args()
    DATASETS[argumentos.dataset](argumentos.destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
