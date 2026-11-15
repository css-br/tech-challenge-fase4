"""Baixa os datasets públicos usados no projeto.

Uso (a partir da raiz do projeto):
    python scripts/baixar_dados.py voiced
    python scripts/baixar_dados.py rehab
    python scripts/baixar_dados.py bidmc
    python scripts/baixar_dados.py voiced --destino F:/fase4-techchallenge/data

Os dados ficam fora do git. O destino padrão é a pasta data/raw/.
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PHYSIONET = "https://physionet.org/files"
ZENODO_REHAB = "https://zenodo.org/records/13305826/files"


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


def baixar_arquivo_grande(url: str, destino: Path) -> None:
    """Baixa direto para o disco, sem carregar o arquivo inteiro na memória."""
    if destino.exists() and destino.stat().st_size > 0:
        print(f"{destino.name} já existe")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(url, timeout=120) as resposta, open(parcial, "wb") as arquivo:
        shutil.copyfileobj(resposta, arquivo, length=1024 * 1024)
    parcial.rename(destino)
    print(f"{destino.name}: {destino.stat().st_size / 1e6:.0f} MB")


def baixar_rehab(pasta: Path) -> None:
    """REHAB24-6: vídeos de exercícios de reabilitação com rótulo de execução correta ou incorreta.

    Licença CC BY-NC 4.0. Baixa só os vídeos (2,6 GB) e a tabela de repetições.
    """
    pasta = pasta / "rehab24-6"
    for nome in ("Segmentation.csv", "Segmentation.txt", "joints_names.txt"):
        baixar_arquivo(f"{ZENODO_REHAB}/{nome}?download=1", pasta / nome)
    baixar_arquivo_grande(f"{ZENODO_REHAB}/videos.zip?download=1", pasta / "videos.zip")
    if not (pasta / "videos").exists():
        with zipfile.ZipFile(pasta / "videos.zip") as pacote:
            pacote.extractall(pasta / "videos")
    print(f"REHAB24-6 em {pasta}")


def baixar_bidmc(pasta: Path) -> None:
    """BIDMC: sinais vitais (batimentos, oxigenação, respiração) de 53 pacientes de UTI, a 1 Hz."""
    base = f"{PHYSIONET}/bidmc/1.0.0/bidmc_csv"
    pasta = pasta / "bidmc"
    pares = [
        (f"{base}/bidmc_{numero:02d}_Numerics.csv", pasta / f"bidmc_{numero:02d}_Numerics.csv")
        for numero in range(1, 54)
    ]
    baixar_varios(pares)
    print(f"BIDMC em {pasta}")


DATASETS = {"voiced": baixar_voiced, "rehab": baixar_rehab, "bidmc": baixar_bidmc}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--destino", type=Path, default=Path("data/raw"))
    argumentos = parser.parse_args()
    DATASETS[argumentos.dataset](argumentos.destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
