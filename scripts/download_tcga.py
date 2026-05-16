"""
Scarica i file TCGA Pan-Cancer Atlas necessari per l'analisi:
  - mc3.v0.2.8.PUBLIC.xena.gz                            (~64 MB)
  - EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena (~331 MB)
  - Survival_SupplementalTable_S1_20171025_xena_sp       (~1 MB, cancer type)

Source: UCSC Xena Pan-Cancer Atlas hub
        https://pancanatlas.xenahubs.net/

Output: data/tcga/
"""

import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path("data/tcga")
BASE = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download"

FILES = {
    "mc3.v0.2.8.PUBLIC.xena.gz":
        f"{BASE}/mc3.v0.2.8.PUBLIC.xena.gz",
    "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz":
        f"{BASE}/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
    "Survival_SupplementalTable_S1_20171025_xena_sp":
        f"{BASE}/Survival_SupplementalTable_S1_20171025_xena_sp",
}


def download(url, dst):
    """Scarica con progress in MB."""
    print(f"  {dst.name}")
    last_pct = -1

    def hook(blocks, blocksize, total):
        nonlocal last_pct
        if total <= 0:
            return
        pct = int(blocks * blocksize * 100 / total)
        if pct != last_pct and pct % 5 == 0:
            mb = blocks * blocksize / 1e6
            print(f"    {pct:3d}% ({mb:.0f} MB / {total/1e6:.0f} MB)")
            last_pct = pct

    urllib.request.urlretrieve(url, dst, reporthook=hook)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scarico TCGA in {DATA_DIR}")
    for fname, url in FILES.items():
        dst = DATA_DIR / fname
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"  [skip] {fname} gia' presente ({dst.stat().st_size/1e6:.1f} MB)")
            continue
        try:
            download(url, dst)
        except Exception as e:
            print(f"  ERRORE su {fname}: {e}", file=sys.stderr)
            if dst.exists():
                dst.unlink()
            raise
    print("\nDone.")
    for f in DATA_DIR.iterdir():
        print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
