import csv
from pathlib import Path
from typing import Dict, Optional, Union, List

def load_fairseq_manifest(filename: Union[Path, str]) -> List[Dict[str, str]]:
    with open(filename) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        return [dict(e) for e in reader]


def get_fairseq_manifest_path(
    root: Union[Path, str],
    subset: str,
) -> Path:
    assert root is not None
    assert subset is not None
    manifest_path = Path(root) / f"{subset}.tsv"
    assert manifest_path.exists(), f"{subset}.tsv doesn't exist in {root}"
    return manifest_path