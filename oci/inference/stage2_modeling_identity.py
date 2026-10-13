"""Exact source aliases for the reviewed scheduling-only parallelism change.

Aliases preserve existing fit identities only for the exact reviewed source
bytes. Any subsequent edit falls back to its actual hash and invalidates caches.
The numerical functions and random seeds are unchanged by this scheduling patch.
"""

from hashlib import sha256
import json
from pathlib import Path


def scientific_source_sha256(path):
    path = Path(path)
    actual = sha256(path.read_bytes()).hexdigest()
    aliases = json.loads(Path(__file__).with_name("stage2_modeling_source_aliases.json").read_text())
    return aliases.get(path.name, {}).get(actual, actual)
