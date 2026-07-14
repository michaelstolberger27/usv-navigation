# Puts the repo root and src/ on sys.path so tests import colav_automaton
# (src-layout) and ais_traffic (top-level) without an editable install.
# Needed because the host may have a different colav_automaton pip-installed.
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
