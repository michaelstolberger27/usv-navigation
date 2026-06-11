# Root conftest: puts the repo root and src/ on sys.path so tests import
# colav_automaton (src-layout) and ais_replay (top-level) without an
# editable install. Needed because the host may have a different
# colav_automaton pip-installed (see HANDOFF §4 phase 1 note).
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
