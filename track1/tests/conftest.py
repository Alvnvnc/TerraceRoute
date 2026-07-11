import sys
from pathlib import Path

# Tests import the agent package from the track1 root (same layout as eval/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
