import sys
from pathlib import Path

# lets tests do `from fakes import ...`
sys.path.insert(0, str(Path(__file__).parent))
