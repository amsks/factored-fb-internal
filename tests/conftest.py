import sys
from pathlib import Path

# Add project root to sys.path so all absolute imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))
