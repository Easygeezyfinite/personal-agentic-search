"""Personal agentic search: local retrieval + a small tool-calling agent."""

import os

# faiss and torch each ship their own OpenMP runtime on macOS, and with both
# loaded multi-threaded the process segfaults (exit 139) as soon as the first
# model runs. Single-threaded OpenMP avoids it. Set before torch/faiss import.
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
