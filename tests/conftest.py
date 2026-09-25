"""
Shared pytest configuration.

The test suite lives in ``tests/`` at the project root, next to the modules it
exercises (``architecture.py``, ``losses.py``, ``utils.py`` ...).  We add that
root to ``sys.path`` so the modules import whether pytest is launched from the
root or from inside ``tests/``.

Several modules import :mod:`torch` (and :mod:`lightgbm`) at import time.  Tests
that need them use ``pytest.importorskip`` so the suite still collects and runs
its pure-Python / NumPy / pandas tests in an environment without those heavy
dependencies.  On a full training environment every test runs.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
