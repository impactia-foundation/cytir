"""cytir.sources.families — one module per source family. `news.py` (task 646
S4, moved from task 385's `materialize_news_pack.py`) is the first:
materializes one immutable news pack + explicit predecessor pointers for the
daily news timer and the 2-hourly direct-compose refresh lane.
"""
from __future__ import annotations
