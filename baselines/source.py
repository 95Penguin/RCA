"""Load pinned author modules without changing their on-disk snapshots."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent


def source_lock() -> dict:
    return json.loads((BASE / "sources.lock.json").read_text())


def verify_sources(repo: str | None = None) -> dict[str, str]:
    sources = source_lock()["sources"]
    selected = {repo: sources[repo]} if repo else sources
    result = {}
    for name, info in selected.items():
        for relative, expected in info["files"].items():
            path = BASE / "upstream" / name / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Author snapshot missing or changed: {name}/{relative}")
        result[name] = info["commit"]
    return result


# Compatibility fixes only; the author source remains unchanged and hash-checked.
PATCHES = {
    "eadro": [
        ("dilation=2, dev=\"cpu\"):", "dilation=2, dev=\"cpu\", **kwargs):"),
        ("dropout=trace_dropout)", "dropout=kwargs.get('trace_dropout', 0.0))"),
        ("dropout=metric_dropout)", "dropout=kwargs.get('metric_dropout', 0.0))"),
    ],
    "art": [
        ("dim_feedforward=in_dim*2,", "dim_feedforward=in_dim*2,\n            batch_first=True,"),
        ("self.convs = []", "self.convs = nn.ModuleList()"),
        ("h = h.view(-1, channel_dim, instance_num)", "h = h.reshape(-1, channel_dim, instance_num)"),
        ("h = h.permute(0,2,1).view(bacth_size, series_len, instance_num, channel_dim)",
         "h = h.permute(0,2,1).reshape(bacth_size, series_len, instance_num, channel_dim)"),
    ],
}


def load_author_module(repo: str, relative: str, *, patch: bool = False):
    """Isolate module names; never run an author's experiment entry on import."""
    verify_sources(repo)
    key = "_nextminute_author_" + repo + "_" + relative.replace("/", "_").replace(".", "_")
    key += "_patched_v1" if patch else "_raw"
    if key in sys.modules:
        return sys.modules[key]
    path = BASE / "upstream" / repo / relative
    source = path.read_text()
    if patch:
        for before, after in PATCHES.get(repo, []):
            if source.count(before) != 1:
                raise RuntimeError(f"Compatibility patch no longer applies exactly once: {repo}: {before}")
            source = source.replace(before, after)
    module = types.ModuleType(key)
    module.__file__ = str(path)
    sys.modules[key] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return module
