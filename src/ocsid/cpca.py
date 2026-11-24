"""CPCA: critical-path counterfactual attribution on request span trees.

A request is a tree of spans.  For every span we decompose its duration into

* ``local`` : own compute/IO, not adjacent to a child call;
* ``pre``   : gap right before a child starts (request path + queue);
* ``post``  : gap right after a child ends (return path).

The cost of service ``i`` is its local time plus the gaps on edges into ``i``
(edge waits belong to the callee).  Restoring ``i`` replaces those quantities by
a normal reference and replays the tree; the drop in entry latency is the
counterfactual removable latency.

This module is training-free and is shared by the audit, the fusion script and
the feature builder.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    sid: str
    parent: str
    service: str
    operation: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def normalize_service(name: str) -> str:
    name = name.strip()
    return "frontendservice" if name in {"frontend", "frontend-external"} else name


def union_length(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    current_end: float | None = None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            total += end - start
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end
    return total


def is_entry(span: Span, dataset: str) -> bool:
    """Entry-span policy used by the RE2 labels."""
    parent = (span.parent or "").strip().lower()
    parent_blank = not parent or parent in {"nan", "none", "null"}
    if dataset == "ob":
        return span.service == "frontendservice" and span.operation == "frontend"
    return parent_blank and span.operation.startswith(
        ("GET ", "POST ", "PUT ", "DELETE ", "PATCH ")
    )


def decompose_tree(spans: list[Span], dataset: str):
    """Return ``(by_id, children, root, local, pre, post)`` for one request.

    ``pre[c.sid]`` / ``post[c.sid]`` are the gaps around child ``c`` inside its
    parent.  ``local[s.sid]`` is the parent's self time after removing the child
    union and both gaps, so it is never negative.
    """
    by_id = {span.sid: span for span in spans}
    children: dict[str, list[Span]] = defaultdict(list)
    roots: list[Span] = []
    for span in spans:
        if by_id.get(span.parent) is None:
            roots.append(span)
        else:
            children[span.parent].append(span)
    if not roots:
        return None
    entries = [span for span in roots if is_entry(span, dataset)]
    root = (
        min(entries, key=lambda s: (s.start, -s.duration))
        if entries
        else max(roots, key=lambda s: s.duration)
    )
    local: dict[str, float] = {}
    pre: dict[str, float] = {}
    post: dict[str, float] = {}
    for span in spans:
        kids = sorted(children.get(span.sid, []), key=lambda s: s.start)
        if not kids:
            local[span.sid] = max(span.duration, 0.0)
            continue
        child_union = union_length([(c.start, c.end) for c in kids])
        pre_sum = post_sum = 0.0
        previous_end = span.start
        for index, child in enumerate(kids):
            gap = max(child.start - previous_end, 0.0)
            pre[child.sid] = gap
            pre_sum += gap
            next_start = kids[index + 1].start if index + 1 < len(kids) else span.end
            tail = max(next_start - child.end, 0.0)
            post[child.sid] = tail
            post_sum += tail
            previous_end = max(previous_end, child.end)
        local[span.sid] = max(span.duration - child_union - pre_sum - post_sum, 0.0)
    return by_id, children, root, local, pre, post


def replay(
    span: Span,
    children: dict[str, list[Span]],
    local: dict[str, float],
    pre: dict[str, float],
    post: dict[str, float],
    service: str | None,
    normal_local: dict[tuple[str, str], float],
    normal_edge: dict[tuple[str, str, str, str], tuple[float, float]],
) -> float:
    """Replay one span tree, optionally restoring one service's cost."""
    own_local = local[span.sid]
    if service is not None and span.service == service:
        own_local = normal_local.get((span.service, span.operation), own_local)
    total = own_local
    for child in children.get(span.sid, []):
        child_duration = replay(
            child, children, local, pre, post, service, normal_local, normal_edge
        )
        if service is not None and child.service == service:
            edge = (span.service, span.operation, child.service, child.operation)
            gap_pre, gap_post = normal_edge.get(edge, (pre[child.sid], post[child.sid]))
        else:
            gap_pre, gap_post = pre[child.sid], post[child.sid]
        total += gap_pre + child_duration + gap_post
    return total


def score_request(
    spans: list[Span],
    normal_local: dict[tuple[str, str], float],
    normal_edge: dict[tuple[str, str, str, str], tuple[float, float]],
    dataset: str,
):
    """Return ``(root, deltas, local)`` for one request.

    ``deltas[i]`` is the entry-latency drop when service ``i`` is restored.
    """
    parsed = decompose_tree(spans, dataset)
    if parsed is None:
        return None
    _, children, root, local, pre, post = parsed
    if root.duration <= 0:
        return None
    deltas = {}
    for service in {span.service for span in spans}:
        restored = replay(
            root, children, local, pre, post, service, normal_local, normal_edge
        )
        deltas[service] = max(root.duration - restored, 0.0)
    return root, deltas, local


def rank(scores: dict[str, float], root: str) -> int | None:
    if root not in scores:
        return None
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    for index, (name, _) in enumerate(ordered):
        if name == root:
            return index + 1
    return None


def hit_metrics(ranks: list[int]) -> dict:
    if not ranks:
        return {"events": 0, "hit_at_1": None, "hit_at_3": None, "mrr": None}
    return {
        "events": len(ranks),
        "hit_at_1": sum(r == 1 for r in ranks) / len(ranks),
        "hit_at_3": sum(r <= 3 for r in ranks) / len(ranks),
        "mrr": sum(1.0 / r for r in ranks) / len(ranks),
    }
