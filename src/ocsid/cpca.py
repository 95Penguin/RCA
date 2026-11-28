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

import math
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


def entry_component(spans: list[Span], dataset: str):
    """Keep only nodes connected to one explicit business entry.

    Missing-parent fragments are omitted, never attached using time or labels.
    Duplicate ids and ambiguous/missing business entries remain unsupported.
    """
    ids = {span.sid: span for span in spans}
    if not spans or len(ids) != len(spans):
        return None
    entries = [span for span in spans if span.parent not in ids and is_entry(span, dataset)]
    if len(entries) != 1:
        return None
    children = defaultdict(list)
    for span in spans:
        children[span.parent].append(span.sid)
    seen, stack = set(), [entries[0].sid]
    while stack:
        sid = stack.pop()
        if sid in seen:
            return None
        seen.add(sid)
        stack.extend(children[sid])
    return [span for span in spans if span.sid in seen]


def decompose_tree(spans: list[Span], dataset: str, gap_policy: str = "post",
                   clock_tolerance_us: float = 0., tree_policy: str = "strict"):
    """Return ``(by_id, children, root, local, pre, post)`` for one request.

    ``pre[c.sid]`` / ``post[c.sid]`` are the gaps around child ``c`` inside its
    parent. Every inter-child gap belongs to exactly one edge (post by default).
    Local is the remaining cost under this ownership assumption, not measured CPU time.
    Invalid, incomplete or parallel trees return None.
    """
    if tree_policy not in {"strict", "entry-component"}:
        raise ValueError("tree_policy must be strict or entry-component")
    if tree_policy == "entry-component":
        spans = entry_component(spans, dataset)
        if spans is None:
            return None
    if clock_tolerance_us:
        from .clocks import align_span_clocks
        spans = align_span_clocks(spans, clock_tolerance_us)
        if spans is None:
            return None
    if gap_policy not in {"post", "pre"}:
        raise ValueError("gap_policy must be post or pre")
    if not spans or any(not math.isfinite(s.start) or not math.isfinite(s.duration)
                        or s.duration < 0 for s in spans):
        return None
    by_id = {span.sid: span for span in spans}
    if len(by_id) != len(spans):
        return None
    children: dict[str, list[Span]] = defaultdict(list)
    roots: list[Span] = []
    for span in spans:
        if by_id.get(span.parent) is None:
            roots.append(span)
        else:
            children[span.parent].append(span)
    if len(roots) != 1:
        return None
    entries = [span for span in roots if is_entry(span, dataset)]
    root = (
        min(entries, key=lambda s: (s.start, -s.duration))
        if entries
        else max(roots, key=lambda s: s.duration)
    )
    # A complete serial tree is required. Disconnected cycles, overlapping
    # children and clock-skewed/out-of-parent spans are explicitly unsupported.
    seen, stack = set(), [root]
    while stack:
        node = stack.pop()
        if node.sid in seen:
            return None
        seen.add(node.sid)
        stack.extend(children.get(node.sid, []))
    if len(seen) != len(spans):
        return None
    local, pre, post = {}, {}, {}
    for span in spans:
        kids = sorted(children.get(span.sid, []), key=lambda s: s.start)
        if not kids:
            local[span.sid] = span.duration
            continue
        previous_end = span.start
        for child in kids:
            if child.start < previous_end or child.end > span.end:
                return None
            pre[child.sid] = 0.0
            post[child.sid] = 0.0
            previous_end = child.end
        pre[kids[0].sid] = kids[0].start - span.start
        post[kids[-1].sid] = span.end - kids[-1].end
        for left, right in zip(kids, kids[1:]):
            gap = right.start - left.end
            if gap_policy == "post":
                post[left.sid] = gap
            else:
                pre[right.sid] = gap
        local[span.sid] = max(span.duration - sum(c.duration for c in kids)
                              - sum(pre[c.sid] + post[c.sid] for c in kids), 0.0)
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
    restoration: str = "excess",
) -> float:
    """Replay one span tree, optionally restoring one service's cost."""
    if restoration not in {"replace", "excess"}:
        raise ValueError("restoration must be replace or excess")
    own_local = local[span.sid]
    if service is not None and span.service == service:
        reference = normal_local.get((span.service, span.operation), own_local)
        own_local = min(own_local, reference) if restoration == "excess" else reference
    total = own_local
    for child in children.get(span.sid, []):
        child_duration = replay(
            child, children, local, pre, post, service, normal_local, normal_edge, restoration
        )
        if service is not None and child.service == service:
            edge = (span.service, span.operation, child.service, child.operation)
            gap_pre, gap_post = normal_edge.get(edge, (pre[child.sid], post[child.sid]))
        else:
            gap_pre, gap_post = pre[child.sid], post[child.sid]
        if restoration == "excess":
            gap_pre = min(gap_pre, pre[child.sid])
            gap_post = min(gap_post, post[child.sid])
        total += gap_pre + child_duration + gap_post
    return total


def score_request(
    spans: list[Span],
    normal_local: dict[tuple[str, str], float],
    normal_edge: dict[tuple[str, str, str, str], tuple[float, float]],
    dataset: str,
    restoration: str = "excess",
    gap_policy: str = "post",
    clock_tolerance_us: float = 0.,
    tree_policy: str = "strict",
    *, components: str = "all", edge_owner: str = "callee",
):
    """Return ``(root, deltas, local)`` for one request.

    ``deltas[i]`` is the entry-latency drop when service ``i`` is restored.
    """
    parsed = decompose_tree(spans, dataset, gap_policy, clock_tolerance_us, tree_policy)
    return score_decomposition(parsed, normal_local, normal_edge, dataset, restoration,
                               components=components, edge_owner=edge_owner)


def score_decomposition(parsed, normal_local, normal_edge, dataset,
                        restoration="excess", *, components="all", edge_owner="callee"):
    """Score an already validated tree; optional controls do not alter the tree.

    Components and edge ownership are experimental ablations of the same ledger.
    """
    if parsed is None:
        return None
    _, children, root, local, pre, post = parsed
    # A health-check/internal RPC fragment is not a user entry request.
    if root.duration <= 0 or not is_entry(root, dataset):
        return None
    if components not in {"all", "local", "edge"}:
        raise ValueError("components must be all, local or edge")
    if edge_owner not in {"callee", "caller"}:
        raise ValueError("edge_owner must be callee or caller")
    if restoration not in {"replace", "excess"}:
        raise ValueError("restoration must be replace or excess")
    # On a serial additive tree this is algebraically identical to one replay
    # per service, but O(spans) instead of O(services * spans).
    by_id = parsed[0]
    deltas = defaultdict(float)
    def gain(observed, reference):
        return max(observed - reference, 0.0) if restoration == "excess" else observed - reference
    for span in by_id.values():
        key = (span.service, span.operation)
        deltas[span.service] += 0.0
        if components in {"all", "local"}:
            deltas[span.service] += gain(local[span.sid], normal_local.get(key, local[span.sid]))
        parent = by_id.get(span.parent)
        if parent is not None and components in {"all", "edge"}:
            edge = (parent.service, parent.operation, span.service, span.operation)
            ref_pre, ref_post = normal_edge.get(edge, (pre[span.sid], post[span.sid]))
            owner = span.service if edge_owner == "callee" else parent.service
            deltas[owner] += gain(pre[span.sid], ref_pre) + gain(post[span.sid], ref_post)
    deltas = {service: max(value, 0.0) for service, value in deltas.items()}
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
