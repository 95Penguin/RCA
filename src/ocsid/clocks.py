"""Bounded clock reconciliation for serial distributed traces.

The entry duration stays fixed. Children are translated together with their
subtrees by at most the declared timestamp resolution. Durations are clipped
only when a child cannot fit in its parent within that resolution. Larger
inconsistencies and actual parallel branches are rejected, not serialized.
"""
from dataclasses import replace


def align_span_clocks(spans, tolerance_us=1000.):
    if tolerance_us < 0:
        raise ValueError("tolerance_us must be nonnegative")
    ids = {s.sid: s for s in spans}
    if not spans or len(ids) != len(spans):
        return None
    children = {}
    roots = []
    for span in spans:
        if span.parent in ids:
            children.setdefault(span.parent, []).append(span.sid)
        else:
            roots.append(span.sid)
    if len(roots) != 1:
        return None
    visited = set()
    def shift(sid, delta):
        todo = [sid]
        seen = set()
        while todo:
            key = todo.pop()
            if key in seen:
                return False
            seen.add(key)
            ids[key] = replace(ids[key], start=ids[key].start + delta)
            todo.extend(children.get(key, []))
        return True
    def visit(sid):
        if sid in visited:
            return False
        visited.add(sid)
        parent = ids[sid]
        previous_end = parent.start
        for key in sorted(children.get(sid, []), key=lambda k: ids[k].start):
            child = ids[key]
            start = max(child.start, previous_end)
            if start + child.duration > parent.end:
                start = max(previous_end, parent.end - child.duration)
            duration = min(child.duration, parent.end - start)
            if (duration < 0 or abs(start - child.start) > tolerance_us
                    or abs(duration - child.duration) > tolerance_us):
                return False
            if start != child.start and not shift(key, start - child.start):
                return False
            ids[key] = replace(ids[key], duration=duration)
            if not visit(key):
                return False
            previous_end = ids[key].end
        return True
    if not visit(roots[0]) or len(visited) != len(spans):
        return None
    return [ids[s.sid] for s in spans]
