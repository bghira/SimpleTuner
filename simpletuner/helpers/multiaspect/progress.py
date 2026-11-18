from collections import Counter, deque
from contextlib import contextmanager
from threading import RLock


class LegacySamplerProgress:
    """Keep only unacknowledged batches alongside the sampler's seen counts."""

    def __init__(self, seen, state):
        self.lock = RLock()
        self.reset(seen, state)

    def reset(self, seen, state):
        self.fetched = self.consumed = state
        self.seen = seen
        self.pending = deque()

    def record(self, seen, state, refs):
        self.fetched = {**state, "position": self.fetched["position"] + 1, "refs": tuple(refs)}
        self.pending.append((self.fetched["position"], seen, self.fetched["refs"]))

    def acknowledge(self, state):
        with self.lock:
            if state["position"] != self.consumed["position"] + 1:
                raise ValueError("Legacy sampler batches must be acknowledged in order.")
            if self.pending:
                if self.pending[0][0] != state["position"]:
                    raise ValueError("Consumed legacy batch differs from the fetched batch.")
                _, self.seen, _ = self.pending.popleft()
            else:
                # Context-parallel followers receive the batch without running this sampler.
                if state["repeat"] != self.consumed["repeat"]:
                    self.seen = {}
                for ref in state["refs"]:
                    self.seen[ref] = self.seen.get(ref, 0) + 1
            self.consumed = state

    @contextmanager
    def checkpoint_seen(self):
        with self.lock:
            outstanding = Counter(ref for _, seen, refs in self.pending if seen is self.seen for ref in refs)
            original = {ref: self.seen[ref] for ref in outstanding}
            try:
                for ref, count in outstanding.items():
                    remaining = original[ref] - count
                    if remaining:
                        self.seen[ref] = remaining
                    else:
                        del self.seen[ref]
                yield self.seen
            finally:
                self.seen.update(original)
