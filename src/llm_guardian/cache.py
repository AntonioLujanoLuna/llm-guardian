"""The exact-content feature cache.

The cache holds *features* -- the scanners' findings -- and nothing that resembles a decision.
There is deliberately no verdict, no approval and no principal in a key or a value: the same bytes
under a different authorization must still be re-judged, and the design forbids reusing one
caller's permission for another. A test proves exactly that (same text, different authorization,
features reused, decision recomputed).

The key is ``(segment digest, provenance, scanner-version fingerprint, preprocessing version)``:
change a rule or a scanner version and the entry is simply not found, so a stale finding can never
be served as a fresh one.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from .models import CacheStats, ScannerOutcome
from .provenance import Provenance
from .settings import CacheSettings


@dataclass
class _Entry:
    outcomes: tuple[ScannerOutcome, ...]
    expires_at: float


class FeatureCache:
    """A thread-safe, TTL- and capacity-bounded cache of scanner findings."""

    def __init__(self, settings: CacheSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # -- keys ---------------------------------------------------------------------------------
    @staticmethod
    def make_key(
        digest: str,
        provenance: Provenance,
        scanner_fingerprint: str,
        preprocessing_version: str,
    ) -> str:
        """The cache key: content, provenance and the versions that produced the findings."""
        material = "|".join((digest, provenance.value, scanner_fingerprint, preprocessing_version))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    # -- access -------------------------------------------------------------------------------
    def get(self, key: str) -> tuple[ScannerOutcome, ...] | None:
        if not self._settings.enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at <= time.monotonic():
                del self._store[key]
                self._evictions += 1
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return tuple(outcome.model_copy(update={"cached": True}) for outcome in entry.outcomes)

    def put(self, key: str, outcomes: tuple[ScannerOutcome, ...]) -> None:
        if not self._settings.enabled:
            return
        stored = tuple(outcome.model_copy(update={"cached": False}) for outcome in outcomes)
        entry = _Entry(outcomes=stored, expires_at=time.monotonic() + self._settings.ttl_seconds)
        with self._lock:
            self._store[key] = entry
            self._store.move_to_end(key)
            while len(self._store) > self._settings.max_entries:
                self._store.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            self._prune_expired_locked()
            entries = len(self._store)
        return CacheStats(
            entries=entries,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            max_entries=self._settings.max_entries,
            ttl_seconds=self._settings.ttl_seconds,
        )

    def _prune_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._store.items() if entry.expires_at <= now]
        for key in expired:
            del self._store[key]
            self._evictions += 1
