"""The feature cache: features not decisions, TTL/max-entries, and the counters."""

from __future__ import annotations

import time

from llm_guardian.cache import FeatureCache
from llm_guardian.provenance import Provenance
from llm_guardian.settings import CacheSettings
from tests.helpers import make_finding, make_outcome


def make_cache(**kwargs) -> FeatureCache:
    return FeatureCache(CacheSettings(**kwargs))


def key(cache: FeatureCache, digest: str = "d", prov: Provenance = Provenance.USER) -> str:
    return FeatureCache.make_key(digest, prov, "fp", "pv")


def test_put_then_get_is_a_hit() -> None:
    cache = make_cache()
    k = key(cache)
    outcomes = (make_outcome(make_finding()),)
    cache.put(k, outcomes)
    got = cache.get(k)
    assert got is not None
    assert got[0].cached is True
    assert got[0].findings[0].kind == "instruction_override"
    stats = cache.stats()
    assert (stats.hits, stats.misses, stats.entries) == (1, 0, 1)


def test_miss_counts() -> None:
    cache = make_cache()
    assert cache.get(key(cache)) is None
    assert cache.stats().misses == 1


def test_key_depends_on_every_component() -> None:
    base = FeatureCache.make_key("d", Provenance.USER, "fp", "pv")
    assert base != FeatureCache.make_key("d2", Provenance.USER, "fp", "pv")
    assert base != FeatureCache.make_key("d", Provenance.TOOL_RESULT, "fp", "pv")
    assert base != FeatureCache.make_key("d", Provenance.USER, "fp2", "pv")
    assert base != FeatureCache.make_key("d", Provenance.USER, "fp", "pv2")


def test_ttl_expiry() -> None:
    cache = make_cache(ttl_seconds=0.02)
    k = key(cache)
    cache.put(k, (make_outcome(make_finding()),))
    time.sleep(0.03)
    assert cache.get(k) is None
    assert cache.stats().evictions >= 1


def test_max_entries_eviction() -> None:
    cache = make_cache(max_entries=1)
    cache.put(key(cache, "a"), (make_outcome(make_finding()),))
    cache.put(key(cache, "b"), (make_outcome(make_finding()),))
    assert cache.stats().entries == 1
    assert cache.stats().evictions == 1


def test_disabled_cache_is_a_noop() -> None:
    cache = make_cache(enabled=False)
    k = key(cache)
    cache.put(k, (make_outcome(make_finding()),))
    assert cache.get(k) is None
    assert cache.stats().entries == 0


def test_clear() -> None:
    cache = make_cache()
    cache.put(key(cache), (make_outcome(make_finding()),))
    cache.clear()
    assert cache.stats().entries == 0


def test_stats_prunes_expired() -> None:
    cache = make_cache(ttl_seconds=0.01)
    cache.put(key(cache), (make_outcome(make_finding()),))
    time.sleep(0.02)
    assert cache.stats().entries == 0


def test_stats_reports_configuration() -> None:
    cache = make_cache(max_entries=7, ttl_seconds=12.0)
    stats = cache.stats()
    assert stats.max_entries == 7
    assert stats.ttl_seconds == 12.0


def test_cache_stores_features_not_decisions() -> None:
    """The cached value is findings only: no verdict, no policy decision, no authorization."""
    cache = make_cache()
    k = key(cache)
    cache.put(k, (make_outcome(make_finding()),))
    got = cache.get(k)
    assert got is not None
    for outcome in got:
        assert not hasattr(outcome, "decision")
        for finding in outcome.findings:
            assert not hasattr(finding, "decision")
    # The cache exposes no method that could store a policy decision or an authorization.
    public = {name for name in dir(cache) if not name.startswith("_")}
    assert not (public & {"decide", "authorize", "store_verdict", "principal"})
