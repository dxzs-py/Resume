"""rate_limiter 测试：假时钟离线测试单机滑动窗口；Redis 版在 fakeredis 不可用时跳过。"""

import threading
import time

import pytest

from rate_limiter.redis_limiter import RedisRateLimiter, make_client
from rate_limiter.sliding_window import SlidingWindowRateLimiter


class FakeClock:
    """可手动推进的假时钟，使窗口语义测试离线、无睡眠。"""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


@pytest.fixture
def clock():
    return FakeClock()


# ---------- 基础限流 ----------

def test_limit_enforced(clock):
    limiter = SlidingWindowRateLimiter(10, 3, clock=clock)
    assert [limiter.allow("u1") for _ in range(3)] == [True, True, True]
    assert limiter.allow("u1") is False  # 第 4 次超限


def test_users_are_isolated(clock):
    limiter = SlidingWindowRateLimiter(10, 2, clock=clock)
    assert limiter.allow("a") and limiter.allow("a")
    assert limiter.allow("a") is False
    assert limiter.allow("b") is True  # 不同用户互不影响


def test_invalid_params(clock):
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(0, 3, clock=clock)
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(10, 0, clock=clock)


# ---------- 滑动窗口语义（非固定窗口） ----------

def test_sliding_window_recovers_gradually(clock):
    """额度随时间戳逐个滑出而逐个恢复，而非整窗重置。"""
    limiter = SlidingWindowRateLimiter(10, 2, clock=clock)
    limiter.allow("u")          # t=1000.0
    clock.advance(5)
    limiter.allow("u")          # t=1005.0
    clock.advance(4.9)          # t=1009.9：两条记录都在窗口内
    assert limiter.allow("u") is False
    clock.advance(0.2)          # t=1010.1：t=1000.0 已滑出（左开右闭），恢复 1 个额度
    assert limiter.allow("u") is True
    assert limiter.allow("u") is False  # 其余额度仍被 t=1005.0 占用
    clock.advance(5.0)          # t=1015.1：t=1005.0 也滑出
    assert limiter.allow("u") is True


def test_no_burst_across_window_boundary(clock):
    """固定窗口在接缝处会放过 2×limit，滑动窗口不允许。"""
    limiter = SlidingWindowRateLimiter(10, 2, clock=clock)
    clock.advance(9.9)
    assert limiter.allow("u") and limiter.allow("u")  # 旧窗口末尾打满
    clock.advance(0.2)                                # 跨过窗口边界
    assert limiter.allow("u") is False                # 新窗口内依然受限


# ---------- 拒绝不记账 ----------

def test_rejected_requests_do_not_consume_quota(clock):
    limiter = SlidingWindowRateLimiter(10, 2, clock=clock)
    limiter.allow("u") and limiter.allow("u")
    for _ in range(50):          # 大量被拒请求不写入记录
        limiter.allow("u")
    clock.advance(10.1)          # 两条放行记录全部滑出
    assert limiter.allow("u") is True  # 若拒绝记账，此处队列仍满、应失败


# ---------- 内存清理 ----------

def test_lazy_cleanup_reclaims_idle_user(clock):
    limiter = SlidingWindowRateLimiter(10, 3, clock=clock)
    limiter.allow("u")
    clock.advance(10.1)          # 整窗过期后再次访问
    limiter.allow("u")
    assert limiter.active_users() == 1  # 旧记录被惰性回收，未累积


def test_sweep_removes_inactive_users(clock):
    limiter = SlidingWindowRateLimiter(10, 3, clock=clock)
    limiter.allow("a")
    clock.advance(3)
    limiter.allow("b")
    clock.advance(8)             # a 已过期（last=1000.0 <= 1011-10），b 未过期
    removed = limiter._sweep()
    assert removed == 1
    assert limiter.active_users() == 1


def test_evict_removes_oldest_user_on_overflow(clock):
    limiter = SlidingWindowRateLimiter(10, 5, max_users=2, clock=clock)
    limiter.allow("a")
    clock.advance(1)
    limiter.allow("b")
    clock.advance(1)
    assert limiter.allow("c") is True        # 超 max_users，触发淘汰
    assert limiter.active_users() == 2
    assert limiter.allow("a") is True        # a（最久未活跃）被逐出，等同新用户
    assert limiter.allow("a") is True


def test_background_sweeper_thread():
    """后台清扫线程用真实时钟端到端验证：窗口过后用户被自动回收。"""
    limiter = SlidingWindowRateLimiter(0.1, 1, sweep_interval=0.05)
    try:
        assert limiter.allow("u") is True
        deadline = time.time() + 2
        while limiter.active_users() > 0 and time.time() < deadline:
            time.sleep(0.02)
        assert limiter.active_users() == 0
    finally:
        limiter.stop()


# ---------- 并发正确性 ----------

def test_concurrent_allow_never_exceeds_limit():
    limiter = SlidingWindowRateLimiter(60, 100)
    allowed = []
    lock = threading.Lock()

    def worker():
        local = sum(limiter.allow("hot") for _ in range(10))
        with lock:
            allowed.append(local)

    threads = [threading.Thread(target=worker) for _ in range(20)]  # 200 次并发
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(allowed) == 100  # 放行总数恰为 limit


# ---------- Redis 版（需要 fakeredis+lupa，未安装则跳过） ----------

def _fakeredis_available() -> bool:
    try:
        import fakeredis  # noqa: F401
        import lupa       # noqa: F401  # fakeredis 执行 Lua 依赖
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _fakeredis_available(), reason="需要 fakeredis 与 lupa")
def test_redis_limiter_allow_and_expiry():
    import fakeredis

    client = fakeredis.FakeStrictRedis()
    limiter = RedisRateLimiter(client, window_seconds=10, limit=2)
    assert limiter.allow("u") is True
    assert limiter.allow("u") is True
    assert limiter.allow("u") is False
    assert client.exists("ratelimit:u") == 1
    assert 0 < client.pttl("ratelimit:u") <= 10_000  # 每次调用续期窗口时长


@pytest.mark.skipif(not _fakeredis_available(), reason="需要 fakeredis 与 lupa")
def test_redis_limiter_sliding_recovery():
    import fakeredis

    client = fakeredis.FakeStrictRedis()
    limiter = RedisRateLimiter(client, window_seconds=0.2, limit=1,
                               clock=lambda: time.time())
    assert limiter.allow("u") is True
    assert limiter.allow("u") is False
    time.sleep(0.25)             # 窗口整体滑过，额度恢复
    assert limiter.allow("u") is True


def test_redis_limiter_make_client_missing_package(monkeypatch):
    """未安装 redis 库时 make_client 抛出带安装提示的 ImportError。"""
    import builtins
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("No module named 'redis'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install redis"):
        make_client("redis://localhost:6379/0")
