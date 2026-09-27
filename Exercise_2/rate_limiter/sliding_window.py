"""单机滑动窗口限流器：每用户时间戳队列实现严格滑动窗口，线程安全，纯标准库。"""

import heapq
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple


class SlidingWindowRateLimiter:
    """单机滑动窗口限流器，窗口为左开右闭区间 ``(now - window_seconds, now]``。

    为何用"每用户时间戳 deque"而不是固定窗口计数：
    固定窗口在窗口接缝处最多放过 2*limit 的突发流量；本实现为每个用户维护窗口内的
    精确放行时间戳队列，判定前先从队首弹出全部已滑出窗口的时间戳——每个时间戳至多
    入队/出队各一次，均摊 O(1)——再以 ``len(deque) >= limit`` 判定，得到无近似误差
    的严格滑动窗口语义。

    为何整个 allow 过程持 threading.Lock：
    单实例会被多线程并发调用，"清理 -> 判定 -> 写入"必须原子执行，否则并发下各自
    读到同一份 ``len(deque)`` 会放行超额请求；锁内只有内存操作，临界区极短。
    跨进程/跨实例场景请改用 Redis 版。默认时钟为 time.time；若部署环境存在校时
    回拨，可注入 ``time.monotonic`` 提升健壮性（时钟可注入也是测试用假时钟的基础）。

    三种清理机制的分工与取舍：
    1. 惰性清理（allow 内）：只顺带清理当前被访问用户的过期时间戳，均摊 O(1)，
       缺点是长期不被访问的用户不会被回收；
    2. 后台清扫（_sweep）：删除"最近一次放行已整窗过期"的不活跃用户，是常态化的
       内存回收手段；sweep_interval=None 时不启动线程（测试可直接调用 _sweep）；
    3. 容量淘汰（_evict，兜底）：追踪用户数超过 max_users 时按最久未活跃淘汰，
       被淘汰用户下次访问等同于全新用户（额度重置）。正常流量靠 1/2 控制内存，
       淘汰只在瞬时大量一次性用户击穿 max_users 时触发。

    被拒绝的请求不写入记录的权衡：若拒绝也记账，被刷用户的队列会长期被垃圾请求
    占满、恢复更慢，且内存占用被攻击流量放大；生产上多数限流器（如 NGINX
    limit_req）只对放行请求记账，本实现遵循该惯例。
    """

    def __init__(self, window_seconds: float, limit: int,
                 max_users: int = 100_000,
                 sweep_interval: Optional[float] = None,   # None=不开后台清扫线程
                 clock: Callable[[], float] = time.time):  # 时钟可注入，测试用假时钟
        if window_seconds <= 0:
            raise ValueError("window_seconds 必须大于 0")
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        if max_users <= 0:
            raise ValueError("max_users 必须大于 0")
        self._window_seconds = float(window_seconds)
        self._limit = limit
        self._max_users = max_users
        self._clock = clock
        # 每用户窗口内的放行时间戳，队首最旧
        self._hits: Dict[str, Deque[float]] = {}
        # 用户最近一次放行时间，供清扫与淘汰使用（与 _hits 的 key 集合始终一致）
        self._last_access: Dict[str, float] = {}
        # 淘汰堆：元素 (last_access, user_id)，惰性失效
        self._heap: List[Tuple[float, str]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sweeper: Optional[threading.Thread] = None
        if sweep_interval is not None:
            self._sweeper = threading.Thread(
                target=self._sweep_loop, args=(sweep_interval,),
                name="rate-limiter-sweeper", daemon=True)
            self._sweeper.start()

    def allow(self, user_id: str) -> bool:
        """判定用户本次请求是否放行；放行则记录时间戳，拒绝则不记账。"""
        now = self._clock()
        with self._lock:
            boundary = now - self._window_seconds
            dq = self._hits.get(user_id)
            if dq is not None:
                # 惰性清理：左开右闭，队首时间戳 <= now-window 即已滑出窗口
                while dq and dq[0] <= boundary:
                    dq.popleft()
                if not dq:
                    # 窗口内已无记录，整体回收该用户（两个 dict 同步删除）
                    del self._hits[user_id]
                    self._last_access.pop(user_id, None)
                    dq = None
            if dq is None:
                dq = deque()
                self._hits[user_id] = dq
            if len(dq) >= self._limit:
                # 被拒绝的请求不写入记录（不占额度），设计权衡见类 docstring
                return False
            dq.append(now)
            self._last_access[user_id] = now
            if len(self._hits) > self._max_users:
                self._evict()
            return True

    def _evict(self) -> None:
        """容量兜底：按最久未活跃（_last_access 最小）淘汰用户，直到 <= max_users。

        堆条目惰性维护：仅当堆为空时按当前 _last_access 全量重建（本方法持锁、
        状态冻结，新条目必然有效）；用户此后再次活跃会使旧条目过期，弹出时与
        _last_access 比对，不一致则丢弃重弹。相比"每次放行都入堆"，堆规模受
        用户数上界约束，避免堆内存随总请求量无界增长。
        """
        while len(self._hits) > self._max_users:
            if not self._heap:
                self._heap = [(ts, uid) for uid, ts in self._last_access.items()]
                heapq.heapify(self._heap)
            ts, uid = heapq.heappop(self._heap)
            if self._last_access.get(uid) != ts:
                continue  # 惰性失效：条目已过期，丢弃重弹
            del self._hits[uid]
            del self._last_access[uid]

    def _sweep(self, now: Optional[float] = None) -> int:
        """清扫：删除 ``last_access <= now - window_seconds`` 的不活跃用户，返回删除数。

        判定安全性：last_access 是该用户队列内的最大时间戳，它滑出窗口即意味着
        该用户整个队列均已过期，可整体回收。测试可直接调用本方法，不依赖后台线程。
        """
        if now is None:
            now = self._clock()
        boundary = now - self._window_seconds
        with self._lock:
            stale = [uid for uid, ts in self._last_access.items() if ts <= boundary]
            for uid in stale:
                del self._hits[uid]
                del self._last_access[uid]
            return len(stale)

    def _sweep_loop(self, interval: float) -> None:
        """后台清扫线程主循环：每 interval 秒清扫一次，stop() 后退出。"""
        while not self._stop_event.wait(interval):
            self._sweep()

    def stop(self) -> None:
        """停止后台清扫线程：置位停止事件并 join 等待线程退出。"""
        self._stop_event.set()
        if self._sweeper is not None:
            self._sweeper.join()

    def active_users(self) -> int:
        """当前追踪的用户数（窗口内仍有放行记录的用户）。"""
        with self._lock:
            return len(self._hits)
