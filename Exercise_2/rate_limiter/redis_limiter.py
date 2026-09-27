"""Redis 分布式滑动窗口限流器：ZSET + Lua 原子脚本，Redis Cluster 兼容，不硬依赖 redis 库。"""

import time
import uuid
from typing import Callable, Optional

# 严格滑动窗口判定脚本：KEYS[1]=限流 key，ARGV=[now_ms, window_ms, limit, member]
_ALLOW_SCRIPT_LUA = """
-- 1. 清理窗口外过期成员（左开右闭：score <= now-window 视为过期）
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1] - ARGV[2])
-- 2. 判断
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[3]) then
    redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])
    redis.call('PEXPIRE', KEYS[1], ARGV[2])
    return 1
end
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 0
"""


def make_client(url: str = "redis://localhost:6379/0"):
    """延迟 import redis 并创建 redis-py 客户端；未安装时抛带安装提示的 ImportError。

    本模块不硬依赖 redis：只有调用本函数时才尝试导入，纯单机场景可零依赖使用。
    """
    try:
        import redis
    except ImportError as exc:
        raise ImportError(
            "未安装 redis 库，无法创建 Redis 客户端；请先执行: pip install redis"
        ) from exc
    return redis.Redis.from_url(url)


class RedisRateLimiter:
    """基于 Redis ZSET 的分布式严格滑动窗口限流器，窗口为左开右闭 ``(now-window, now]``。

    数据结构：每个用户一个 key（``f"{key_prefix}:{user_id}"``），ZSET 的 member 为
    唯一化的毫秒时间戳、score 即时间戳，窗口内成员数即已放行请求数。

    为何用 Lua 而非 MULTI/EXEC：MULTI/EXEC 只是把命令入队后顺序执行，队列内部无法
    "先读 ZCARD 计数、再按计数结果条件写 ZADD"（入队命令间不允许数据依赖）；改用
    WATCH + MULTI + EXEC 则并发冲突时 EXEC 失败、需整体重试，热点用户竞争激烈时
    会形成重试风暴。Lua 脚本在 Redis 单线程中原子执行，一次 RTT 完成
    "清理-判定-写入-续期"。

    member 唯一性：``f"{now_ms:013d}-{uuid4.hex[:12]}"``。若 member 只用毫秒时间戳，
    同一毫秒内到达的并发请求会因 member 相同而互相 ZADD 覆盖，导致少计数。

    Redis Cluster：全部操作均为单 key 命令，天然路由到该 key 所在 slot，无需 hash tag。

    TTL 与清理：每次调用都 PEXPIRE 窗口时长，不活跃用户的 key 到期自动删除，
    因此 Redis 版无需清扫线程或淘汰堆。

    时钟权衡：默认用客户端毫秒时钟（可经 clock 注入校准时钟），多机时钟偏差会轻微
    影响窗口精度；改用 Redis 服务端 TIME 虽可消除偏差，但脚本内取 TIME 属非确定性
    命令，受脚本复制/缓存的版本约束，工程上常以 NTP 校准 + 客户端传时折衷。

    client 为鸭子类型：任何具备 ``register_script()`` 的 redis-py 兼容客户端均可注入
    （redis-py 的 Script 对象自动处理 EVALSHA 失败回退 EVAL），本模块顶层不 import redis。
    """

    def __init__(self, client, window_seconds: float, limit: int,
                 key_prefix: str = "ratelimit",
                 clock: Optional[Callable[[], float]] = None):  # None 则用 time.time
        if client is None:
            raise ValueError("client 不能为 None，请注入具备 register_script() 的 redis-py 兼容客户端")
        if window_seconds <= 0:
            raise ValueError("window_seconds 必须大于 0")
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        self._client = client
        self._window_seconds = float(window_seconds)
        self._limit = limit
        self._key_prefix = key_prefix
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        # redis-py 返回 Script 对象：优先 EVALSHA，NOSCRIPT 时自动回退 EVAL 并重试
        self._script = client.register_script(_ALLOW_SCRIPT_LUA)

    def allow(self, user_id: str) -> bool:
        """判定用户本次请求是否放行；放行写入唯一 member，拒绝不写（不占额度）。"""
        now_ms = int(round(self._clock() * 1000))
        window_ms = int(round(self._window_seconds * 1000))
        # member 附加 uuid 后缀保证唯一：同毫秒并发请求互不覆盖（见类 docstring）
        member = f"{now_ms:013d}-{uuid.uuid4().hex[:12]}"
        key = f"{self._key_prefix}:{user_id}"
        return self._script(keys=[key], args=[now_ms, window_ms, self._limit, member]) == 1
