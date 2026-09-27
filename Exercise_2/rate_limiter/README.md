# 接口限流器设计（滑动窗口，按 userId）

对应《架构设计题》第 6 题。本目录含两个实现：

- `sliding_window.py`：单机版，线程安全，纯标准库，零依赖
- `redis_limiter.py`：Redis 集群版，ZSET + Lua 原子脚本，Redis Cluster 兼容

```python
from rate_limiter.sliding_window import SlidingWindowRateLimiter

limiter = SlidingWindowRateLimiter(window_seconds=60, limit=100,
                                   sweep_interval=30)  # 后台清扫周期
if limiter.allow(user_id):
    handle_request()
else:
    return 429
```

## 需求

1. 按 userId 限流：任意连续 `windowSeconds` 秒内每用户最多 `limit` 次
2. `bool allow(String userId)`
3. 严格滑动窗口（非固定窗口）
4. 不活跃用户数据可清理，内存有界

## 1. 数据结构与核心逻辑

### 单机版

| 数据 | 结构 | 用途 |
|---|---|---|
| `hits` | `Dict[userId, Deque[float]]` | 窗口内的放行时间戳，队首最旧 |
| `last_access` | `Dict[userId, float]` | 最近一次放行时间，供清扫/淘汰 |
| 淘汰堆 | 最小堆 `(last_access, userId)` | 容量兜底，条目惰性失效 |

```python
def allow(user_id):
    now = clock()
    with lock:                              # "清理→判定→写入"必须原子，否则并发读到同一
        boundary = now - window_seconds     # len(deque) 会放行超额请求
        dq = hits.get(user_id)
        if dq:
            while dq and dq[0] <= boundary: # 惰性清理：弹出已滑出窗口的时间戳
                dq.popleft()                # 左开右闭：ts <= now-window 即过期
            if not dq:
                del hits[user_id]; last_access.pop(user_id, None)
                dq = None                   # 窗口内无记录，整用户回收
        if dq is None:
            dq = hits[user_id] = deque()
        if len(dq) >= limit:
            return False                    # 拒绝不记账：不占额度，内存不被刷子放大
        dq.append(now)
        last_access[user_id] = now
        if len(hits) > max_users:
            evict_oldest()                  # 兜底：按最久未活跃淘汰
        return True
```

**为何不用固定窗口计数**：固定窗口在窗口接缝处最多放过 `2×limit` 的突发（前一窗口末尾 + 后一窗口开头），不满足"任意连续 windowSeconds"语义。

**内存不涨的三层机制**：

1. **惰性清理**（`allow` 内）：只顺带清理当前被访问用户的过期时间戳，均摊 O(1)；缺点是长期不被访问的用户不会被回收。
2. **后台清扫**：定时删除"最近一次放行已整窗过期"的用户——`last_access` 是该用户队列内最大时间戳，它滑出窗口即整个队列已过期，可整体回收。
3. **容量淘汰**（兜底）：追踪用户数超过 `max_users` 时按最久未活跃淘汰。正常流量靠 1/2 控制，③ 只在瞬时大量一次性用户击穿上限时触发。

**时钟**：默认 `time.time`，可注入 `time.monotonic`（防校时回拨）或假时钟（测试）。

### 复杂度

- **时间**：每个时间戳至多入队/出队各一次，`allow` 均摊 **O(1)**（单次最坏 O(弹出数)，均摊归还）。后台清扫单轮 O(活跃用户数)。
- **空间**：**O(活跃用户数 × min(limit, 窗口内请求数))**，由 `max_users` 和清扫机制双重约束，有界。

## 2. 四种限流算法的适用场景

| 算法 | 原理 | 优点 | 缺点 | 适用 |
|---|---|---|---|---|
| 固定窗口 | 每窗口一个计数器 | O(1) 内存与时间，实现最简 | 接缝处可突发 2×limit | 精度要求低的粗粒度配额、统计 |
| 滑动窗口（日志法） | 精确记录窗口内每次请求 | 严格精确，无近似误差 | 内存与请求量成正比 | 强"恰好 N 次"语义；QPS 不极高 |
| 滑动窗口计数器 | 两相邻固定窗口按时间占比加权 | O(1) 内存 | 近似值，插值误差 | 高 QPS 网关限流，精度"近似即可" |
| 令牌桶 | 桶容量 B，恒速 r 补充令牌，有令牌才放行 | 允许突发到 B，长期平均速率 r | 参数需调 | 保护下游突发容忍能力（API 网关、外部付费 API） |
| 漏桶 | 请求入队，恒定速率流出 | 输出绝对匀速，削平一切突发 | 排队增加延迟，无法突发 | 严格整形场景（短信通道、消息队列入口） |

本题语义是"任意连续窗口内最多 limit 次"，即**严格滑动窗口**：单机用日志法，集群用 Redis ZSET。

## 3. Redis 集群版设计

### Key / 数据结构

- **Key**：`ratelimit:{userId}`，每用户一个 key。所有命令均为单 key 命令，天然路由到该 key 所在 slot，**Cluster 下无需 hash tag**。
- **结构**：ZSET，`score = 毫秒时间戳`，`member = "{now_ms:013d}-{uuid12}"`。
  - member 必须唯一：若只用时间戳，同一毫秒内并发请求会因 member 相同互相 ZADD 覆盖，导致少计数。
  - 窗口内成员数 = 已放行请求数。

### 原子操作（Lua）

```lua
-- KEYS[1] = ratelimit:{userId}
-- ARGV[1]=now_ms, ARGV[2]=window_ms, ARGV[3]=limit, ARGV[4]=member
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1] - ARGV[2])  -- 1. 清过期
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[3]) then            -- 2. 判定
    redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])                   -- 3. 写入
end
redis.call('PEXPIRE', KEYS[1], ARGV[2])                             -- 4. 续期
return redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) and 0 or 1
```

**为何用 Lua 而非 MULTI/EXEC**：MULTI/EXEC 只是把命令入队顺序执行，队列内无法"先读 ZCARD、再按计数条件 ZADD"（命令间不允许数据依赖）；WATCH + MULTI + EXEC 在并发冲突时 EXEC 失败需整体重试，热点用户下形成重试风暴。Lua 脚本在 Redis 单线程中原子执行，一次 RTT 完成"清理-判定-写入-续期"。

**内存清理**：每次调用 `PEXPIRE window_ms` 续期，不活跃用户的 key 到期自动删除——Redis 版无需清扫线程或淘汰堆。

**复杂度**：单次 `allow` 一次 RTT，服务端 O(M)，M 为该用户窗口内成员数（≤ limit）。

### 工程注意事项

- **时钟偏差**：多机客户端传时间，NTP 校准下偏差轻微影响窗口精度。改用 Redis 服务端 `TIME` 可消除偏差，但脚本内取 TIME 属非确定性命令，受脚本复制/缓存版本约束，工程上常以"NTP + 客户端传时"折衷。
- **拒绝不记账**：被拒请求不 ZADD，避免被刷用户的 ZSET 长期被垃圾请求占满、恢复变慢，内存不被攻击流量放大。
- **精度换内存的替代**：若用户量极大且 limit 很高（ZSET 成员多），可退化为两窗插值（两个固定 counter 加权），O(1) 内存，接受近似误差。

## 测试

`rate_limiter` 当前无测试用例。建议用例（时钟可注入使离线测试可行）：

1. 假时钟推进：恰好 limit 次放行、第 limit+1 次拒绝
2. 滑动语义：窗口中点放行后推进半窗，验证早期请求过期、额度恢复
3. 拒绝不记账：连续拒绝后再等窗口滑过，立即放行（额度未被占）
4. 清理：假时钟推进整窗后 `_sweep` 返回删除数、`active_users` 归零
5. 淘汰：超过 `max_users` 时最久未活跃用户被逐出
6. 并发：多线程对同一 userId 并发 `allow`，总放行数 ≤ limit
