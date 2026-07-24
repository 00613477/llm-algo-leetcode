# 41. KV Cache Scheduling | KV Cache 调度
**难度：** Hard | **环境：** GPU required | **标签：** `KV Cache`, `Scheduling`, `推理优化` | **目标人群：** 推理系统与缓存工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/41_KV_Cache_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

长上下文推理里，KV Cache 不只是“能不能存下来”的问题。多个请求共享前缀、持续生成、不断进入和退出队列时，系统还要决定哪些 cache block 值得保留，哪些可以驱逐，以及高复用前缀是否应该留在更热的路径上。

本节用一个极简 `KVCacheSchedulerSim` 模拟 cache 命中、优先级评分、堆队列刷新和容量驱逐。学完后，你应该能看清 KV Cache Scheduling 的主线：用复用次数、最近访问时间和缓存大小共同决定缓存资源如何分配，而不是只按请求到达顺序管理 cache。

**关键词：** `KV cache scheduling`, `cache reuse`, `eviction`

---

## 前置阅读

**导语：** 先看前缀缓存、解码调度和 KV Cache 量化，再看缓存调度会更容易。

- [36. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](../02_PyTorch_Algorithms/36_Prefix_Caching_and_Chunked_Prefill.md)
- [38. Decode Scheduling | 解码调度](../02_PyTorch_Algorithms/38_Decode_Scheduling.md)
- [40. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化](../02_PyTorch_Algorithms/40_FP8_and_KV_Cache_Quantization.md)
- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)

## 相关阅读

**导语：** KV Cache 调度之后，可以继续看并行策略和通信 profiling。

- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.md)
- [42. Communication Profiling with NCCL | NCCL 通信性能剖析](../02_PyTorch_Algorithms/42_Communication_Profiling_with_NCCL.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

---
### Step 1: 原理与痛点

前缀缓存解决的是“相同前缀能不能复用”，PagedAttention 解决的是“KV Cache 怎么按 block 管理”。但在真实推理服务里，只知道“能复用”和“能分块”还不够：并发请求不断进来，缓存容量是有限的，系统必须决定哪些 cache 留在显存里，哪些 cache 可以被驱逐。

KV Cache Scheduling 关注的就是这个决策过程。它不是单纯的缓存存取，而是在回答三个问题：

- **谁更值得保留**：复用次数高、刚刚访问过、被多个请求共享的前缀，通常应该保留；
- **谁应该被驱逐**：长期未访问、体积较大、复用价值低的缓存，应该优先释放；
- **怎么避免错误驱逐**：缓存状态会不断更新，调度器需要识别旧的优先级记录，不能用过期信息做决策。

这一步的核心直觉是：KV Cache 的价值不是固定的，它会随着请求访问、复用次数和时间变化动态改变。

### Step 2: 代码实现框架

下面的代码会实现一个最小 cache scheduler。它不模拟真实 KV 张量，而是用 `prefix -> CacheEntry` 表示一段可复用缓存，并记录它的大小、命中次数、最近访问时间和保留优先级。

代码拆成五个动作：

| 动作 | 对应方法 | 作用 |
|------|----------|------|
| 评分 | `_score` | 根据复用次数、最近访问和大小计算缓存保留价值 |
| 刷新队列 | `_refresh_queue` | 把最新 priority 写入堆队列，供后续驱逐使用 |
| 容量驱逐 | `_evict_until_fit` | 当容量不足时，持续移除低优先级缓存 |
| 访问缓存 | `touch` | 处理一次 prefix 访问：命中则更新，未命中则新增 |
| 状态导出 | `snapshot` | 按优先级输出当前缓存状态，便于观察调度结果 |

这里要特别注意：堆队列里可能存在同一个 prefix 的多条历史记录。因为每次命中都会刷新 priority，旧记录不会立刻删除，而是在弹出时用 `is_stale` 判断是否过期。这是很多缓存调度器里常见的懒删除思路。

### Step 3: 核心机制

本节用一个简化评分函数来表达缓存保留价值：

\[
score = reuse\_bonus + 0.5 	imes recency - 0.25 	imes size\_penalty
\]

其中：

- `reuse_bonus` 来自命中次数，命中越多说明未来继续复用的可能性越高；
- `recency` 来自最近访问时间，越新的缓存越可能还在热路径上；
- `size_penalty` 来自缓存大小，越大的缓存占用越多显存，驱逐收益也更高。

这个公式不是工业系统的固定答案，而是一个教学用的可解释策略。它把 LFU（看复用次数）、LRU（看最近访问）和容量成本（看大小）合在一起，让读者看到 cache scheduling 的基本权衡：**保留高价值缓存，同时在容量不足时释放低价值缓存**。

### Step 4: 动手实战

**要求**：请补全下方 `KVCacheSchedulerSim`，实现一个极简版的 KV Cache 调度器。你需要重点完成五个关键位置：计算保留分数、刷新堆队列、识别过期堆项、创建新缓存条目，以及按优先级导出缓存状态。

完成后观察测试中的日志：`add` 表示新增缓存，`reuse` 表示命中复用，`evict` 表示容量不足时触发驱逐。只要这三类事件能串起来，就说明你已经跑通了“复用 -> 评分 -> 驱逐 -> 快照”的缓存调度闭环。


```python
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

```


```python
@dataclass(order=True)
class CacheEntry:
    priority: float
    last_used: int
    prefix: str = field(compare=False)
    hits: int = field(default=0, compare=False)
    bytes: int = field(default=0, compare=False)


class KVCacheSchedulerSim:
    """极简版 KV Cache 调度器。"""

    def __init__(self, capacity_bytes: int = 1024):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self.time = 0
        self.entries: Dict[str, CacheEntry] = {}
        self.queue: List[Tuple[float, int, str]] = []
        self.log: List[str] = []

    def _score(self, hits: int, size: int, last_used: int) -> float:
        recency = 1.0 / (1.0 + max(self.time - last_used, 0))
        reuse_bonus = float(hits)
        size_penalty = size / max(self.capacity_bytes, 1)
        # ==========================================
        # TODO 1: 计算 cache entry 的保留优先级
        # 提示: 复用次数越多越该保留，越新越该保留，越大越需要惩罚
        # ==========================================
        # score = ???
        return score

    def _refresh_queue(self, prefix: str):
        entry = self.entries[prefix]
        entry.priority = self._score(entry.hits, entry.bytes, entry.last_used)
        # ==========================================
        # TODO 2: 把最新优先级写入堆队列
        # 提示: 这里要弹出低 priority 的缓存，因此不要对 priority 取负
        # ==========================================
        # queue_item = ???
        heapq.heappush(self.queue, queue_item)

    def _evict_until_fit(self, needed: int):
        while self.current_bytes + needed > self.capacity_bytes and self.entries:
            while self.queue:
                priority, last_used, prefix = heapq.heappop(self.queue)
                entry = self.entries.get(prefix)
                if entry is None:
                    continue
                # ==========================================
                # TODO 3: 跳过堆中的过期记录
                # 提示: entry 的 priority 或 last_used 已变化时，旧堆项不再有效
                # ==========================================
                # is_stale = ???
                if is_stale:
                    continue
                break
            else:
                entry = min(self.entries.values(), key=lambda e: (e.priority, e.last_used))
                prefix = entry.prefix

            self.current_bytes -= entry.bytes
            self.entries.pop(prefix, None)
            self.log.append(f"evict:{prefix}")

    def touch(self, prefix: str, bytes_: int):
        if bytes_ > self.capacity_bytes:
            raise ValueError("single cache entry exceeds capacity")

        self.time += 1
        if prefix in self.entries:
            entry = self.entries[prefix]
            entry.hits += 1
            entry.last_used = self.time
            self._refresh_queue(prefix)
            self.log.append(f"reuse:{prefix}")
            return

        self._evict_until_fit(bytes_)
        # ==========================================
        # TODO 4: 创建新的 cache entry
        # 提示: 新 entry 的 hits 从 1 开始，last_used 使用当前 time
        # ==========================================
        # entry = ???
        self.entries[prefix] = entry
        self.current_bytes += bytes_
        self._refresh_queue(prefix)
        self.log.append(f"add:{prefix}")

    def schedule(self, requests: List[Tuple[str, int]]) -> List[str]:
        for prefix, bytes_ in requests:
            self.touch(prefix, bytes_)
        return list(self.log)

    def snapshot(self) -> List[Tuple[str, int, float, int]]:
        # ==========================================
        # TODO 5: 按优先级导出当前 cache 状态
        # 提示: 高 priority 在前；priority 相同则按 last_used 和 prefix 稳定排序
        # ==========================================
        # ordered_entries = ???
        return [(e.prefix, e.bytes, round(e.priority, 4), e.hits) for e in ordered_entries]

```


```python
# 测试你的实现
def test_kv_cache_scheduler():
    try:
        sim = KVCacheSchedulerSim(capacity_bytes=128)
        requests = [
            ('a', 40),
            ('b', 48),
            ('a', 40),
            ('c', 56),
            ('d', 48),
            ('a', 40),
        ]
        log = sim.schedule(requests)
        snap = sim.snapshot()

        assert len(log) >= len(requests)
        assert any(item.startswith('reuse:a') for item in log)
        assert any(item.startswith('evict:') for item in log)
        assert sim.current_bytes <= sim.capacity_bytes
        assert isinstance(snap, list)
        assert all(len(item) == 4 for item in snap)
        priorities = [item[2] for item in snap]
        assert priorities == sorted(priorities, reverse=True)

        print('✅ KVCacheSchedulerSim 测试通过')
    except NotImplementedError as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_kv_cache_scheduler()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
# TODO：下面是题目区的参考实现。

@dataclass(order=True)
class CacheEntry:
    priority: float
    last_used: int
    prefix: str = field(compare=False)
    hits: int = field(default=0, compare=False)
    bytes: int = field(default=0, compare=False)


class KVCacheSchedulerSim:
    """极简版 KV Cache 调度器。"""

    def __init__(self, capacity_bytes: int = 1024):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self.time = 0
        self.entries: Dict[str, CacheEntry] = {}
        self.queue: List[Tuple[float, int, str]] = []
        self.log: List[str] = []

    def _score(self, hits: int, size: int, last_used: int) -> float:
        recency = 1.0 / (1.0 + max(self.time - last_used, 0))
        reuse_bonus = float(hits)
        size_penalty = size / max(self.capacity_bytes, 1)
        # ==========================================
        # TODO 1: 计算 cache entry 的保留优先级
        # 提示: 复用次数越多越该保留，越新越该保留，越大越需要惩罚
        # ==========================================
        score = reuse_bonus + 0.5 * recency - 0.25 * size_penalty
        return score

    def _refresh_queue(self, prefix: str):
        entry = self.entries[prefix]
        entry.priority = self._score(entry.hits, entry.bytes, entry.last_used)
        # ==========================================
        # TODO 2: 把最新优先级写入堆队列
        # 提示: 这里要弹出低 priority 的缓存，因此不要对 priority 取负
        # ==========================================
        queue_item = (entry.priority, entry.last_used, prefix)
        heapq.heappush(self.queue, queue_item)

    def _evict_until_fit(self, needed: int):
        while self.current_bytes + needed > self.capacity_bytes and self.entries:
            while self.queue:
                priority, last_used, prefix = heapq.heappop(self.queue)
                entry = self.entries.get(prefix)
                if entry is None:
                    continue
                # ==========================================
                # TODO 3: 跳过堆中的过期记录
                # 提示: entry 的 priority 或 last_used 已变化时，旧堆项不再有效
                # ==========================================
                is_stale = (priority, last_used) != (entry.priority, entry.last_used)
                if is_stale:
                    continue
                break
            else:
                entry = min(self.entries.values(), key=lambda e: (e.priority, e.last_used))
                prefix = entry.prefix

            self.current_bytes -= entry.bytes
            self.entries.pop(prefix, None)
            self.log.append(f"evict:{prefix}")

    def touch(self, prefix: str, bytes_: int):
        if bytes_ > self.capacity_bytes:
            raise ValueError("single cache entry exceeds capacity")

        self.time += 1
        if prefix in self.entries:
            entry = self.entries[prefix]
            entry.hits += 1
            entry.last_used = self.time
            self._refresh_queue(prefix)
            self.log.append(f"reuse:{prefix}")
            return

        self._evict_until_fit(bytes_)
        # ==========================================
        # TODO 4: 创建新的 cache entry
        # 提示: 新 entry 的 hits 从 1 开始，last_used 使用当前 time
        # ==========================================
        entry = CacheEntry(priority=0.0, last_used=self.time, prefix=prefix, hits=1, bytes=bytes_)
        self.entries[prefix] = entry
        self.current_bytes += bytes_
        self._refresh_queue(prefix)
        self.log.append(f"add:{prefix}")

    def schedule(self, requests: List[Tuple[str, int]]) -> List[str]:
        for prefix, bytes_ in requests:
            self.touch(prefix, bytes_)
        return list(self.log)

    def snapshot(self) -> List[Tuple[str, int, float, int]]:
        # ==========================================
        # TODO 5: 按优先级导出当前 cache 状态
        # 提示: 高 priority 在前；priority 相同则按 last_used 和 prefix 稳定排序
        # ==========================================
        ordered_entries = sorted(self.entries.values(), key=lambda e: (-e.priority, e.last_used, e.prefix))
        return [(e.prefix, e.bytes, round(e.priority, 4), e.hits) for e in ordered_entries]

```

### 解析

**1. TODO 1: 计算缓存保留优先级**
- **实现方式**：`score = reuse_bonus + 0.5 * recency - 0.25 * size_penalty`
- **关键点**：复用次数越多、最近访问越近，优先级越高；缓存越大，保留成本越高
- **技术细节**：`recency = 1 / (1 + time_gap)` 会随时间间隔衰减，避免长期未访问的缓存一直占据热路径

**2. TODO 2: 刷新堆队列**
- **实现方式**：`queue_item = (entry.priority, entry.last_used, prefix)`，再调用 `heapq.heappush`
- **关键点**：这里的堆用于驱逐，因此低 priority 应该更早被弹出，不需要对 priority 取负
- **技术细节**：同一个 prefix 可能多次刷新优先级，堆里会残留旧记录，后续需要用 stale check 跳过

**3. TODO 3: 跳过过期堆项**
- **实现方式**：`is_stale = (priority, last_used) != (entry.priority, entry.last_used)`
- **关键点**：堆中的记录不一定代表当前最新状态，必须和 `entries` 里的 entry 再核对一次
- **技术细节**：这是懒删除策略：刷新时只压入新记录，不立即删除旧记录；弹出时再判断是否过期

**4. TODO 4: 创建新的缓存条目**
- **实现方式**：`entry = CacheEntry(priority=0.0, last_used=self.time, prefix=prefix, hits=1, bytes=bytes_)`
- **关键点**：新缓存第一次写入时，命中次数从 1 开始，最近访问时间就是当前 `time`
- **技术细节**：新 entry 加入 `entries` 后再调用 `_refresh_queue`，由统一逻辑计算 priority 并写入堆

**5. TODO 5: 导出缓存状态**
- **实现方式**：`ordered_entries = sorted(self.entries.values(), key=lambda e: (-e.priority, e.last_used, e.prefix))`
- **关键点**：快照按高 priority 在前排序，便于观察当前哪些缓存最应该保留
- **技术细节**：`round(e.priority, 4)` 只影响展示，不影响内部调度精度

**KV Cache Scheduling 核心机制**
- **复用价值**：命中次数高的前缀代表未来继续复用的概率更高，通常应该提高保留优先级
- **容量压力**：KV Cache 会随上下文长度和并发请求增长，容量不足时必须选择低价值缓存驱逐
- **懒删除堆**：优先级队列允许重复记录，通过 stale check 保证真正驱逐的是最新的低优先级 entry

**工程优化要点**
- **驱逐策略**：真实系统通常会混合 LRU、LFU、prefix sharing、租户优先级和请求 deadline
- **显存安全**：单个 cache block 不能超过容量，否则应直接拒绝或走降级路径
- **调度联动**：KV Cache 调度通常要和 decode scheduling、prefix caching、PagedAttention block 管理一起设计
