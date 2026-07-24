# 38. Decode Scheduling | 解码调度
**难度：** Hard | **环境：** GPU required | **标签：** `解码`, `Scheduling`, `推理优化` | **目标人群：** 推理系统与服务调度

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/38_Decode_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面几节已经看过投机解码、前缀缓存和多 token 生成：它们都在减少单个请求的解码开销。但真实推理服务里，请求不是一个一个孤立到来的，而是同时排队、同时竞争 GPU、同时处在不同阶段。调度器要解决的，就是如何让这些请求有序进入 prefill 和 decode，而不是让 GPU 在等待和碎片化切换中浪费时间。

本节用一个极简 `DecodeSchedulerSim` 模拟请求入队、优先级排序、阶段切换和持续调度。学完后，你应该能看清 decode scheduling 的主线：不是简单“谁先来谁先跑”，而是在吞吐、延迟、cache 命中和公平性之间建立一套可解释的选择规则。

**关键词：** `decode scheduling`, `prefill`, `batch reordering`

---

## 前置阅读

**导语：** 先把解码策略、投机解码和前缀缓存理顺，再看调度规则会更容易。

- [21. Decoding Strategies | 解码策略](../02_PyTorch_Algorithms/21_Decoding_Strategies.md)
- [23. Speculative Decoding | 投机解码](../02_PyTorch_Algorithms/23_Speculative_Decoding.md)
- [36. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](../02_PyTorch_Algorithms/36_Prefix_Caching_and_Chunked_Prefill.md)
- [37. Multi-Token Decoding | 多 Token 解码](../02_PyTorch_Algorithms/37_Multi_Token_Decoding.md)

## 相关阅读

**导语：** 解码调度之后，可以继续看 PagedAttention 和 RadixAttention。

- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [24. SGLang RadixAttention | SGLang 基数注意力](../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- [P1: 17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md)

---
### Step 1: 原理与痛点

> **为什么推理服务不能只靠单个请求的解码优化？**
>
> 前面几节已经看过投机解码、前缀缓存和多 token 解码，它们都能降低单个请求的生成成本。但线上服务面对的是一批请求：有的刚进入 prefill，有的正在 decode，有的命中 cache，有的 prompt 很长。如果没有调度策略，GPU 很容易在短 decode、长 prefill 和队列等待之间来回切换。

Decode Scheduling 解决的不是“谁先来谁先跑”，而是如何在吞吐、延迟和公平性之间做选择。调度器至少要回答三个问题：

- **先调谁**：cache hit 请求、短请求、高优先级请求，谁应该先进入 GPU；
- **调什么阶段**：当前更应该推进 prefill，还是继续推进 decode；
- **什么时候结束**：请求生成到目标长度后，如何从 active 队列中退出。

这一步的核心直觉是：请求的优先级不是静态队列顺序，而是由阶段、cache 状态、业务优先级和当前长度共同决定。

### Step 2: 代码实现框架

本节会实现一个最小 `DecodeSchedulerSim`。它不模拟真实模型前向，也不做 batch packing，而是把一个请求的生命周期压缩成两个阶段：先 `prefill`，再多步 `decode`，直到 `generated_len` 达到目标长度。

代码拆成四个动作：

| 动作 | 对应方法 | 作用 |
|------|----------|------|
| 入队 | `enqueue` | 创建请求状态，默认从 prefill 阶段开始 |
| 排序 | `_schedule_key` | 根据阶段、cache 命中、优先级和长度定义调度顺序 |
| 单步调度 | `step` | 选择一个 active 请求并推进一个状态变化 |
| 持续运行 | `run` | 重复执行 step，直到请求完成或达到步数上限 |

这套代码的重点不是实现最优调度器，而是把调度决策显式化。只要 `_schedule_key` 改变，整个调度行为就会改变，这也是推理服务调度器最核心的设计入口。

### Step 3: 核心机制

本节用一个 tuple 作为调度排序键：

$$
key = (phase\_rank, cache\_rank, -priority, total\_len, request\_id)
$$

排序键越小，越先被调度。这里的含义是：先区分阶段，再考虑 cache 命中，再考虑业务优先级，最后用长度和 request id 做稳定排序。

这个规则不是唯一答案，而是一个可解释的教学策略。`phase_rank` 控制 prefill / decode 的切换倾向，`cache_rank` 体现 cache hit 的复用收益，`-priority` 让高优先级请求排在前面，`total_len` 避免长序列长期占住前排。真实系统会再加入 token budget、batch size、KV Cache 容量和等待时间等约束。

### Step 4: 动手实战

**要求**：请补全下方 `DecodeSchedulerSim`，跑通“入队 -> 排序 -> 单步推进 -> 持续调度”这条链路。你需要重点完成四个位置：构造请求状态、定义排序键、选择本轮请求，以及更新已执行步数。

完成后观察测试中的事件序列：`prefill_to_decode` 表示请求完成预填充并进入 decode，`decode_one_step` 表示生成推进一步，`finish` 表示请求完成。只要这些事件能按调度规则串起来，就说明 decode scheduling 的最小闭环已经跑通。


```python
from dataclasses import dataclass
from typing import Dict, List, Literal

```


```python
Phase = Literal['prefill', 'decode']


@dataclass
class RequestState:
    request_id: int
    prompt_len: int
    generated_len: int = 0
    phase: Phase = 'prefill'
    priority: int = 0
    cache_hit: bool = False

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_len

    @property
    def done(self) -> bool:
        return self.generated_len >= self.prompt_len


class DecodeSchedulerSim:
    """极简版 prefill / decode 调度器。"""

    def __init__(self):
        self.queue: List[RequestState] = []
        self.timeline: List[Dict[str, int | str]] = []

    def enqueue(self, request_id: int, prompt_len: int, priority: int = 0, cache_hit: bool = False) -> None:
        # ==========================================
        # TODO 1: 构造请求状态，并加入调度队列
        # 提示: RequestState 默认处于 prefill 阶段，只需要传入请求属性
        # ==========================================
        # request = ???
        self.queue.append(request)

    def _schedule_key(self, req: RequestState):
        # ==========================================
        # TODO 2: 定义调度排序键
        # 提示: prefill 优先于 decode，cache hit 优先于 cache miss，高 priority 更靠前
        # ==========================================
        phase_rank = 0 if req.phase == 'prefill' else 1
        cache_rank = 0 if req.cache_hit else 1
        # key = ???
        return key

    def step(self) -> Dict[str, int | str] | None:
        active = [req for req in self.queue if not req.done]
        if not active:
            return None

        # ==========================================
        # TODO 3: 从 active 请求中选择本轮要调度的请求
        # 提示: 使用 min(..., key=self._schedule_key) 选择排序最靠前的请求
        # ==========================================
        # chosen = ???

        if chosen.phase == 'prefill':
            chosen.phase = 'decode'
            action = 'prefill_to_decode'
        else:
            chosen.generated_len += 1
            action = 'finish' if chosen.done else 'decode_one_step'

        event = {
            'request_id': chosen.request_id,
            'phase': 'prefill' if action == 'prefill_to_decode' else 'decode',
            'action': action,
            'prompt_len': chosen.prompt_len,
            'generated_len': chosen.generated_len,
        }
        self.timeline.append(event)
        return event

    def run(self, max_steps: int = 100) -> List[Dict[str, int | str]]:
        steps = 0
        while steps < max_steps:
            event = self.step()
            if event is None:
                break
            # ==========================================
            # TODO 4: 更新已经执行的调度步数
            # 提示: 每成功执行一次 step，steps 增加 1
            # ==========================================
            # steps = ???
        return self.timeline

```


```python
# 测试你的实现
def test_decode_scheduler():
    try:
        sim = DecodeSchedulerSim()
        sim.enqueue(request_id=1, prompt_len=2, priority=2, cache_hit=True)
        sim.enqueue(request_id=2, prompt_len=3, priority=1, cache_hit=False)
        sim.enqueue(request_id=3, prompt_len=1, priority=3, cache_hit=True)

        assert len(sim.queue) == 3
        assert sim.queue[0].phase == 'prefill'

        events = sim.run(max_steps=20)
        assert len(events) == 9
        assert events[0]['request_id'] == 3
        assert events[0]['action'] == 'prefill_to_decode'
        assert any(e['action'] == 'finish' for e in events)
        assert all(req.done for req in sim.queue)
        assert sim.timeline is events

        print('✅ DecodeSchedulerSim 测试通过')
    except NotImplementedError as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_decode_scheduler()

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

Phase = Literal['prefill', 'decode']


@dataclass
class RequestState:
    request_id: int
    prompt_len: int
    generated_len: int = 0
    phase: Phase = 'prefill'
    priority: int = 0
    cache_hit: bool = False

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_len

    @property
    def done(self) -> bool:
        return self.generated_len >= self.prompt_len


class DecodeSchedulerSim:
    """极简版 prefill / decode 调度器。"""

    def __init__(self):
        self.queue: List[RequestState] = []
        self.timeline: List[Dict[str, int | str]] = []

    def enqueue(self, request_id: int, prompt_len: int, priority: int = 0, cache_hit: bool = False) -> None:
        # ==========================================
        # TODO 1: 构造请求状态，并加入调度队列
        # 提示: RequestState 默认处于 prefill 阶段，只需要传入请求属性
        # ==========================================
        request = RequestState(
            request_id=request_id,
            prompt_len=prompt_len,
            priority=priority,
            cache_hit=cache_hit,
        )
        self.queue.append(request)

    def _schedule_key(self, req: RequestState):
        # ==========================================
        # TODO 2: 定义调度排序键
        # 提示: prefill 优先于 decode，cache hit 优先于 cache miss，高 priority 更靠前
        # ==========================================
        phase_rank = 0 if req.phase == 'prefill' else 1
        cache_rank = 0 if req.cache_hit else 1
        key = (phase_rank, cache_rank, -req.priority, req.total_len, req.request_id)
        return key

    def step(self) -> Dict[str, int | str] | None:
        active = [req for req in self.queue if not req.done]
        if not active:
            return None

        # ==========================================
        # TODO 3: 从 active 请求中选择本轮要调度的请求
        # 提示: 使用 min(..., key=self._schedule_key) 选择排序最靠前的请求
        # ==========================================
        chosen = min(active, key=self._schedule_key)
        if chosen.phase == 'prefill':
            chosen.phase = 'decode'
            action = 'prefill_to_decode'
        else:
            chosen.generated_len += 1
            action = 'finish' if chosen.done else 'decode_one_step'

        event = {
            'request_id': chosen.request_id,
            'phase': 'prefill' if action == 'prefill_to_decode' else 'decode',
            'action': action,
            'prompt_len': chosen.prompt_len,
            'generated_len': chosen.generated_len,
        }
        self.timeline.append(event)
        return event

    def run(self, max_steps: int = 100) -> List[Dict[str, int | str]]:
        steps = 0
        while steps < max_steps:
            event = self.step()
            if event is None:
                break
            # ==========================================
            # TODO 4: 更新已经执行的调度步数
            # 提示: 每成功执行一次 step，steps 增加 1
            # ==========================================
            steps = steps + 1
        return self.timeline

```

### 解析

**1. TODO 1: 构造请求状态**
- **实现方式**：`request = RequestState(request_id=request_id, prompt_len=prompt_len, priority=priority, cache_hit=cache_hit)`
- **关键点**：请求入队时默认处于 `prefill` 阶段，`generated_len` 也从 0 开始
- **技术细节**：`RequestState` 把请求的阶段、优先级、cache 命中和长度状态收在一起，后续调度只需要读取这个状态对象

**2. TODO 2: 定义调度排序键**
- **实现方式**：`key = (phase_rank, cache_rank, -req.priority, req.total_len, req.request_id)`
- **关键点**：排序键越小越优先，因此 `prefill` 用 0、`decode` 用 1，cache hit 用 0、cache miss 用 1
- **技术细节**：`-req.priority` 用来让更高优先级排在前面；`total_len` 和 `request_id` 用作稳定的 tie-breaker

**3. TODO 3: 选择本轮调度请求**
- **实现方式**：`chosen = min(active, key=self._schedule_key)`
- **关键点**：调度器不是按入队顺序直接执行，而是每一步都根据当前状态重新选择最适合推进的请求
- **技术细节**：`active` 已经过滤掉完成请求，`min(..., key=...)` 会把排序规则集中交给 `_schedule_key`，避免调度逻辑散落在多个地方

**4. TODO 4: 更新调度步数**
- **实现方式**：`steps = steps + 1`
- **关键点**：只有 `step()` 真正返回事件时才增加步数；如果返回 `None`，说明没有可调度请求，应立即退出
- **技术细节**：`max_steps` 是防御性上限，避免调度规则写错后进入无限循环

**Decode Scheduling 核心机制**
- **阶段差异**：`prefill` 通常计算量大、适合批处理；`decode` 每步短但频繁，直接影响 token latency
- **排序规则**：调度器需要同时考虑请求阶段、cache 命中、业务优先级和序列长度，而不是简单 FIFO
- **状态推进**：一次 prefill 会把请求切到 decode；decode 每执行一步就增加 `generated_len`，直到请求完成

**工程优化要点**
- **吞吐与延迟权衡**：过度追求大 batch 会增加等待时间，过度追求低延迟又会降低 GPU 利用率
- **Cache-aware 调度**：cache hit 请求通常更便宜，优先调度可以提升复用收益，但不能让长请求长期饥饿
- **真实系统扩展**：生产调度器还会加入 token budget、KV Cache 容量、请求超时、抢占和多队列策略
