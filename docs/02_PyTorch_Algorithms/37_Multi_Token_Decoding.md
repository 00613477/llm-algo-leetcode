# 37. Multi Token Decoding | 多 Token 解码
**难度：** Hard | **环境：** GPU required | **标签：** `解码`, `Multi-Token Decoding`, `推理优化` | **目标人群：** 推理系统与系统工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/37_Multi_Token_Decoding.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

自回归生成最朴素的路径是“一次生成一个 token”：模型前向一次、更新一次 KV Cache、再进入下一轮解码。这个流程清晰但开销很碎，尤其在长输出场景中，很多时间会消耗在反复进入 decoder、频繁调度 kernel 和维护 cache 状态上。

本节关注一个更激进的方向：能不能在一次解码步里先提出多个候选 token，再用验证规则决定哪些可以接受、从哪里需要回退。你会实现一个极简版 `MultiTokenDecoderSim`，把“提议、验证、接受、回退”这条链路拆开，理解多 token 解码与投机解码之间的关系。

**关键词：** `multi-token decoding`, `draft model`, `verification`, `rollback`

---

## 前置阅读

**导语：** 先看投机解码、解码策略和 PagedAttention，再看多 token 解码会更清楚。

- [23. Speculative Decoding | 投机解码](../02_PyTorch_Algorithms/23_Speculative_Decoding.md)
- [21. Decoding Strategies | 解码策略](../02_PyTorch_Algorithms/21_Decoding_Strategies.md)
- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

## 相关阅读

**导语：** 多 token 解码之后，可以继续看前缀缓存和 RadixAttention。

- [36. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](../02_PyTorch_Algorithms/36_Prefix_Caching_and_Chunked_Prefill.md)
- [24. SGLang RadixAttention | SGLang 基数注意力](../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- [P1: 17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md)

---
## Step 1: 原理与痛点

单 token 解码的瓶颈在于每次只能推进一个 token：模型需要反复进入 decoder，KV Cache 也要频繁追加和读取。对短输出来说这不是大问题，但对长文本生成、代码生成、多轮对话这类场景，逐 token 调度会放大延迟和系统开销。

Multi-Token Decoding 的目标不是“无条件一次吐出更多 token”，而是在可验证的前提下，让一次解码步尽量推进多个 token。如果连续几个候选 token 都能被接受，就减少了多轮单 token 解码；如果中途被拒绝，则立刻停止并回退到更保守的路径。

它和 Speculative Decoding 的关系可以这样理解：Speculative Decoding 强调“草稿模型先生成，再由目标模型验证”；Multi-Token Decoding 更强调“一个解码步里尝试推进多个 token”。两者都在减少 token-level 往返，核心难点也都落在接受率、验证成本和回退策略上。

## Step 2: 代码实现框架

下面的代码会模拟一条最小的多 token 解码链路。为了让重点清晰，我们不实现真实采样器，也不接入大模型，只用给定的 `draft_tokens`、`draft_probs` 和 `target_probs` 来模拟“草稿提议”和“目标验证”。

这条链路拆成四个动作：

| 动作 | 对应方法 | 作用 |
|------|----------|------|
| 提议 | `propose` | 从草稿 token 中截取本轮最多可尝试的候选序列 |
| 接受判断 | `_accept_token` | 比较目标概率和草稿概率，决定单个 token 是否可靠 |
| 逐 token 验证 | `verify` | 从左到右验证候选 token，遇到首次拒绝就停止 |
| 解码汇总 | `decode` | 返回提议、接受、拒绝位置和需要回退的后缀 |

这个实现保留了多 token 解码最关键的控制流：先批量提议，再顺序验证，最后根据首次拒绝位置切分“可接受前缀”和“回退后缀”。

## Step 3: 核心机制

多 token 解码的收益来自“连续接受”。如果本轮提出 4 个候选 token，并且前 3 个都通过验证，那么系统相当于用一次解码流程推进了 3 个 token；如果第 1 个就被拒绝，则这轮几乎没有收益，还需要回到保守生成路径。

因此，提议长度和接受阈值需要平衡：

| 参数 | 过大时的问题 | 过小时的问题 |
|------|--------------|--------------|
| `max_proposal_len` | 候选更激进，拒绝和回退概率更高 | 单轮推进短，加速效果有限 |
| `min_accept_ratio` | 接受规则更严格，通过率下降 | 接受规则更宽松，可能引入更大偏差 |

本节用一个简化规则来模拟接受判断：若目标模型给候选 token 的概率不低于草稿概率的一定比例，就认为它可以被接受。真实系统会使用更严格的分布校正或采样验证规则，但“连续接受、首次拒绝、后缀回退”的主线是一致的。

## Step 4: 动手实战

**要求**：请补全下方 `MultiTokenDecoderSim`，实现一个极简版的多 token 生成与验证模拟器。重点不是复杂采样，而是把“提议 -> 验证 -> 接受 / 回退”这条链路跑通。


```python
from typing import List, Sequence, Tuple
import torch
```


```python
class MultiTokenDecoderSim:
    """极简版多 token 生成与验证模拟器。"""

    def __init__(self, max_proposal_len: int = 4, min_accept_ratio: float = 0.5):
        if max_proposal_len <= 0:
            raise ValueError("max_proposal_len must be positive")
        if not (0.0 < min_accept_ratio <= 1.0):
            raise ValueError("min_accept_ratio must be in (0, 1]")
        self.max_proposal_len = max_proposal_len
        self.min_accept_ratio = min_accept_ratio
        self.history: List[dict] = []

    def propose(self, draft_tokens: Sequence[int]) -> List[int]:
        # ==========================================
        # TODO 1: 从草稿 token 中生成本轮候选序列
        # 提示: 先把 draft_tokens 转成 list，再截取前 max_proposal_len 个 token
        # ==========================================
        # proposed = ???
        return proposed

    def _accept_token(self, draft_prob: float, target_prob: float) -> bool:
        # ==========================================
        # TODO 2: 判断单个候选 token 是否被目标模型接受
        # 提示: 正常情况下，target_prob 至少要达到 draft_prob * min_accept_ratio
        # ==========================================
        if draft_prob <= 0:
            return target_prob > 0
        # accepted = ???
        return accepted

    def verify(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: Sequence[int],
    ) -> Tuple[List[int], int | None]:
        proposed = self.propose(draft_tokens)
        accepted_tokens: List[int] = []
        rejected_at = None
        draft_probs = torch.as_tensor(draft_probs)
        target_probs = torch.as_tensor(target_probs)

        for i, token_id in enumerate(proposed):
            draft_prob = float(draft_probs[i, token_id])
            target_prob = float(target_probs[i, token_id])
            # ==========================================
            # TODO 3: 逐个验证候选 token，遇到第一次拒绝就停止
            # 提示: 调用 _accept_token 得到 accepted；后续接受/拒绝分支已经给出
            # ==========================================
            # accepted = ???

            if accepted:
                accepted_tokens = accepted_tokens + [token_id]
            else:
                rejected_at = i
                break

        return accepted_tokens, rejected_at

    def decode(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: Sequence[int],
    ) -> dict:
        proposed = self.propose(draft_tokens)
        accepted_tokens, rejected_at = self.verify(draft_probs, target_probs, draft_tokens)
        # ==========================================
        # TODO 4: 切出被拒绝后缀，形成完整解码结果
        # 提示: rejected_at 为 None 表示全部接受；否则从 rejected_at 开始都是回退后缀
        # ==========================================
        # rejected_suffix = ???

        result = {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted_tokens,
            "accepted_len": len(accepted_tokens),
            "rejected_at": rejected_at,
            "rejected_suffix": rejected_suffix,
        }
        self.history.append(result)
        return result

```


```python
def test_multi_token_decoder():
    try:
        sim = MultiTokenDecoderSim(max_proposal_len=3, min_accept_ratio=0.6)
        draft_tokens = [10, 20, 30, 31]
        draft_probs = torch.zeros(4, 40)
        target_probs = torch.zeros(4, 40)

        for i, tok in enumerate(draft_tokens):
            draft_probs[i, tok] = 0.5
            target_probs[i, tok] = 0.8 if i < 2 else 0.2

        assert sim.propose(draft_tokens) == [10, 20, 30]
        assert sim._accept_token(0.5, 0.31) is True
        assert sim._accept_token(0.5, 0.2) is False

        accepted, rejected_at = sim.verify(draft_probs, target_probs, draft_tokens)
        assert accepted == [10, 20]
        assert rejected_at == 2

        result = sim.decode(draft_probs, target_probs, draft_tokens)
        assert result["proposed_tokens"] == [10, 20, 30]
        assert result["accepted_tokens"] == [10, 20]
        assert result["accepted_len"] == 2
        assert result["rejected_at"] == 2
        assert result["rejected_suffix"] == [30]
        assert len(sim.history) == 1

        print("✅ MultiTokenDecoderSim 测试通过！")
    except NotImplementedError as e:
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except (AttributeError, NameError, TypeError, ValueError, IndexError) as e:
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        raise AssertionError("测试未通过，请检查提议、验证和回退逻辑。") from e


test_multi_token_decoder()

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

class MultiTokenDecoderSim:
    """极简版多 token 生成与验证模拟器。"""

    def __init__(self, max_proposal_len: int = 4, min_accept_ratio: float = 0.5):
        if max_proposal_len <= 0:
            raise ValueError("max_proposal_len must be positive")
        if not (0.0 < min_accept_ratio <= 1.0):
            raise ValueError("min_accept_ratio must be in (0, 1]")
        self.max_proposal_len = max_proposal_len
        self.min_accept_ratio = min_accept_ratio
        self.history: List[dict] = []

    def propose(self, draft_tokens: Sequence[int]) -> List[int]:
        # ==========================================
        # TODO 1: 从草稿 token 中生成本轮候选序列
        # 提示: 先把 draft_tokens 转成 list，再截取前 max_proposal_len 个 token
        # ==========================================
        proposed = list(draft_tokens)[: self.max_proposal_len]
        return proposed

    def _accept_token(self, draft_prob: float, target_prob: float) -> bool:
        # ==========================================
        # TODO 2: 判断单个候选 token 是否被目标模型接受
        # 提示: 正常情况下，target_prob 至少要达到 draft_prob * min_accept_ratio
        # ==========================================
        if draft_prob <= 0:
            return target_prob > 0
        accepted = target_prob >= draft_prob * self.min_accept_ratio
        return accepted

    def verify(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: Sequence[int],
    ) -> Tuple[List[int], int | None]:
        proposed = self.propose(draft_tokens)
        accepted_tokens: List[int] = []
        rejected_at = None
        draft_probs = torch.as_tensor(draft_probs)
        target_probs = torch.as_tensor(target_probs)

        for i, token_id in enumerate(proposed):
            draft_prob = float(draft_probs[i, token_id])
            target_prob = float(target_probs[i, token_id])
            # ==========================================
            # TODO 3: 逐个验证候选 token，遇到第一次拒绝就停止
            # 提示: 调用 _accept_token 得到 accepted；后续接受/拒绝分支已经给出
            # ==========================================
            accepted = self._accept_token(draft_prob, target_prob)

            if accepted:
                accepted_tokens = accepted_tokens + [token_id]
            else:
                rejected_at = i
                break

        return accepted_tokens, rejected_at

    def decode(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: Sequence[int],
    ) -> dict:
        proposed = self.propose(draft_tokens)
        accepted_tokens, rejected_at = self.verify(draft_probs, target_probs, draft_tokens)
        # ==========================================
        # TODO 4: 切出被拒绝后缀，形成完整解码结果
        # 提示: rejected_at 为 None 表示全部接受；否则从 rejected_at 开始都是回退后缀
        # ==========================================
        rejected_suffix = proposed[rejected_at:] if rejected_at is not None else []

        result = {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted_tokens,
            "accepted_len": len(accepted_tokens),
            "rejected_at": rejected_at,
            "rejected_suffix": rejected_suffix,
        }
        self.history.append(result)
        return result

```

### 解析

**1. TODO 1: 生成本轮候选序列**
- **实现方式**：`proposed = list(draft_tokens)[: self.max_proposal_len]`
- **关键点**：一次只截取最多 `max_proposal_len` 个候选，避免提议过长导致回退成本过高
- **技术细节**：这里不实现真实采样器，而是用已给定的 `draft_tokens` 模拟草稿模型提出的候选序列

**2. TODO 2: 单 token 接受判断**
- **实现方式**：`accepted = target_prob >= draft_prob * self.min_accept_ratio`
- **关键点**：用目标概率相对草稿概率的比例，近似表示目标模型是否认可这个候选 token
- **技术细节**：`draft_prob <= 0` 的边界分支已经提前返回；正常分支只需要比较 `target_prob` 是否达到接受阈值

**3. TODO 3: 接入逐 token 验证**
- **实现方式**：`accepted = self._accept_token(draft_prob, target_prob)`
- **关键点**：验证循环必须从左到右执行，只有当前 token 被接受，后面的候选才仍然处在正确前缀下
- **技术细节**：接受后的追加逻辑和拒绝后的 `rejected_at + break` 已经给出；一旦首次拒绝，后续候选不能继续直接采用

**4. TODO 4: 切分回退后缀**
- **实现方式**：`rejected_suffix = proposed[rejected_at:] if rejected_at is not None else []`
- **关键点**：`accepted_tokens` 是可以直接写入输出序列的前缀，`rejected_suffix` 是需要丢弃或重新生成的部分
- **技术细节**：当 `rejected_at is None` 时表示本轮候选全部接受，因此回退后缀为空列表

**Multi-Token Decoding 核心机制**
- **单 token 解码的瓶颈**：每次只推进一个 token，会带来频繁的 decoder 调用、kernel 调度和 KV Cache 更新
- **多 token 推进**：草稿路径一次提出多个候选，目标路径验证后尽可能接受连续前缀，从而减少 token-level 往返
- **首次拒绝原则**：候选序列具有前缀依赖，一旦某个位置被拒绝，它之后的候选就不再可靠，需要回退

**工程优化要点**
- **接受率权衡**：提议越长，理论加速空间越大，但被拒绝和回退的概率也越高
- **验证成本**：多 token 解码只有在验证成本低于逐 token 生成成本时才有收益
- **系统联动**：真实实现通常还要和 KV Cache 管理、batch 调度、采样策略和投机解码校正规则一起设计
