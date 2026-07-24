# 13. End to End Fine Tuning Experiment | 端到端微调实验

**难度：** Medium | **环境：** CPU-first | **标签：** `训练闭环`, `SFT`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面的小节已经分别讲过模型封装、优化器、损失函数和梯度累积，但真实微调不是把这些概念单独跑通就结束。只要数据构造、label 对齐、loss 计算或参数更新里有一个环节接错，训练就会表现成 loss 不降、shape 对不上，或者看似运行但模型没有真正学习。

本节把这些训练要素收成一个最小端到端 SFT 实验：先构造 prompt / response 样本，再计算自回归 loss，最后走完 backward、梯度累积和 optimizer step。完成后，你应该能用一个小模型验证完整微调闭环是否跑通，也能为后面的 LoRA 项目、RLHF/PPO 和训练性能分析建立统一基线。

**关键词：** `end-to-end`, `fine-tuning`, `loop`

---
## 前置阅读

**导语：** 先把模型封装、LoRA 适配、优化器、梯度累积和最小训练接口看过，再做端到端微调实验最顺。
- [P0: 09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- [10. LoRA Tutorial | LoRA 教程](../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)
- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 12. PyTorch Minimal Training Interface | PyTorch 最小训练接口](../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)

## 相关阅读

**导语：** 完成最小 SFT 闭环后，可以继续看对齐训练的显存压力、LoRA 项目化落地和训练性能分析。
- [14. RLHF PPO Memory | RLHF 与 PPO 显存占用与流转](../02_PyTorch_Algorithms/14_RLHF_PPO_Memory.md)
- [30. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/30_LoRA_Fine_Tuning_Project.md)
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)

---
### Step 1: 端到端训练闭环长什么样
端到端微调实验的核心，是把数据、模型、loss 和优化器四层接成一个可运行闭环。

一个完整的微调实验通常包含四层：
1. **数据层**：将 prompt/response 构造为 tokenized batch（input_ids + labels），并进行 padding 对齐，作为模型的直接输入。
2. **模型层**：输入 token 经过 embedding → Transformer → LM head，输出每个位置的 logits。
3. **优化层**：计算 SFT loss，执行 backward、step 和 zero_grad。
4. **训练控制层**：控制梯度累积、参数更新频率和 loss 记录。

这一页承担 `00-12` 的阶段性项目收口：用一个极小语言模型，把前面的训练组件串成完整闭环。后面的 `TODO 1-3` 会分别把这四层拆开实现，再重新合回一个训练闭环。

### Step 2: 为什么要把它做成实验
先说明为什么要把单点函数串成完整实验，再进入代码。

如果只会单点函数，很容易在真实项目里出现“会公式，不会落地”的问题。端到端实验的价值在于——从确认接口正确，到观察训练收敛，再到快速定位问题，最后用极端测试验证闭环：
- 你能确认数据、模型、loss、优化器之间的接口是对的。（例如：模型的输出 shape 是否匹配 loss 函数的输入 shape？优化器的参数是否真的被更新了？）。
- 你能观察训练 loss 是否真的下降。
- 你能快速定位是数据问题、loss 问题，还是优化器问题。
- 你能通过"重复样本过拟合测试"快速验证闭环是否跑通：如果模型在重复样本上 loss 能显著下降，说明数据、loss、优化器链路完整；如果 loss 不降，说明链路中有环节断裂。

### Step 3: 代码实现框架

本实验的模型层（TinyCausalLM）已直接给出，无需修改。它是一个极小的自回归模型（embedding → GRU → LM head），参数规模极小，便于快速验证闭环。
三个 TODO 与训练闭环的对应关系如下：
- TODO 1 → 数据构造（build_sft_batch）
- TODO 2 → 损失计算（compute_sft_loss）
- TODO 3 → 训练更新（run_finetuning_experiment：backward、梯度累积、optimizer step）
- 模型层 → TinyCausalLM 已给出，用于验证闭环，不作为 TODO

下面会实现三块代码：
- `build_sft_batch`：将原始 prompt/response 转为 tokenized batch（返回 input_ids、labels），作为模型的直接输入。
- `compute_sft_loss`：完成 next-token 对齐并计算 SFT loss。
- `run_finetuning_experiment`：驱动训练循环，包含 loss 计算、反向传播、梯度累积和参数更新，返回每步 loss 记录用于观察收敛。
- `TinyCausalLM`：已给出的极小自回归模型，用于验证闭环，不追求下游任务效果。


```python
import torch
import torch.nn as nn

```


```python
def build_sft_batch(prompt_ids, response_ids, pad_id=0, max_len=10):
    # ==========================================
    # 先拼接 prompt 和 response，再决定哪些位置参与监督。
    # TODO 1: 构造 SFT 样本
    # 提示: prompt 部分的 labels mask 为 -100，response 部分保留
    # 注意: 超长时直接从开头截断（本实验数据较短，暂不处理复杂截断）
    # ==========================================
    # input_ids = ???
    # labels = ???

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    else:
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        hidden, _ = self.rnn(x)
        logits = self.lm_head(hidden)
        return logits


def compute_sft_loss(logits, labels):
    # ==========================================
    # TODO 2: 对齐 next-token 预测并计算 SFT loss
    # 提示: logits 取前 t-1 个位置，labels 取后 t-1 个位置
    #       (position t 的 logits 预测 position t+1 的 token)
    #       使用 CrossEntropyLoss(ignore_index=-100)
    # ==========================================
    # shift_logits = ???
    # shift_labels = ???
    # loss = ???
    return loss


def run_finetuning_experiment(model, optimizer, input_ids, labels, accum_steps=2, num_updates=40):
    """
    在同一批样本上反复训练，观察端到端训练闭环是否跑通。
    """
    if input_ids.size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    # ==========================================
    # 先按 micro-batch 跑 forward/backward，最后统一 step。
    # TODO 3: 端到端训练闭环
    # 提示: 切 micro-batch -> 缩放 loss 并 backward -> 最后 step / zero_grad / 返回 history
    # ==========================================
    history = []
    micro_size = input_ids.size(0) // accum_steps

    for _ in range(num_updates):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for idx in range(accum_steps):
            # mb_input = ???
            # mb_labels = ???
            # logits = ???
            # loss = ???
            # total_loss = ???
            pass

        optimizer.step()
        history.append(total_loss)

    return history
```


```python
# 运行此单元格以测试你的实现
def test_end_to_end_finetuning():
    try:
        torch.manual_seed(7)

        prompt = [1, 2, 3]
        response = [4, 5, 6, 7]
        single_input, single_labels = build_sft_batch(prompt, response, pad_id=0, max_len=8)

        # 构造一个 batch，重复同一条样本，便于快速过拟合并验证训练闭环
        input_ids = single_input.unsqueeze(0).repeat(4, 1)
        labels = single_labels.unsqueeze(0).repeat(4, 1)

        model = TinyCausalLM(vocab_size=64, hidden_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

        with torch.no_grad():
            init_loss = compute_sft_loss(model(input_ids), labels).item()

        history = run_finetuning_experiment(model, optimizer, input_ids, labels, accum_steps=2, num_updates=30)

        final_loss = compute_sft_loss(model(input_ids), labels).item()
        print(f"Initial loss: {init_loss:.4f}")
        print(f"Final loss  : {final_loss:.4f}")

        assert len(history) == 30, "训练步数不对"
        assert final_loss < init_loss, "训练没有让 loss 下降，闭环可能有问题"

        print("✅ 测试通过！端到端微调闭环运行正常，loss 已下降。")
    except NotImplementedError:
        print("请先完成 TODO 部分。")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义" if isinstance(e, NameError) else "代码可能未完成，导致了类型错误")
        raise NotImplementedError("请先完成 TODO 部分。") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分。") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_end_to_end_finetuning()
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
import torch
import torch.nn as nn

def build_sft_batch(prompt_ids, response_ids, pad_id=0, max_len=10):
    # 先拼接 prompt 和 response，再决定哪些位置参与监督。
    # TODO 1: 构造 SFT 样本
    # 提示: prompt 部分的 labels mask 为 -100，response 部分保留
    # 注意: 超长时直接从开头截断（本实验数据较短，暂不处理复杂截断）
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    else:
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        hidden, _ = self.rnn(x)
        logits = self.lm_head(hidden)
        return logits


def compute_sft_loss(logits, labels):
    # 先把时间步错一位，再做 token 级分类损失。
    # TODO 2: 对齐 next-token 预测并计算 SFT loss
    # 提示: logits 取前 t-1 个位置，labels 取后 t-1 个位置
    #       (position t 的 logits 预测 position t+1 的 token)
    #       使用 CrossEntropyLoss(ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def run_finetuning_experiment(model, optimizer, input_ids, labels, accum_steps=2, num_updates=40):
    # 先按 micro-batch 跑 forward/backward，最后统一 step。
    # TODO 3: 端到端训练闭环
    # 提示: 切 micro-batch -> 缩放 loss 并 backward -> 最后 step / zero_grad / 返回 history
    if input_ids.size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    history = []
    micro_size = input_ids.size(0) // accum_steps

    for _ in range(num_updates):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for idx in range(accum_steps):
            mb_input = input_ids[idx * micro_size:(idx + 1) * micro_size]
            mb_labels = labels[idx * micro_size:(idx + 1) * micro_size]
            logits = model(mb_input)
            loss = compute_sft_loss(logits, mb_labels) / accum_steps
            loss.backward()
            total_loss += loss.detach().item()

        optimizer.step()
        history.append(total_loss)

    return history
```

### 答案与直觉

- **这一题要解决什么**：把 SFT 数据构造、loss 计算和训练更新串成一个最小闭环。
- **为什么这样做**：只有把输入、监督信号和优化器路径全部对齐，实验结果才有可解释性。
- **带走的直觉**：端到端实验的重点不是堆功能，而是确认整条训练链路能稳定跑通。

**1. TODO 1 (构造 SFT 样本)**

- **拼接输入：** `input_ids` 由 `prompt + response` 拼接得到，保持样本的完整上下文。
- **监督标签：** `labels` 里，`prompt` 对应的位置要 mask 成 `-100`，只让模型学习回答部分。
- **长度处理：** 超过 `max_len` 时要裁剪，不足时要补 `pad_id` 和 `-100`。
- **训练目标：** SFT 关注的是“模型对回答部分的预测能力”，而不是复述提示词本身。

**2. TODO 2 (对齐 next-token 并计算 SFT loss)**

- **一位错位：** `shift_logits = logits[..., :-1, :]`，`shift_labels = labels[..., 1:]`。
- **损失函数：** 使用 `CrossEntropyLoss(ignore_index=-100)` 计算 loss，让 prompt 和 padding 位置自然忽略。
- **监督范围：** 训练信号只来自 response 的有效 token，next-token 对齐要和 causal LM 的训练目标一致。
- **形状检查：** 这一层本质上是在确认 logits 和 labels 的时间步是否对齐。

**3. TODO 3 (训练闭环)**

- **micro-batch：** 先把 batch 切成多个 `micro-batch`，再逐个累积梯度。
- **loss 缩放：** 每个 `micro-batch` 的 loss 要除以 `accum_steps`，保证和完整 batch 的梯度一致。
- **参数更新：** 所有 `micro-batch` 处理完之后，再统一执行 `optimizer.step()` 和 `optimizer.zero_grad()`。
- **训练记录：** 最后返回 `history`，方便观察训练过程中 loss 是否下降。

**进阶思考：为什么要做重复样本验证？**

- **一致性检查：** 通过重复样本验证，可以确认梯度累积是否真的等价于完整 batch。
- **闭环意义：** 这条链路把 `SFT Loss`、`梯度累积`、`参数更新` 连接成一个可运行的小闭环。
- **工程价值：** 只要这套链路对齐，后续再切换更复杂的数据和更大的 batch 也更稳。