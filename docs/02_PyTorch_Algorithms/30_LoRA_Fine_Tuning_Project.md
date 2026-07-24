# 30. LoRA Fine Tuning Project | LoRA 微调项目

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `LoRA`, `Finetuning` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/30_LoRA_Fine_Tuning_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面已经分别讲过 LoRA 机制、训练闭环和显存账本，但真实项目里不能只回答“LoRA 能不能跑”。更关键的问题是：相同任务下，LoRA 相比 baseline 到底少训练了多少参数，省了多少显存，速度和 loss 又付出了什么代价。

本节把 LoRA 微调做成一个项目账本：先比较 LoRA adapter 和全参数线性层的参数量，再把参数、显存、速度和 loss 收成 baseline vs LoRA 的项目结论。代码区只实现最小可复用的参数账本和结果汇总，完整训练循环、loss 曲线和 profiling 截图可以基于这份账本继续补充。

**关键词：** `LoRA`, `training`, `project`, `profiling`

---



## 前置阅读

**导语：** 先看 LoRA 机制、端到端训练闭环和显存优化，再做这个项目；本节默认你已经知道训练循环怎么跑，重点转向 LoRA 方案是否值得采用。
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [11. LR Schedulers WSD Cosine | WSD 余弦学习率调度器](./11_LR_Schedulers_WSD_Cosine.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
- [P0: 20. Profiling and Memory Ledger | 性能剖析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)

## 相关阅读

**导语：** 完成 LoRA 项目账本后，建议继续用训练性能分析、推理性能对比和 profiling 方法验证这套方案的实际成本。
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)
- [31. Inference Performance Comparison | 推理性能对比实验](./31_Inference_Performance_Comparison.md)
- [32. Training Performance Analysis | 训练性能分析](./32_Training_Performance_Analysis.md)


### Step 1: 定义 LoRA 微调目标
先回答一个问题：在尽量少训练参数的前提下，LoRA 能否完成目标任务，并保留可接受的 loss / accuracy 表现？

- 固定底座模型、数据集、batch size、seq len、优化器、学习率和训练 step 数。
- 明确 baseline 是全参数微调、冻结底座不训练，还是已有的普通微调配置。
- 统一记录核心指标：可训练参数量、参数占比、step time、peak memory、loss / accuracy。
- 这节先建立 LoRA 参数账本，再把参数、显存、速度和效果收成一份项目汇总。

### Step 2: 跑通 baseline 并记录账本

LoRA 的收益必须和稳定 baseline 对比，不能只看 LoRA 自己能不能跑。

- 先在同一批样本和同一套训练配置下跑通 baseline。
- 记录 baseline 的可训练参数量、loss 曲线、平均 step time 和 peak memory。
- 确认 baseline loss 能正常下降，再进入 LoRA 对比。
- 如果 baseline 本身不稳定，后面的 LoRA 结果就没有可解释性。

### Step 3: 插入 LoRA 并做同口径对比

把 LoRA adapter 插到 attention projection 或 MLP linear layer 上，只训练低秩旁路。

- 冻结底座权重，只让 LoRA 的 `A / B` 矩阵参与训练。
- 先计算单层 LoRA 参数量，再估算多层插入后的总可训练参数量。
- 用同样的 batch、输入长度和训练步数比较 LoRA 与 baseline。
- 重点看三个问题：参数量省了多少，显存 / 速度是否改善，loss 是否仍然正常下降。

### Step 4: 输出微调项目结论

最后把 LoRA 和 baseline 放到同一张表里，说明这次微调方案是否值得采用。

- 输出 baseline vs LoRA 对比表，至少包含 trainable params、param ratio、step time、peak memory、loss / accuracy。
- 写清楚 LoRA 节省的是训练参数和优化器状态，不等于底座模型权重不存在。
- 记录本次 rank、插层位置、学习率和训练步数，方便后续复现实验。
- 如果效果不足，下一轮优先调整 rank、插层范围、学习率或 gradient accumulation。
- 最终产物应回答：LoRA 少训练了多少参数，换来了多少显存 / 速度收益，效果是否还能接受。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整 LoRA 微调项目流程。下面的代码只实现其中最小、可复用的两块：计算 LoRA 参数账本，并汇总 baseline / LoRA 的核心实验结果。真实项目中的训练循环、loss 曲线和 profiling 截图，可以基于这份账本继续补充。


```python
import math

```


```python
# TODO: 完成 LoRA 参数账本和项目汇总
# 目标：从参数量估算到 baseline vs LoRA 项目报告

def lora_trainable_params(in_dim, out_dim, rank):
    # ==========================================
    # TODO 1: 计算单层 LoRA 的可训练参数量
    # 提示：LoRA 旁路包含 A 和 B 两个低秩矩阵。
    # ==========================================
    # trainable_params = ???
    return trainable_params

def full_linear_params(in_dim, out_dim):
    # ==========================================
    # TODO 2: 计算完整线性层的参数量
    # 提示：这里只统计 weight，不额外考虑 bias。
    # ==========================================
    # total_params = ???
    return total_params

def lora_param_ratio(in_dim, out_dim, rank):
    # ==========================================
    # TODO 3: 计算 LoRA 参数占比
    # 提示：先算 LoRA 参数量和全参基线，再做比例。
    # ==========================================
    trainable = lora_trainable_params(in_dim, out_dim, rank)
    total = full_linear_params(in_dim, out_dim)
    # ratio = ???
    return ratio

def summarize_lora_project(baseline_metrics, lora_metrics):
    # ==========================================
    # TODO 4: 汇总 baseline 和 LoRA 的项目指标
    # 提示：delta = baseline - lora，正数表示 LoRA 更省或更快。
    # ==========================================
    # param_reduction = ???
    # memory_delta = ???
    # time_delta = ???
    # loss_delta = ???
    return {
        'param_reduction': round(param_reduction, 4),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'step_time_delta_ms': round(time_delta, 2),
        'loss_delta': round(loss_delta, 4),
    }

```


```python
# 测试你的实现
def test_lora_project_template():
    try:
        trainable = lora_trainable_params(8, 8, 2)
        total = full_linear_params(8, 8)
        ratio = lora_param_ratio(8, 8, 2)

        assert trainable == 32, "LoRA 可训练参数量计算不正确！"
        assert total == 64, "完整线性层参数量计算不正确！"
        assert abs(ratio - 0.5) < 1e-12, "LoRA 参数占比计算不正确！"

        baseline = {'trainable_params': 64, 'step_time_ms': 20.0, 'peak_mem_mb': 1024.0, 'final_loss': 0.50}
        lora = {'trainable_params': 32, 'step_time_ms': 22.0, 'peak_mem_mb': 768.0, 'final_loss': 0.52}
        summary = summarize_lora_project(baseline, lora)
        assert summary['param_reduction'] == 0.5, "param_reduction 计算不正确！"
        assert summary['peak_mem_delta_mb'] == 256.0, "peak_mem_delta_mb 计算不正确！"
        assert summary['step_time_delta_ms'] == -2.0, "step_time_delta_ms 计算不正确！"
        assert summary['loss_delta'] == 0.02, "loss_delta 计算不正确！"
        print("✅ LoRA 项目模板代码通过基础校验。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了数值错误")
        elif isinstance(e, AssertionError):
            print(f"❌ 测试失败: {e}")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_lora_project_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 计算单层 LoRA 的可训练参数量
def lora_trainable_params(in_dim, out_dim, rank):
    """Estimate trainable LoRA parameters for a single linear layer."""
    trainable_params = rank * (in_dim + out_dim)
    return trainable_params

# TODO 2: 计算完整线性层的参数量
def full_linear_params(in_dim, out_dim):
    total_params = in_dim * out_dim
    return total_params

# TODO 3: 计算 LoRA 参数占比
def lora_param_ratio(in_dim, out_dim, rank):
    trainable = lora_trainable_params(in_dim, out_dim, rank)
    total = full_linear_params(in_dim, out_dim)
    ratio = trainable / total
    return ratio

# TODO 4: 汇总 baseline 和 LoRA 的项目指标
def summarize_lora_project(baseline_metrics, lora_metrics):
    param_reduction = 1.0 - lora_metrics['trainable_params'] / baseline_metrics['trainable_params']
    memory_delta = baseline_metrics['peak_mem_mb'] - lora_metrics['peak_mem_mb']
    time_delta = baseline_metrics['step_time_ms'] - lora_metrics['step_time_ms']
    loss_delta = lora_metrics['final_loss'] - baseline_metrics['final_loss']
    return {
        'param_reduction': round(param_reduction, 4),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'step_time_delta_ms': round(time_delta, 2),
        'loss_delta': round(loss_delta, 4),
    }

for hidden_size, rank in [(4096, 8), (4096, 16), (8192, 16)]:
    trainable = lora_trainable_params(hidden_size, hidden_size, rank)
    total = full_linear_params(hidden_size, hidden_size)
    ratio = lora_param_ratio(hidden_size, hidden_size, rank)
    print(f"hidden={hidden_size}, rank={rank} -> trainable={trainable:,}, full={total:,}, ratio={ratio:.4%}")

baseline = {'trainable_params': 64, 'step_time_ms': 20.0, 'peak_mem_mb': 1024.0, 'final_loss': 0.50}
lora = {'trainable_params': 32, 'step_time_ms': 22.0, 'peak_mem_mb': 768.0, 'final_loss': 0.52}
print(summarize_lora_project(baseline, lora))

```

### 解析

**1. TODO 1: 计算单层 LoRA 的可训练参数量**
- **实现方式**：LoRA 为一个线性层增加两个低秩矩阵，`A` 的参数量是 `rank * in_dim`，`B` 的参数量是 `rank * out_dim`，合起来是 `rank * (in_dim + out_dim)`。
- **关键点**：这里统计的是 LoRA adapter 的可训练参数，不包括冻结的底座权重。
- **项目意义**：这是 LoRA 微调项目的第一张账本，用来说明训练侧到底少更新了多少参数。

**2. TODO 2: 计算完整线性层的参数量**
- **实现方式**：完整线性层的 weight 参数量是 `in_dim * out_dim`。本节为了突出主线，不额外统计 bias。
- **关键点**：全参线性层是 baseline，用来衡量 LoRA 的参数节省比例。
- **技术细节**：如果真实模型中包含 bias 或多个投影层，需要把这些层逐项累加。

**3. TODO 3: 计算 LoRA 参数占比**
- **实现方式**：先分别计算 LoRA 参数量和完整线性层参数量，再用 `trainable / total` 得到参数占比。
- **关键点**：参数占比越小，说明同一层上需要训练和保存的 adapter 越少。
- **项目意义**：这个比例可以和 step time、peak memory、loss 一起放进项目报告，不能单独作为最终结论。

**4. TODO 4: 汇总 baseline 和 LoRA 项目指标**
- **实现方式**：`param_reduction = 1 - lora_trainable / baseline_trainable`，`memory_delta = baseline_memory - lora_memory`，`time_delta = baseline_time - lora_time`，`loss_delta = lora_loss - baseline_loss`。
- **关键点**：这里的 delta 采用统一口径：资源类指标用 `baseline - LoRA`，正数表示 LoRA 更省；loss 用 `LoRA - baseline`，正数表示 LoRA 效果更差。
- **工程判断**：如果参数和显存明显下降，但 loss 损失很小，LoRA 方案通常值得保留；如果 loss 明显变差，需要继续调整 rank、插层位置或学习率。

**LoRA 微调项目的实验原则**
- **不要重复训练闭环**：完整训练循环已经在第 13 节跑通，本节重点是把 LoRA 放进项目对比里做决策。
- **固定比较口径**：baseline 和 LoRA 必须使用相同模型、数据、batch size、seq len、训练 step 数和评测方式。
- **多指标判断**：LoRA 省参数不等于一定更快；最终要同时看参数量、显存、速度和 loss / accuracy。
- **工程产物**：建议保存 baseline vs LoRA 对比表、rank 设置、插层位置、loss 曲线和下一轮计划。
