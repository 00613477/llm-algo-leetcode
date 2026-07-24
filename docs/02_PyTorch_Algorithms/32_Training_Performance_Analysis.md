# 32. Training Performance Analysis | 训练性能分析

**难度：** Hard | **环境：** CPU-first | **标签：** `训练`, `profiling`, `显存` | **目标人群：** 训练工程与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/32_Training_Performance_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

训练变慢时，不能只看一个总耗时。一个 training step 里面可能慢在数据加载，也可能慢在 forward / backward、optimizer step、显存峰值或同步等待；如果不先拆开定位，优化动作很容易变成盲目调参。

本节把训练性能分析做成一个项目模板：先固定 baseline，再把 step time、samples/s、peak memory 和 loss 放到同一张表里，对比改动前后的收益与代价。代码区只实现最小训练 step 测量和结果汇总，真实项目中的 profiling 截图、瓶颈拆解和下一轮优化计划需要基于这些指标继续完成。

**关键词：** `training`, `profiling`, `memory`, `step time`

---

### Step 1: 定义训练 baseline
先回答一个问题：在固定训练配置下，单个 training step 到底慢在哪里，显存峰值卡在哪里？

- 固定模型、数据集、batch size、seq len、优化器、精度模式和训练 step 数，保证前后对比可复现。
- Baseline 建议从 `PyTorch eager + 固定 batch + 固定输入长度 + warm-up 若干步 + 测量 N 个 step` 开始。
- 统一记录核心指标：step time / samples per second / peak memory / loss。
- 这节的目标不是单纯让 step 更快，而是在 loss 不异常的前提下定位训练瓶颈。

### Step 2: 拆解 training step

训练性能分析要把一个 step 拆开看，而不是只盯总耗时。

- 数据加载：DataLoader、CPU 预处理、CPU -> GPU 拷贝是否让 GPU 等待。
- 前向计算：Attention、Linear、LayerNorm 等 forward kernel 是否占主要时间。
- 反向计算：backward kernel、梯度计算和梯度累积是否成为瓶颈。
- 优化器更新：optimizer step 是否占用明显时间，尤其是大模型参数更新。
- 显存峰值：激活、梯度、优化器状态和临时 buffer 是否接近上限。
- 同步开销：是否存在不必要的 CPU/GPU 同步或多卡通信等待。

这一步的目标是把“训练慢”具体化成某一类瓶颈。

### Step 3: 针对瓶颈做最小修改

定位瓶颈后，只选择一个方向修改，再用同样的指标复测。

| 瓶颈类型 | 可尝试的优化手段 |
| --- | --- |
| 数据加载慢 | 增加 `num_workers`、开启 `pin_memory`、预取或缓存样本 |
| forward / backward 慢 | 混合精度、`torch.compile`、算子融合、减少不必要同步 |
| optimizer 慢 | 使用 fused optimizer、减少参数更新规模、LoRA / adapter 微调 |
| 显存峰值高 | gradient checkpointing、减小 batch size、activation offload、量化或 LoRA |
| step time 抖动大 | 增加 warm-up、固定输入形状、检查数据加载和同步点 |

- 改完后重新测同样的指标，比较 baseline / tuned 的差异。
- 如果 step time 变快但 loss 异常、显存更高或稳定性变差，要把取舍写清楚。
- 这一轮修改的目标是建立因果关系，而不是同时把所有参数都调一遍。

### Step 4: 输出训练性能报告

回到 Step 1 的目标，用对比数据判断这次优化是否真的有效。

- 输出 baseline / tuned 对比表，至少包含 step time、samples/s、peak memory、loss 和备注。
- 附上 profiling 截图或关键统计，说明瓶颈来自数据、forward、backward、optimizer 还是显存。
- 写清楚本次改动、收益、代价和是否满足 loss / 收敛约束。
- 如果还有后续优化空间，就列出下一轮优先级。
- 最终产物应回答：训练一步慢在哪里，显存卡在哪里，当前改动是否值得保留。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整训练性能分析流程。下面的代码只实现其中最小、可复用的两块：测量训练 step 的平均耗时与峰值显存，并汇总 baseline / tuned 的差异。真实项目中的 forward / backward / optimizer 拆解和 loss 约束，需要在 profiling 报告中继续补充。


```python
import time
import torch

```


```python
# 完成训练性能统计的两个函数
# 目标：完成 measure -> compare 的最小训练性能分析链路

def measure_train_step(train_step_fn, warmup=2, iters=8):
    # ==========================================
    # TODO 1: 记录平均 step time 和 peak memory
    # 提示：先 warmup，再测正式迭代；GPU 场景下记录 peak memory。
    # ==========================================
    for _ in range(warmup):
        train_step_fn()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # start = ???
    for _ in range(iters):
        train_step_fn()
    # end = ???
    # elapsed = ???

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        # peak_mem_mb = ???
        pass

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def summarize_training_result(base_metrics, tuned_metrics):
    # ==========================================
    # TODO 2: 比较 baseline 和 tuned 的指标差值
    # 提示：delta = baseline - tuned，正数表示 tuned 更省或更快。
    # ==========================================
    # time_delta = ???
    # mem_delta = ???
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
    }

```


```python
# 测试你的实现
def test_training_project_template():
    try:
        counter = {'n': 0}

        def train_step():
            counter['n'] += 1

        result = measure_train_step(train_step, warmup=0, iters=2)
        assert counter['n'] == 2, "measure_train_step 没有正确执行训练迭代次数！"
        assert 'step_time_ms' in result and 'peak_mem_mb' in result, "训练统计字段不完整！"
        assert result['step_time_ms'] >= 0.0, "step_time_ms 应为非负数！"
        assert result['peak_mem_mb'] >= 0.0, "peak_mem_mb 应为非负数！"

        baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0}
        tuned = {'step_time_ms': 98.0, 'peak_mem_mb': 6144.0}
        summary = summarize_training_result(baseline, tuned)
        assert summary['step_time_delta_ms'] == 22.0, "step_time_delta_ms 计算不正确！"
        assert summary['peak_mem_delta_mb'] == 2048.0, "peak_mem_delta_mb 计算不正确！"
        assert summary['time_improved'] is True, "time_improved 判断不正确！"
        assert summary['memory_improved'] is True, "memory_improved 判断不正确！"
        print("✅ 训练性能分析项目模板代码通过基础校验。")

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


test_training_project_template()

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
import time
import torch

# TODO 1: 测量训练 step 的平均耗时和峰值显存
def measure_train_step(train_step_fn, warmup=2, iters=8):
    for _ in range(warmup):
        train_step_fn()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        train_step_fn()
    end = time.perf_counter()
    elapsed = (end - start) / iters

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 2: 汇总 baseline 和 tuned 的差异
def summarize_training_result(base_metrics, tuned_metrics):
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    mem_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
    }

counter = {'n': 0}
def train_step():
    counter['n'] += 1
print(measure_train_step(train_step, warmup=0, iters=2))

```

### 解析

**1. TODO 1: 统计训练 step 耗时和峰值显存**
- **实现方式**：先执行 `warmup` 轮训练 step 预热，再用 `time.perf_counter()` 记录正式测量阶段的起点和终点，最后用 `(end - start) / iters` 得到平均 step time。
- **关键点**：warmup 不计入结果，避免首次运行的数据加载、kernel 初始化或缓存状态影响平均耗时。
- **显存统计**：GPU 场景下先调用 `torch.cuda.reset_peak_memory_stats()` 清空历史峰值，再用 `torch.cuda.max_memory_allocated()` 读取本轮训练的峰值显存。CPU 场景下返回 `0.0`，保证模板可以在无 GPU 环境中运行。

**2. TODO 2: 汇总 baseline 和 tuned 的差异**
- **实现方式**：`time_delta = baseline_step_time - tuned_step_time`，`mem_delta = baseline_peak_mem - tuned_peak_mem`。
- **关键点**：这里统一用 `baseline - tuned`，所以 delta 为正表示优化后更快或更省显存。
- **技术细节**：`time_improved` 和 `memory_improved` 只是快速判断标记，真正复盘时还要结合 loss、吞吐和收敛稳定性一起看。

**训练性能分析的实验原则**
- **固定 baseline**：同一轮对比中固定模型、数据、batch size、seq len、优化器和评测方式。
- **一次只改一个变量**：例如只改 batch size、混合精度、gradient checkpointing 或数据加载方式，避免结果不可归因。
- **指标一起看**：step time 变快但 peak memory、loss 或稳定性变差时，要把取舍写清楚。
- **瓶颈归因**：如果 step time 没有改善，需要回到 profiling 结果，判断瓶颈来自数据等待、前向 / 反向算子，还是显存压力。
- **工程产物**：建议保存对比表、profiling 截图、瓶颈结论和下一轮计划，形成可复用的训练性能排障记录。
