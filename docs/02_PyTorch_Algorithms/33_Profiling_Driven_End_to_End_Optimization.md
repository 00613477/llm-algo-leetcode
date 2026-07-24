# 33. Profiling Driven End to End Optimization | profiling 驱动的端到端优化

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `profiling`, `optimization` | **目标人群：** 工程实践与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/33_Profiling_Driven_End_to_End_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

很多优化失败不是因为工具不会用，而是因为流程没有闭环：看到一个耗时点就开始改，改完只看一个指标，最后说不清收益来自哪里，也无法判断这次改动是否值得保留。

本节把 profiling 驱动优化做成一个端到端项目模板：先固定 baseline，再测量定位瓶颈，随后只做一个最小改动并复测，最后把指标变化、瓶颈判断和下一步动作沉淀成报告。代码区只实现最小 benchmark、结果汇总和报告生成，真实项目中的 profiling 截图和优化方案需要基于这份模板继续补充。

**关键词：** `profiling`, `optimization`, `end-to-end`

---

## 前置阅读

**导语：** 先把 profiling 方法、训练/推理对比项目和显存账本看过，再进入端到端优化闭环会更容易判断改动是否有效。
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [30. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/30_LoRA_Fine_Tuning_Project.md)
- [31. Inference Performance Comparison | 推理性能对比实验](../02_PyTorch_Algorithms/31_Inference_Performance_Comparison.md)
- [32. Training Performance Analysis | 训练性能分析](../02_PyTorch_Algorithms/32_Training_Performance_Analysis.md)
- [P0: 20. Profiling and Memory Ledger | 性能剖析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)

## 相关阅读

**导语：** 完成优化闭环后，可以继续看更底层的融合、编译和通信调度，把瓶颈定位落实到具体系统手段。
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)
- [P1: 09. AI Compilers and Graph Optimization | AI 编译器与计算图优化](../01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.md)
- [P1: 27. Communication Scheduling Optimization | 通信调度优化](../01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.md)

---
### Step 1: 定义目标与固定 baseline
先回答一个问题：这次优化到底要解决什么瓶颈，成功标准是什么？

- 固定模型、输入数据、batch size、seq len、硬件环境和运行后端，保证对比对象只差一个变量。
- 明确优化目标，例如降低 step time、提升 throughput、降低 peak memory 或减少通信等待。
- 同时写清约束条件：训练任务要保留 loss / accuracy 约束，推理任务要保留精度、输出一致性或服务 SLA 约束。
- Baseline 需要能稳定复现，不能只跑一次；建议至少 warm-up 若干轮，再测多轮平均值。
- 这一步的目标是让后面的优化有判断标准，而不是只得到一组孤立数字。

### Step 2: Profiling 测量与瓶颈定位

先测 baseline，再把“慢”拆成可解释的瓶颈类型。

- 推荐先记录总耗时、吞吐、峰值显存，再用 profiler 看热点算子和同步点。
- 训练场景优先拆成：数据加载、forward、backward、optimizer step、显存峰值和多卡通信。
- 推理场景优先拆成：prefill、decode、KV cache、采样逻辑、数据搬运和 kernel 开销。
- 不要只找“最慢的一行代码”，而要判断瓶颈属于哪一类资源：计算、显存容量、内存带宽、通信还是调度。
- 这一步的产物应该是一句话瓶颈结论，例如：`当前瓶颈主要来自 decode 阶段 KV cache 读取`。

### Step 3: 修改与复测

针对定位到的瓶颈，只做一个最小可归因改动，然后用同一套指标复测。

- 一次只改一个方向，例如调整 batch size、开启混合精度、替换 kernel、减少同步点或改变 cache 策略。
- 改完后重新测同样的指标，比较 baseline / tuned 的差异。
- 如果改动影响训练 loss、推理输出、显存峰值或系统稳定性，要把代价写清楚。
- 如果某个改动只是在一项指标上变好，却让另一项变差，要把取舍写清楚。
- 这一轮修改的目标是建立因果关系，而不是一次性把所有优化开关都打开。

### Step 4: 复盘与沉淀

回到 Step 1 的目标，用数据判断这次优化是否值得保留。

- 输出 baseline / tuned 对比表，至少包含 step time、throughput、peak memory 和备注。
- 附上 profiling 截图或关键统计，说明瓶颈来自哪一类资源。
- 写清楚本次改动、收益、代价和是否满足约束。
- 如果优化没有达到目标，记录失败原因和下一轮优先级。
- 最终产物应回答：原始瓶颈是什么，做了什么改动，收益有多大，这个改动是否值得保留。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整 profiling 驱动优化流程。下面的代码只实现其中最小、可复用的三块：测平均耗时、汇总 baseline / tuned 指标差异、生成优化报告。真实项目中的 profiling 截图、瓶颈证据和优化策略，需要基于这三个结果继续补充。


```python
import time

```


```python
def benchmark_fn(fn, warmup=3, iters=10):
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和项目报告对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    # start = ???
    for _ in range(iters):
        fn()
    # total = ???
    # avg_time_ms = ???
    return avg_time_ms

def summarize_optimization_result(base_metrics, tuned_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / tuned 的核心指标差异
    # 提示: 正数表示 tuned 相比 baseline 有改善
    # step time / memory 越低越好，throughput 越高越好
    # ==========================================
    # time_delta = ???
    # memory_delta = ???
    # throughput_delta = ???

    summary = {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
    }
    return summary


def format_optimization_report(summary, bottleneck, next_action):
    # ==========================================
    # TODO 3: 生成一段最小优化报告
    # 提示: 把指标变化、瓶颈结论和下一步动作放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    # rows = ???
    # conclusion = ???
    return "\n".join([header, sep] + rows + [conclusion])

```

### 测试


```python
def test_optimization_project_template():
    try:
        counter = {'n': 0}

        def fn():
            counter['n'] += 1

        result = benchmark_fn(fn, warmup=0, iters=2)
        assert counter['n'] == 2, "benchmark 应该运行 iters 次"
        assert result >= 0.0, "平均耗时应该非负"

        baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0, 'throughput': 80.0}
        tuned = {'step_time_ms': 96.0, 'peak_mem_mb': 7168.0, 'throughput': 100.0}
        summary = summarize_optimization_result(baseline, tuned)

        assert summary['step_time_delta_ms'] == 24.0
        assert summary['peak_mem_delta_mb'] == 1024.0
        assert summary['throughput_delta'] == 20.0
        assert summary['time_improved'] is True
        assert summary['memory_improved'] is True
        assert summary['throughput_improved'] is True

        report = format_optimization_report(summary, 'backward kernel 占比过高', '保留混合精度并继续检查 optimizer')
        assert '| 指标 | 变化 | 判断 |' in report
        assert 'backward kernel 占比过高' in report
        assert '保留混合精度并继续检查 optimizer' in report

        print("✅ profiling 驱动的端到端优化项目模板代码通过基础校验。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_optimization_project_template()

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
def benchmark_fn(fn, warmup=3, iters=10):
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和项目报告对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    avg_time_ms = total / iters * 1000
    return avg_time_ms


def summarize_optimization_result(base_metrics, tuned_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / tuned 的核心指标差异
    # 提示: 正数表示 tuned 相比 baseline 有改善
    # step time / memory 越低越好，throughput 越高越好
    # ==========================================
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    memory_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    throughput_delta = tuned_metrics['throughput'] - base_metrics['throughput']

    summary = {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
    }
    return summary


def format_optimization_report(summary, bottleneck, next_action):
    # ==========================================
    # TODO 3: 生成一段最小优化报告
    # 提示: 把指标变化、瓶颈结论和下一步动作放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| step time | {summary['step_time_delta_ms']} ms | {'改善' if summary['time_improved'] else '未改善'} |",
        f"| peak memory | {summary['peak_mem_delta_mb']} MB | {'改善' if summary['memory_improved'] else '未改善'} |",
        f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
    ]
    conclusion = f"瓶颈判断：{bottleneck}。下一步：{next_action}。"
    return "\n".join([header, sep] + rows + [conclusion])

```

### 解析

- **这一题要解决什么**：把 profiling 优化流程压缩成一个最小可复用模板，保证每次优化都能留下可比较的指标和明确结论。
- **为什么这样做**：性能优化不能只看单次运行结果，必须固定 baseline、测量同一组指标，并把改动前后的差异收敛成项目报告。
- **带走的直觉**：profiling 的价值不是“找到一个慢点”，而是建立 `测量 -> 定位 -> 修改 -> 复测 -> 复盘` 的闭环。

**1. TODO 1 (benchmark_fn)**

- **warmup**：先运行若干轮，不计入统计，避免初始化、缓存和调度抖动影响结果。
- **计时范围**：只把正式测量的 `iters` 轮放进 `start / total` 之间。
- **单位统一**：返回 ms，而不是秒，方便和 step time、latency 表格放在一起比较。
- **工程注意**：真实 GPU 场景下通常还需要 `torch.cuda.synchronize()`，否则异步 kernel 可能让计时偏小；本节保持 CPU-first，不强制加入。

**2. TODO 2 (summarize_optimization_result)**

- **step time 差值**：`baseline - tuned`，正数表示 tuned 更快。
- **peak memory 差值**：`baseline - tuned`，正数表示 tuned 更省显存。
- **throughput 差值**：`tuned - baseline`，正数表示 tuned 吞吐更高。
- **布尔判断**：`time_improved / memory_improved / throughput_improved` 把数值变化变成可读结论，方便项目报告直接引用。

**3. TODO 3 (format_optimization_report)**

- **表格部分**：把核心指标变化放到同一张 Markdown 表里，便于复盘和横向比较。
- **瓶颈判断**：不要只输出数字，还要写清楚瓶颈来自哪里，例如数据加载、backward kernel、KV cache 或通信同步。
- **下一步动作**：每轮优化结束都应该留下后续优先级，否则下一轮很容易重新从零开始定位。

**项目化原则**

- **一次只改一个变量**：否则收益不可归因。
- **指标要成组出现**：只看变快不够，还要看显存、吞吐、loss / 精度或输出一致性。
- **结论要回扣目标**：最终判断必须回答 Step 1 的问题：这次优化是否达成目标，是否值得保留。
