# 34. Distributed Parallel Benchmark | 分布式并行基准

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `distributed`, `benchmark` | **目标人群：** 分布式训练工程与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/34_Distributed_Parallel_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

分布式训练里最常见的误区是先选方案再找理由：模型放不下就上 ZeRO，吞吐不够就加 Pipeline，单层太大就开 Tensor Parallelism。但不同并行策略切分的对象不同，带来的通信、显存和调度代价也不同，不能只凭直觉选型。

本节把 ZeRO、Pipeline 和 Tensor Parallelism 放进同一套 benchmark 项目模板：先固定 workload 和评测口径，再记录 peak memory、throughput、latency 和 communication overhead，最后输出并行策略选型结论。代码区只实现最小 benchmark、指标汇总和选型报告，真实项目中的多卡运行和 profiler 证据需要基于这份模板继续补充。

**关键词：** `distributed training`, `benchmark`, `parallelism`

---

## 前置阅读

**导语：** 先把三类并行策略和 profiling 方法看过，再进入并行 benchmark 项目会更容易把显存、吞吐和通信代价区分开。
- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.md)
- [28. Pipeline Parallelism MicroBatch | Pipeline 并行微批次](../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.md)
- [29. Tensor Parallelism Sim | Tensor 并行模拟](../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 05. Communication Topologies | 通信拓扑与分布式基石](../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)

## 相关阅读

**导语：** 完成并行 benchmark 后，可以继续看通信调度、并行策略决策和多卡策略项目。
- [P1: 26. Parallel Strategy Decision Framework | 并行策略决策框架](../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md)
- [P1: 27. Communication Scheduling Optimization | 通信调度优化](../01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.md)
- [35. Multi-GPU Strategy Selection Project | 多卡策略选择项目](../02_PyTorch_Algorithms/35_Multi_GPU_Strategy_Selection_Project.md)

---
### Step 1: 统一 workload 与评测口径
先回答一个问题：在同一模型、同一输入和同一硬件条件下，哪种并行策略更适合当前瓶颈？

- 固定模型结构、参数量、输入长度、global batch size、micro-batch 数和训练 step 数。
- 固定硬件环境、GPU 数、网络拓扑和运行后端，避免把环境差异误判为策略收益。
- 统一记录 peak memory、throughput、latency / step time、communication overhead 和是否稳定收敛。
- 先写清约束条件，例如单卡显存上限、最低吞吐要求、最大可接受延迟或通信占比。
- 这一步的目标是保证 ZeRO、Pipeline、Tensor Parallelism 能在同一口径下比较。

### Step 2: 运行并行策略对比

把不同并行策略放到同一套指标表里，观察它们分别改善了什么，又引入了什么代价。

- ZeRO 主要观察训练状态切分后的单卡显存下降，以及 Reduce-Scatter / All-Gather 的通信代价。
- Pipeline 主要观察模型按层切分后的显存下降、吞吐变化和 bubble ratio。
- Tensor Parallelism 主要观察单层矩阵切分后的显存 / 计算分摊，以及 All-Gather / All-Reduce 开销。
- 每种策略至少跑 baseline 和 candidate 两组结果，避免只描述理论收益。
- 重点不是证明某个策略永远最好，而是看它在当前 workload 下是否解决了主要瓶颈。

### Step 3: 归因收益与代价

把指标差异转成可解释的工程判断。

- 如果 peak memory 降了但 throughput 也下降，要判断通信或调度开销是否抵消了显存收益。
- 如果 throughput 提升但 latency 变差，要说明这个策略更适合离线训练还是在线服务。
- 如果通信占比过高，要记录瓶颈来自 All-Reduce、All-Gather、Reduce-Scatter 还是 pipeline bubble。
- 如果某个策略只在大 batch 或特定模型规模下有效，要写清适用条件。
- 这一阶段的产物应该是“收益 + 代价 + 适用条件”，而不是只输出排行榜。

### Step 4: 输出选型报告

最后把实验结果收成并行策略决策建议。

- 输出 ZeRO / Pipeline / Tensor Parallelism 的对比表，至少包含 peak memory、throughput、latency 和 communication overhead。
- 写清楚每种策略适合什么模型规模、显存瓶颈和通信条件。
- 给出“什么时候选它、什么时候别选它”的结论。
- 如果后面要扩展 FSDP、sequence parallel 或 expert parallel，就沿用同一套评测字段。
- 最终产物应回答：当前 workload 下推荐哪种并行策略，理由是什么，下一轮需要验证什么。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整分布式并行 benchmark 流程。下面的代码只实现其中最小、可复用的三块：测平均耗时、汇总 baseline / parallel 指标差异、生成并行策略报告。真实项目中的多卡启动、通信 trace 和 profiler 截图，需要基于这三个结果继续补充。


```python
import time

```


```python
def benchmark_fn(fn, warmup=2, iters=5):
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和 latency / step time 对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    # start = ???
    for _ in range(iters):
        fn()
    # total = ???
    # avg_time_ms = ???
    return avg_time_ms


def summarize_parallel_result(base_metrics, parallel_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / parallel 的核心指标差异
    # 提示: memory / latency / communication 越低越好，throughput 越高越好
    # 正数表示 parallel 相比 baseline 有改善
    # ==========================================
    # memory_delta = ???
    # throughput_delta = ???
    # latency_delta = ???
    # communication_delta = ???

    summary = {
        'memory_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'latency_delta_ms': round(latency_delta, 2),
        'communication_delta_ms': round(communication_delta, 2),
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
        'latency_improved': latency_delta > 0,
        'communication_improved': communication_delta > 0,
    }
    return summary


def format_parallel_report(strategy_name, summary, recommendation):
    # ==========================================
    # TODO 3: 生成并行策略选型报告
    # 提示: 把策略名、核心指标变化和推荐结论放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    # rows = ???
    # conclusion = ???
    return "\n".join([f"策略：{strategy_name}", header, sep] + rows + [conclusion])

```

### 测试


```python
def test_parallel_project_template():
    try:
        counter = {'n': 0}

        def fn():
            counter['n'] += 1

        avg = benchmark_fn(fn, warmup=0, iters=2)
        assert counter['n'] == 2, "benchmark 应该运行 iters 次"
        assert avg >= 0.0, "平均耗时应该非负"

        baseline = {
            'peak_mem_mb': 12000.0,
            'throughput': 80.0,
            'latency_ms': 120.0,
            'communication_ms': 30.0,
        }
        parallel = {
            'peak_mem_mb': 9000.0,
            'throughput': 100.0,
            'latency_ms': 96.0,
            'communication_ms': 24.0,
        }
        summary = summarize_parallel_result(baseline, parallel)

        assert summary['memory_delta_mb'] == 3000.0
        assert summary['throughput_delta'] == 20.0
        assert summary['latency_delta_ms'] == 24.0
        assert summary['communication_delta_ms'] == 6.0
        assert summary['memory_improved'] is True
        assert summary['throughput_improved'] is True
        assert summary['latency_improved'] is True
        assert summary['communication_improved'] is True

        report = format_parallel_report('Tensor Parallelism', summary, '当前单层矩阵较大，优先保留 TP 并继续观察 All-Reduce')
        assert 'Tensor Parallelism' in report
        assert '| 指标 | 变化 | 判断 |' in report
        assert 'All-Reduce' in report

        print("✅ 分布式并行基准项目模板代码通过基础校验。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_parallel_project_template()

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
def benchmark_fn(fn, warmup=2, iters=5):
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和 latency / step time 对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    avg_time_ms = total / iters * 1000
    return avg_time_ms


def summarize_parallel_result(base_metrics, parallel_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / parallel 的核心指标差异
    # 提示: memory / latency / communication 越低越好，throughput 越高越好
    # 正数表示 parallel 相比 baseline 有改善
    # ==========================================
    memory_delta = base_metrics['peak_mem_mb'] - parallel_metrics['peak_mem_mb']
    throughput_delta = parallel_metrics['throughput'] - base_metrics['throughput']
    latency_delta = base_metrics['latency_ms'] - parallel_metrics['latency_ms']
    communication_delta = base_metrics['communication_ms'] - parallel_metrics['communication_ms']

    summary = {
        'memory_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'latency_delta_ms': round(latency_delta, 2),
        'communication_delta_ms': round(communication_delta, 2),
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
        'latency_improved': latency_delta > 0,
        'communication_improved': communication_delta > 0,
    }
    return summary


def format_parallel_report(strategy_name, summary, recommendation):
    # ==========================================
    # TODO 3: 生成并行策略选型报告
    # 提示: 把策略名、核心指标变化和推荐结论放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| peak memory | {summary['memory_delta_mb']} MB | {'改善' if summary['memory_improved'] else '未改善'} |",
        f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
        f"| latency | {summary['latency_delta_ms']} ms | {'改善' if summary['latency_improved'] else '未改善'} |",
        f"| communication | {summary['communication_delta_ms']} ms | {'改善' if summary['communication_improved'] else '未改善'} |",
    ]
    conclusion = f"推荐结论：{recommendation}。"
    return "\n".join([f"策略：{strategy_name}", header, sep] + rows + [conclusion])

```

### 解析

- **这一题要解决什么**：把并行策略 benchmark 收敛成一套最小模板，方便比较 ZeRO、Pipeline 和 Tensor Parallelism 的收益与代价。
- **为什么这样做**：并行策略不是只看显存或吞吐的单项结果，而是要同时比较 memory、throughput、latency 和 communication overhead。
- **带走的直觉**：不同并行策略切分的对象不同，最终选型必须回到当前 workload 的主要瓶颈。

**1. TODO 1 (benchmark_fn)**

- **warmup**：先运行若干轮，不计入统计，避免初始化和缓存抖动影响结果。
- **平均耗时**：正式测量阶段只统计 `iters` 轮，并返回单次平均耗时。
- **单位统一**：返回 ms，便于和 latency / step time / communication time 放到同一张表中。
- **真实多卡场景**：如果使用 GPU，需要在计时前后加入同步，避免异步执行导致计时偏小。

**2. TODO 2 (summarize_parallel_result)**

- **显存差值**：`baseline - parallel`，正数表示并行策略降低了单卡显存。
- **吞吐差值**：`parallel - baseline`，正数表示并行策略提升了处理能力。
- **延迟差值**：`baseline - parallel`，正数表示单步或单请求更快。
- **通信差值**：`baseline - parallel`，正数表示通信等待减少；如果为负，说明并行策略引入了更多通信开销。

**3. TODO 3 (format_parallel_report)**

- **策略名**：报告必须写清当前评估的是 ZeRO、Pipeline 还是 Tensor Parallelism。
- **对比表**：将显存、吞吐、延迟和通信放到同一张表，避免只凭单项指标做判断。
- **推荐结论**：用一句话说明当前策略是否值得保留，以及下一轮需要继续观察哪类开销。

**并行 benchmark 的项目原则**

- **先固定 workload**：模型、输入长度、batch、GPU 数和后端都要固定。
- **一次比较一个策略维度**：不要同时改并行策略、batch size 和精度模式。
- **指标必须成组解释**：显存下降但通信暴涨，未必是更好的方案。
- **结论要能指导选型**：最终输出不只是数字，而是“当前资源条件下该选什么、为什么”。
