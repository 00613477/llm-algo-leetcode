# 31. Inference Performance Comparison | 推理性能对比实验

**难度：** Hard | **环境：** CPU-first | **标签：** `推理`, `benchmark`, `profiling` | **目标人群：** 推理工程与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/31_Inference_Performance_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

推理优化里最容易出现的问题是只看单点收益：某个方案 latency 更低，另一个方案 throughput 更高，还有一个方案显存更省。如果模型、输入、batch size、精度和评测方式没有固定，这些数字很难放在一起比较，也很难支撑工程选型。

本节把推理优化做成一个对比项目：围绕同一个模型和同一批输入，记录 prefill latency、decode latency、throughput 和 peak memory，再回答“在给定约束下哪种推理方案最划算”。代码区只实现最小 benchmark、指标汇总和对比表生成，真实项目中的 profiling 截图、瓶颈判断和选型结论需要基于这些结果继续完成。

**关键词：** `benchmark`, `latency`, `throughput`, `memory`

---

### Step 1: 定义问题与固定 baseline
先回答一个问题：在同一模型、同一输入集和同一硬件环境下，哪种推理策略更划算？

- 固定模型、数据、batch size、seq len、运行环境和评测方式，保证对比对象只差一个变量。
- Baseline 建议从 `PyTorch eager + batch=1 + 固定 seq len + warm-up 10 轮 + 测量 100 轮` 开始；模型可以选 GPT-2 / LLaMA-2-7B 这类容易复现的起点。
- 统一输出 token 数和统计口径，只看一组核心指标，例如 prefill latency / decode latency / throughput(tok/s) / peak memory。
- 提前列出至少两个对比方向，如推理后端、精度模式、batch size、解码策略，避免后面没有可比对象。
- 这节的目标不是比较“谁都能跑”，而是找出在相同约束下最划算的候选方案，并写清楚为什么选它。
- 如果是训练任务，再补一条精度或 loss 约束，避免只追求更快。

### Step 2: 测量与定位
记录 profiling 结果，判断主要瓶颈类型是什么。

- 推荐工具：PyTorch 原生推理用 `torch.profiler + tensorboard`，通用 CPU profiling 用 `py-spy` 或 `cProfile`，显存跟踪用 `torch.cuda.memory_stats()`；如果是 vLLM / TensorRT-LLM，则优先用其内置 profiling 接口。
- 先跑一轮 baseline，再看时间分布、显存曲线和热点算子。
- 推理场景优先拆成：算子耗时、数据搬运、显存占用、解码开销；多卡场景再看通信同步。
- 重点不是找一个最慢点，而是判断瓶颈主要落在哪一层。
- 这一步的目标是把“慢”具体化，而不是先急着改代码。

### Step 3: 修改与复测
针对瓶颈做最小修改，再次测量验证收益。

- 一次只改一个方向，例如 batch size、precision、cache 策略或 kernel 实现。
- 常见瓶颈与优化手段可先对照：

| 瓶颈类型 | 可尝试的优化手段 |
| --- | --- |
| 算子耗时高 | `torch.compile`、算子融合、精度降级（FP16/INT8） |
| 数据搬运慢 | 预加载、`pin_memory`、异步数据加载 |
| 显存峰值高 | KV cache 量化、激活检查点、减少 batch size |
| 解码开销大 | 改用 vLLM / TensorRT-LLM、调整采样参数 |
| 通信同步重 | 减少同步点、调整并行策略、合并通信 |

- 改完后重新测同样的指标，比较改前 / 改后差异。
- 如果某个改动只是在一项指标上变好，却让另一项变差，要把取舍写清楚。
- 这一轮修改的目标是建立因果关系，而不是同时把所有参数都调一遍。

### Step 4: 复盘与沉淀
先回到 Step 1 定义的目标和约束，用对比数据验证是否达成目标。

- 输出改动前后对比表、profiling 截图和最终判断，把这次经验收成可复用的优化记录。
- 推荐对比表字段：配置 / prefill latency / decode latency / throughput(tok/s) / peak memory / 备注。

| 配置 | prefill latency | decode latency | throughput(tok/s) | peak memory | 备注 |
| --- | --- | --- | --- | --- | --- |
| Baseline |  |  |  |  |  |
| Candidate |  |  |  |  |  |

- 记录本次瓶颈来自哪里，以及下次优先看哪一层。
- 把这次优化的取舍和结论写成可复用的排障路径。
- 如果还有后续优化空间，就把下一轮优先级列出来。
- 最终判断要直接回答 Step 1 的问题：在给定约束下，哪种策略最划算，理由是什么。
- 最终产物至少应包含：对比表、瓶颈结论、改动说明和下一轮计划。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整推理选型项目流程。下面的代码只实现其中最小、可复用的三块：测平均耗时、汇总核心指标、生成对比表。真实项目中的 profiling 截图、策略选择和结论判断，需要基于这三个结果继续完成。


```python
import time

```


```python
# 补全推理性能对比的三个关键函数
# 目标：完成 benchmark -> summary -> report 的最小实验链路

def benchmark_fn(fn, warmup=3, iters=10):
    # ==========================================
    # TODO 1: 先 warmup，再计时求平均
    # 提示：使用 time.perf_counter() 记录起点和终点，
    #   total 表示正式测量阶段的总耗时。
    # ==========================================
    for _ in range(warmup):
        fn()
    # start = ???
    for _ in range(iters):
        fn()
    # end = ???
    # total = ???
    return total / iters

def summarize_inference_result(prefill_ms, decode_ms, peak_mem_mb, generated_tokens):
    # ==========================================
    # TODO 2: 汇总核心指标
    # 提示：total 是 prefill 和 decode 的总延迟；
    #   throughput 表示每秒生成 token 数。
    # ==========================================
    # total_ms = ???
    # decode_share = ???
    # throughput_tok_s = ???
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'throughput_tok_s': round(throughput_tok_s, 2),
        'total_ms': round(total_ms, 2),
        'decode_share': round(decode_share, 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def format_comparison_report(baseline_name, baseline_summary, candidate_name, candidate_summary):
    # ==========================================
    # TODO 3: 按 Step 4 字段拼接对比表
    # 提示：每一行从 summary 中取 prefill/decode/throughput/peak_mem，
    #   最后用换行符拼成 markdown table。
    # ==========================================
    rows = [
        (baseline_name, baseline_summary),
        (candidate_name, candidate_summary),
    ]
    # header = ???
    # sep = ???
    # lines = ???
    for name, summary in rows:
        # report_row = ???
        lines.append(report_row)
    return "\n".join(lines)

```


```python
# 测试你的实现
def test_inference_project_template():
    try:
        summary = summarize_inference_result(10.0, 5.0, 256.0, 100)
        assert summary['total_ms'] == 15.0, "total_ms 计算不正确！"
        assert summary['decode_share'] == 0.333, "decode_share 计算不正确！"
        assert summary['throughput_tok_s'] == 6666.67, "throughput_tok_s 计算不正确！"
        assert summary['peak_mem_mb'] == 256.0, "peak_mem_mb 计算不正确！"

        counter = {'n': 0}
        def fn():
            counter['n'] += 1

        avg = benchmark_fn(fn, warmup=0, iters=3)
        assert counter['n'] == 3, "benchmark_fn 没有正确执行迭代次数！"
        assert avg >= 0.0, "benchmark_fn 的返回值应为非负数！"
        report = format_comparison_report("Baseline", summary, "Candidate", summary)
        assert "Baseline" in report and "Candidate" in report, "对比表未包含 baseline / candidate！"
        assert "throughput(tok/s)" in report, "对比表字段不完整！"
        print("✅ 推理性能对比项目模板代码通过基础校验。")

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
            print("代码可能未完成，导致了张量维度错误")
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


test_inference_project_template()

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
# TODO 1: 统计平均 benchmark 耗时
def benchmark_fn(fn, warmup=3, iters=10):
    """Measure average runtime after warmup."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    total = end - start
    return total / iters

# TODO 2: 汇总推理指标
def summarize_inference_result(prefill_ms, decode_ms, peak_mem_mb, generated_tokens):
    total_ms = prefill_ms + decode_ms
    decode_share = decode_ms / total_ms if total_ms else 0.0
    throughput_tok_s = generated_tokens / (total_ms / 1000.0) if total_ms and generated_tokens else 0.0
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'throughput_tok_s': round(throughput_tok_s, 2),
        'total_ms': round(total_ms, 2),
        'decode_share': round(decode_share, 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 3: 生成 baseline vs candidate 的对比表
def format_comparison_report(baseline_name, baseline_summary, candidate_name, candidate_summary):
    rows = [
        (baseline_name, baseline_summary),
        (candidate_name, candidate_summary),
    ]
    header = "| 配置 | prefill latency | decode latency | throughput(tok/s) | peak memory | 备注 |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for name, summary in rows:
        report_row = f"| {name} | {summary['prefill_ms']} | {summary['decode_ms']} | {summary['throughput_tok_s']} | {summary['peak_mem_mb']} | {summary.get('note', '')} |"
        lines.append(report_row)
    return "\n".join(lines)

baseline = summarize_inference_result(42.5, 18.0, 5120.0, 100)
baseline['note'] = 'Baseline'
candidate = summarize_inference_result(35.0, 15.0, 4608.0, 100)
candidate['note'] = 'Candidate'
print(format_comparison_report('Baseline', baseline, 'Candidate', candidate))

```

### 解析

**1. TODO 1: 统计平均 benchmark 耗时**
- **实现方式**：先执行 `warmup` 轮预热，再用 `time.perf_counter()` 记录正式测量阶段的起点和终点，最后返回 `total / iters`。
- **关键点**：warmup 不计入结果，避免首次运行的缓存、编译或初始化开销污染平均延迟。
- **技术细节**：这里返回的是单次调用的平均耗时，适合作为 baseline 和 candidate 的统一比较口径。

**2. TODO 2: 汇总推理指标**
- **实现方式**：`total_ms = prefill_ms + decode_ms`，`decode_share = decode_ms / total_ms`，`throughput_tok_s = generated_tokens / (total_ms / 1000.0)`。
- **关键点**：latency 反映单次请求速度，throughput 反映单位时间产出，二者需要一起看。
- **技术细节**：`decode_share` 可以帮助判断瓶颈是否主要来自 decode 阶段；若 `total_ms` 为 0，需要避免除零。

**3. TODO 3: 生成对比表**
- **实现方式**：先准备固定表头和分隔行，再遍历 baseline / candidate，将每个 summary 拼成一行 markdown table。
- **关键点**：表格字段要和 Step 4 保持一致，包括 prefill latency / decode latency / throughput(tok/s) / peak memory / 备注。
- **技术细节**：`"\n".join(lines)` 会把多行字符串用换行符连接起来，生成可直接展示的 markdown 表格。

**推理性能对比的实验原则**
- **变量控制**：同一轮对比中只改一个变量，例如 batch size、precision、推理后端或 cache 策略。
- **指标闭环**：每次实验至少记录 latency、throughput 和 peak memory，避免只用一个指标判断方案好坏。
- **结果复盘**：最终输出要回扣 Step 1 的问题：在给定约束下，哪种推理策略最划算，理由是什么。
- **工程产物**：建议把 profiling 截图、对比表、瓶颈结论和下一轮计划一起保存，形成可复用的优化记录。
