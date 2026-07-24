# 35. Quantized Inference and Deployment | 量化推理与部署

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `quantization`, `deployment` | **目标人群：** 推理工程与模型部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/35_Quantized_Inference_and_Deployment.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

量化方案真正进入部署时，不能只看“显存少了多少”。W8A16、INT8、4-bit 和 QLoRA 相关方案会同时影响 latency、throughput、VRAM、输出误差和部署复杂度；如果没有固定输入、评测口径和精度约束，很容易把压缩效果误判成部署收益。

本节把量化推理做成一个部署选型项目模板：先固定 baseline 和量化对象，再比较量化前后的速度、显存和误差，最后输出“是否适合部署”的结论。代码区只实现最小 benchmark、量化结果汇总和部署报告生成，真实项目中的真实模型量化、输出一致性检查和线上约束需要基于这份模板继续补充。

**关键词：** `quantization`, `inference`, `deployment`

---

## 前置阅读

**导语：** 先把 W8A16、QLoRA、量化理论和推理性能对比看过，再进入量化部署项目会更容易判断压缩收益是否值得保留。
- [25. Quantization W8A16 | W8A16 量化](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [31. Inference Performance Comparison | 推理性能对比实验](../02_PyTorch_Algorithms/31_Inference_Performance_Comparison.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)

## 相关阅读

**导语：** 完成量化部署选型后，可以继续看 profiling、显存账本和算子融合，把部署结论落到具体系统瓶颈上。
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)
- [33. Profiling-Driven End-to-End Optimization | profiling 驱动的端到端优化](../02_PyTorch_Algorithms/33_Profiling_Driven_End_to_End_Optimization.md)

---
### Step 1: 固定部署目标与量化对象
先回答一个问题：这次量化是为了解决显存、吞吐、延迟，还是部署成本？

- 固定模型、输入集、batch size、seq len、解码策略、硬件环境和推理后端。
- 明确量化对象：只量化权重、权重和激活都量化，还是只做底座 4-bit 存储。
- 明确量化粒度：per-tensor、per-channel、per-group 或 block-wise，不同粒度会影响误差和元数据开销。
- 写清约束条件，例如最大可接受输出误差、最低 throughput、最大 latency、最大 VRAM 或是否允许依赖特定推理库。
- 这一步的目标是先定义“可部署”的标准，而不是只追求更低 bit 数。

### Step 2: 跑通 baseline 与量化方案

在同一输入集上比较量化前后结果，先确认链路正确，再谈优化收益。

- Baseline 先记录 FP16 / FP32 的 latency、throughput、VRAM 和输出参考结果。
- Candidate 再记录 W8A16、INT8、4-bit 或其他量化方案的同一组指标。
- 输出误差至少要有一个可复查指标，例如 max error、cosine similarity、perplexity 变化或任务 accuracy 变化。
- 如果 latency 没变快，要区分是量化 kernel 没生效、反量化开销过大，还是瓶颈本来不在权重读取。
- 如果 VRAM 下降但输出误差明显增大，要先回到 scale 粒度、异常值处理和校准数据上排查。

### Step 3: 归因部署收益与代价

把指标差异转成部署判断，而不是只输出“量化后更小”。

- latency 下降说明单请求更快，但不一定代表吞吐一定提升。
- throughput 上升说明单位时间产出更高，但要结合 batch size 和服务并发解释。
- VRAM 下降可以释放 batch / context / 并发空间，但如果误差超出约束，不能直接部署。
- 精度误差需要和业务容忍度绑定：离线压缩实验可以更激进，在线服务通常要更保守。
- 这一阶段的产物应该是“收益 + 误差 + 部署条件”，而不是单项指标排行榜。

### Step 4: 输出部署选型报告

最后回到 Step 1 的部署目标，用数据判断量化方案是否值得采用。

- 输出 baseline / quantized 对比表，至少包含 latency、throughput、VRAM、error 和备注。
- 写清楚量化方案、量化粒度、校准数据、运行后端和硬件环境。
- 给出“什么时候用、什么时候别用”的结论。
- 如果本轮方案不可部署，要记录主要阻塞原因和下一轮尝试方向。
- 最终产物应回答：这次量化是否满足部署约束，收益来自哪里，代价是否可接受。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整量化推理与部署选型流程。下面的代码只实现其中最小、可复用的三块：测平均耗时、汇总 baseline / quantized 指标差异、生成部署报告。真实项目中的模型量化、校准、输出误差评估和线上压测，需要基于这三个结果继续补充。


```python
import time

```


```python
def benchmark_fn(fn, warmup=2, iters=5):
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和 latency 对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    # start = ???
    for _ in range(iters):
        fn()
    # total = ???
    # avg_latency_ms = ???
    return avg_latency_ms


def summarize_quantized_result(base_metrics, quant_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / quantized 的核心指标差异
    # 提示: latency / vram / error 越低越好，throughput 越高越好
    # 正数表示 quantized 相比 baseline 有改善，error_delta 除外
    # ==========================================
    # latency_delta = ???
    # throughput_delta = ???
    # vram_delta = ???
    # error_delta = ???

    summary = {
        'latency_delta_ms': round(latency_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'vram_delta_mb': round(vram_delta, 2),
        'error_delta': round(error_delta, 4),
        'latency_improved': latency_delta > 0,
        'throughput_improved': throughput_delta > 0,
        'vram_improved': vram_delta > 0,
        'error_within_budget': error_delta <= quant_metrics['error_budget'],
    }
    return summary


def format_deployment_report(quant_name, summary, recommendation):
    # ==========================================
    # TODO 3: 生成量化部署报告
    # 提示: 把指标变化、误差约束和部署建议放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    # rows = ???
    # conclusion = ???
    return "\n".join([f"量化方案：{quant_name}", header, sep] + rows + [conclusion])

```

### 测试


```python
def test_quantized_project_template():
    try:
        counter = {'n': 0}

        def fn():
            counter['n'] += 1

        avg = benchmark_fn(fn, warmup=0, iters=2)
        assert counter['n'] == 2, "benchmark 应该运行 iters 次"
        assert avg >= 0.0, "平均耗时应该非负"

        baseline = {
            'latency_ms': 100.0,
            'throughput': 80.0,
            'vram_mb': 12000.0,
            'error': 0.0,
        }
        quantized = {
            'latency_ms': 72.0,
            'throughput': 120.0,
            'vram_mb': 7000.0,
            'error': 0.012,
            'error_budget': 0.02,
        }
        summary = summarize_quantized_result(baseline, quantized)

        assert summary['latency_delta_ms'] == 28.0
        assert summary['throughput_delta'] == 40.0
        assert summary['vram_delta_mb'] == 5000.0
        assert summary['error_delta'] == 0.012
        assert summary['latency_improved'] is True
        assert summary['throughput_improved'] is True
        assert summary['vram_improved'] is True
        assert summary['error_within_budget'] is True

        report = format_deployment_report('W8A16', summary, '满足误差预算，可进入更大样本回归测试')
        assert 'W8A16' in report
        assert '| 指标 | 变化 | 判断 |' in report
        assert '更大样本回归测试' in report

        print("✅ 量化推理与部署项目模板代码通过基础校验。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_quantized_project_template()

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
    # 返回单位统一为 ms，方便和 latency 对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    avg_latency_ms = total / iters * 1000
    return avg_latency_ms


def summarize_quantized_result(base_metrics, quant_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / quantized 的核心指标差异
    # 提示: latency / vram / error 越低越好，throughput 越高越好
    # 正数表示 quantized 相比 baseline 有改善，error_delta 除外
    # ==========================================
    latency_delta = base_metrics['latency_ms'] - quant_metrics['latency_ms']
    throughput_delta = quant_metrics['throughput'] - base_metrics['throughput']
    vram_delta = base_metrics['vram_mb'] - quant_metrics['vram_mb']
    error_delta = quant_metrics['error'] - base_metrics['error']

    summary = {
        'latency_delta_ms': round(latency_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'vram_delta_mb': round(vram_delta, 2),
        'error_delta': round(error_delta, 4),
        'latency_improved': latency_delta > 0,
        'throughput_improved': throughput_delta > 0,
        'vram_improved': vram_delta > 0,
        'error_within_budget': error_delta <= quant_metrics['error_budget'],
    }
    return summary


def format_deployment_report(quant_name, summary, recommendation):
    # ==========================================
    # TODO 3: 生成量化部署报告
    # 提示: 把指标变化、误差约束和部署建议放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| latency | {summary['latency_delta_ms']} ms | {'改善' if summary['latency_improved'] else '未改善'} |",
        f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
        f"| VRAM | {summary['vram_delta_mb']} MB | {'改善' if summary['vram_improved'] else '未改善'} |",
        f"| error | {summary['error_delta']} | {'满足预算' if summary['error_within_budget'] else '超出预算'} |",
    ]
    conclusion = f"部署建议：{recommendation}。"
    return "\n".join([f"量化方案：{quant_name}", header, sep] + rows + [conclusion])

```

### 解析

- **这一题要解决什么**：把量化推理部署压缩成一个最小选型模板，帮助比较 baseline 和 quantized 的速度、显存和误差。
- **为什么这样做**：量化部署不是只看模型文件变小，必须同时检查 latency、throughput、VRAM 和误差是否满足约束。
- **带走的直觉**：低 bit 只是手段，是否可部署取决于收益是否覆盖误差和工程复杂度。

**1. TODO 1 (benchmark_fn)**

- **warmup**：先运行若干轮，不计入统计，减少首次运行、缓存和调度抖动影响。
- **平均延迟**：正式测量阶段统计 `iters` 轮，再转换成单次平均 latency。
- **单位统一**：返回 ms，方便和部署报告里的 latency 字段直接对应。
- **真实推理场景**：如果使用 GPU，还需要同步 CUDA kernel；如果使用服务端压测，还要区分端到端延迟和模型内部延迟。

**2. TODO 2 (summarize_quantized_result)**

- **latency 差值**：`baseline - quantized`，正数表示量化后单次请求更快。
- **throughput 差值**：`quantized - baseline`，正数表示量化后单位时间产出更高。
- **VRAM 差值**：`baseline - quantized`，正数表示量化后显存更省。
- **error 差值**：`quantized - baseline`，通常越小越好；它不应该被解释成性能改善，而是部署约束。
- **误差预算**：`error_within_budget` 用来判断量化误差是否仍在可接受范围内。

**3. TODO 3 (format_deployment_report)**

- **方案名**：报告必须写清是 W8A16、INT8、NF4/4-bit 还是其他方案。
- **对比表**：把 latency、throughput、VRAM 和 error 放到同一张表中，避免只看显存收益。
- **部署建议**：最后用一句话说明是否进入更大样本回归、是否需要重新校准，或是否暂缓部署。

**量化部署项目原则**

- **先固定输入和评测口径**：prompt、batch、seq len、解码策略和硬件环境必须一致。
- **误差和性能一起看**：显存下降但误差超预算，不能直接上线。
- **区分存储和计算**：Weight-only 量化主要省权重读取和显存，不一定让所有计算都变成 INT8。
- **结论要能落地**：最终输出应回答“这套量化方案是否满足部署约束，下一轮要扩大测试还是调整量化配置”。
