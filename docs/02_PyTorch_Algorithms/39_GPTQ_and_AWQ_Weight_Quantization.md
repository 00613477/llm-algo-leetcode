# 39. GPTQ and AWQ Weight Quantization | GPTQ 与 AWQ 权重量化
**难度：** Hard | **环境：** GPU required | **标签：** `量化`, `GPTQ`, `AWQ` | **目标人群：** 模型压缩与部署工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/39_GPTQ_and_AWQ_Weight_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

第 25 节和第 26 节已经把量化的两条主线铺开：W8A16 说明了 weight-only 量化如何减少权重读取压力，QLoRA 说明了 4-bit 权重如何服务于低成本微调。但部署阶段还会遇到一个更细的问题：同样是把权重压到低比特，哪些权重更敏感，哪些误差可以接受，校准数据又应该如何参与量化决策？

本节用一个极简 `WeightQuantizerSim` 模拟 GPTQ / AWQ 的核心直觉：GPTQ 更关注校准后的重构误差，AWQ 更强调激活感知和敏感通道保护。学完后，你应该能看清“校准 -> 分组 -> 量化 -> 保护 -> 反量化 -> 误差检查”这条权重量化链路。

**关键词：** `GPTQ`, `AWQ`, `weight quantization`

---

## 前置阅读

**导语：** 先把 W8A16、QLoRA 和量化理论理顺，再看 GPTQ / AWQ 会更容易。

- [25. Quantization W8A16 | W8A16 量化](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- [P1: 01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)

## 相关阅读

**导语：** 权重量化之后，可以继续看 FP8、KV Cache Quantization 和 cache scheduling。

- [40. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化](../02_PyTorch_Algorithms/40_FP8_and_KV_Cache_Quantization.md)
- [41. KV Cache Scheduling | KV Cache 调度](../02_PyTorch_Algorithms/41_KV_Cache_Scheduling.md)
- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)

---
### Step 1: 原理与痛点

> **为什么不能只把 W8A16 继续压到 4-bit？**
>
> 因为 bit 数降低以后，量化误差会明显放大。8-bit 量化通常还有比较宽的表示空间，但 4-bit 只有 16 个离散状态，如果仍然对所有权重一视同仁地量化，少数敏感通道的误差就可能被放大到影响模型输出。

GPTQ 和 AWQ 都属于面向部署的后训练量化思路。它们不重新训练完整模型，而是利用校准数据判断权重和激活的统计特性，再决定 scale、分组方式和误差处理策略。

两者的直觉可以这样区分：

- **GPTQ**：更关注量化后如何让层输出重构误差尽量小；
- **AWQ**：更关注哪些通道被激活放大、对输出更敏感，因此需要更保守地处理；
- **共同点**：都不是简单按权重绝对值压缩，而是让校准信息参与量化决策。

本节不会复现真实 GPTQ 的 Hessian 近似或 AWQ 的完整搜索流程，而是保留教学主线：用激活统计构造通道重要性，再用分组 scale 和敏感通道保护模拟它们的核心差异。

### Step 2: 代码实现框架

下面的代码会实现一个最小 `WeightQuantizerSim`。输入是一层 Linear 的二维权重矩阵 `weight`，可选输入是一批校准激活 `activations`。代码会把权重量化拆成六个动作：

| 动作 | 对应方法 / 变量 | 作用 |
|------|------------------|------|
| 统计重要性 | `_collect_importance` | 根据校准激活估计每个输入通道的重要程度 |
| 分组 | `group_size` / `n_groups` | 每组单独计算 scale，避免全局 scale 被极端值支配 |
| 敏感通道保护 | `protected_mask` | AWQ 模式下保留少量高重要性通道的原始权重 |
| 量化 | `qweight` / `scales` | 把普通通道映射到低比特整数表示 |
| 反量化 | `dequantize` | 用 scale 把整数权重恢复成近似浮点权重 |
| 误差检查 | `mse` | 对比原始权重和恢复权重的重构误差 |

这个实现故意把“量化后的权重”和“被保护的权重”分开保存。这样读者可以清楚看到：普通通道走低比特量化，敏感通道则通过 mask 恢复为原始浮点值。

### Step 3: 核心机制

分组对称量化的核心公式仍然很简单。对某一组权重 $W_g$，先计算 scale：

$$
scale_g = \frac{\max(|W_g|)}{q_{max}}
$$

再量化为整数：

$$
Q_g = \mathrm{clamp}\left(\mathrm{round}\left(\frac{W_g}{scale_g}\right), -q_{max}, q_{max}\right)
$$

反量化时再恢复为：

$$
\hat{W}_g = Q_g \cdot scale_g
$$

AWQ 的额外直觉是：不是所有通道都同等重要。如果校准激活显示某些输入通道经常被放大，那么这些通道对应的权重误差更容易影响输出。本节用 `importance` 选择每组内的 top-k 敏感通道，并用 `protected_mask` 让这些通道保留原始浮点权重。

### Step 4: 动手实战

**要求**：请补全下方 `WeightQuantizerSim`，跑通“校准 -> 分组 -> 量化 -> 保护 -> 反量化 -> 误差检查”这条链路。你需要重点完成六个位置：激活重要性统计、分组数量、敏感通道 mask、分组 scale、反量化恢复，以及 MSE 误差计算。

完成后观察测试结果：`qweight` 应该是 INT8 容器里的低比特整数，`scales` 应该按输出通道和分组保存，AWQ 模式下 `protected_mask` 应该保护至少一部分敏感通道。只要恢复权重形状正确、前向输出形状正确、误差为非负，就说明最小权重量化闭环已经跑通。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

```


```python
class WeightQuantizerSim(nn.Module):
    """极简版 GPTQ / AWQ 权重量化模拟器。"""

    def __init__(self, bits: int = 4, group_size: int = 32, method: str = "gptq", protect_ratio: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if bits < 2:
            raise ValueError("bits must be >= 2")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.bits = bits
        self.group_size = group_size
        self.method = method.lower()
        self.protect_ratio = protect_ratio
        self.eps = eps
        self.qmax = 2 ** (bits - 1) - 1

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("scales", torch.empty(0), persistent=False)
        self.register_buffer("protected_weight", torch.empty(0), persistent=False)
        self.register_buffer("protected_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("importance", torch.empty(0), persistent=False)
        self.weight_shape = None

    def _collect_importance(self, activations: torch.Tensor, in_features: int) -> torch.Tensor:
        act = activations.detach().float()
        if act.ndim == 1:
            importance = act.abs()
        else:
            reduce_dims = tuple(range(act.ndim - 1))
            # ==========================================
            # TODO 1: 根据校准激活统计输入通道重要性
            # 提示: 对除最后一维外的维度求 RMS，最后一维对应 in_features
            # ==========================================
            # importance = ???
        if importance.numel() != in_features:
            raise ValueError(f"Calibration importance dim mismatch: expected {in_features}, got {importance.numel()}")
        return importance

    def fit(self, weight: torch.Tensor, activations: torch.Tensor | None = None) -> "WeightQuantizerSim":
        w = weight.detach().float()
        if w.ndim != 2:
            raise ValueError("WeightQuantizerSim only supports 2D linear weights.")

        out_features, in_features = w.shape
        self.weight_shape = (out_features, in_features)
        importance = torch.ones(in_features, device=w.device, dtype=w.dtype) if activations is None else self._collect_importance(activations, in_features)
        self.importance = importance

        # ==========================================
        # TODO 2: 计算输入维度需要被切成多少个 group
        # 提示: 使用向上取整，最后一组可以不足 group_size
        # ==========================================
        # n_groups = ???
        qweight = torch.zeros_like(w, dtype=torch.int8)
        scales = torch.zeros((out_features, n_groups), dtype=w.dtype, device=w.device)
        protected_weight = torch.zeros_like(w)
        protected_mask = torch.zeros_like(w, dtype=torch.bool)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                wg = w[row, start:end]
                ig = importance[start:end]
                if wg.numel() == 0:
                    continue

                mask = torch.zeros_like(ig, dtype=torch.bool)
                if self.method == "awq":
                    k = max(1, int(round(wg.numel() * self.protect_ratio)))
                    k = min(k, wg.numel())
                    topk = torch.topk(ig, k=k, largest=True).indices
                    # ==========================================
                    # TODO 3: 标记本组中需要保护的敏感通道
                    # 提示: topk 是通道下标，把这些位置在 mask 中置为 True
                    # ==========================================
                    # mask[topk] = ???
                    protected_mask[row, start:end] = mask
                    protected_weight[row, start:end] = wg * mask.to(wg.dtype)

                base = wg[~mask]
                if base.numel() == 0:
                    base = wg
                # ==========================================
                # TODO 4: 为未保护的普通通道计算分组 scale
                # 提示: 对称量化 scale = absmax / qmax，并用 eps 避免除零
                # ==========================================
                # scale = ???

                q_group = torch.zeros_like(wg, dtype=torch.int8)
                q_group[~mask] = torch.clamp(torch.round(wg[~mask] / scale), -self.qmax, self.qmax).to(torch.int8)
                qweight[row, start:end] = q_group
                scales[row, g] = scale

        self.qweight = qweight
        self.scales = scales
        self.protected_weight = protected_weight
        self.protected_mask = protected_mask
        return self

    def dequantize(self) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before dequantize().")

        out_features, in_features = self.weight_shape
        n_groups = self.scales.size(1)
        weight = torch.zeros((out_features, in_features), dtype=self.scales.dtype, device=self.scales.device)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                scale = self.scales[row, g]
                q_group = self.qweight[row, start:end].to(self.scales.dtype)
                # ==========================================
                # TODO 5: 将整数权重反量化回浮点近似值
                # 提示: 量化时除以 scale，恢复时乘回 scale
                # ==========================================
                # dequant = ???
                protected = self.protected_mask[row, start:end]
                if protected.any():
                    dequant = dequant.clone()
                    dequant[protected] = self.protected_weight[row, start:end][protected]
                weight[row, start:end] = dequant

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before forward().")
        weight = self.dequantize().to(x.dtype)
        return F.linear(x, weight)

    def mse(self, weight: torch.Tensor) -> torch.Tensor:
        recon = self.dequantize().to(weight.dtype)
        # ==========================================
        # TODO 6: 计算原始权重和恢复权重之间的均方误差
        # 提示: 先相减、平方，再求平均
        # ==========================================
        # error = ???
        return error

```


```python
# 测试你的实现
def test_weight_quantizer():
    try:
        torch.manual_seed(0)
        weight = torch.randn(4, 8)
        acts = torch.randn(16, 8)
        sim = WeightQuantizerSim(bits=4, group_size=4, method="awq", protect_ratio=0.25).fit(weight, acts)
        restored = sim.dequantize()
        y = sim.forward(torch.randn(2, 8))

        assert sim.qweight.dtype == torch.int8
        assert sim.scales.shape == (4, 2)
        assert sim.importance.shape == (8,)
        assert sim.protected_mask.any()
        assert restored.shape == weight.shape
        assert y.shape == (2, 4)
        assert float(sim.mse(weight)) >= 0.0

        gptq = WeightQuantizerSim(bits=4, group_size=4, method="gptq").fit(weight, acts)
        assert not gptq.protected_mask.any()
        assert gptq.dequantize().shape == weight.shape

        print("✅ WeightQuantizerSim 测试通过")
    except NotImplementedError as e:
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except (AttributeError, NameError, TypeError, ValueError, RuntimeError, AssertionError) as e:
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_weight_quantizer()

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

class WeightQuantizerSim(nn.Module):
    """极简版 GPTQ / AWQ 权重量化模拟器。"""

    def __init__(self, bits: int = 4, group_size: int = 32, method: str = "gptq", protect_ratio: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if bits < 2:
            raise ValueError("bits must be >= 2")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.bits = bits
        self.group_size = group_size
        self.method = method.lower()
        self.protect_ratio = protect_ratio
        self.eps = eps
        self.qmax = 2 ** (bits - 1) - 1

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("scales", torch.empty(0), persistent=False)
        self.register_buffer("protected_weight", torch.empty(0), persistent=False)
        self.register_buffer("protected_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("importance", torch.empty(0), persistent=False)
        self.weight_shape = None

    def _collect_importance(self, activations: torch.Tensor, in_features: int) -> torch.Tensor:
        act = activations.detach().float()
        if act.ndim == 1:
            importance = act.abs()
        else:
            reduce_dims = tuple(range(act.ndim - 1))
            # ==========================================
            # TODO 1: 根据校准激活统计输入通道重要性
            # 提示: 对除最后一维外的维度求 RMS，最后一维对应 in_features
            # ==========================================
            importance = act.pow(2).mean(dim=reduce_dims).sqrt()
        if importance.numel() != in_features:
            raise ValueError(f"Calibration importance dim mismatch: expected {in_features}, got {importance.numel()}")
        return importance

    def fit(self, weight: torch.Tensor, activations: torch.Tensor | None = None) -> "WeightQuantizerSim":
        w = weight.detach().float()
        if w.ndim != 2:
            raise ValueError("WeightQuantizerSim only supports 2D linear weights.")

        out_features, in_features = w.shape
        self.weight_shape = (out_features, in_features)
        importance = torch.ones(in_features, device=w.device, dtype=w.dtype) if activations is None else self._collect_importance(activations, in_features)
        self.importance = importance

        # ==========================================
        # TODO 2: 计算输入维度需要被切成多少个 group
        # 提示: 使用向上取整，最后一组可以不足 group_size
        # ==========================================
        n_groups = (in_features + self.group_size - 1) // self.group_size
        qweight = torch.zeros_like(w, dtype=torch.int8)
        scales = torch.zeros((out_features, n_groups), dtype=w.dtype, device=w.device)
        protected_weight = torch.zeros_like(w)
        protected_mask = torch.zeros_like(w, dtype=torch.bool)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                wg = w[row, start:end]
                ig = importance[start:end]
                if wg.numel() == 0:
                    continue

                mask = torch.zeros_like(ig, dtype=torch.bool)
                if self.method == "awq":
                    k = max(1, int(round(wg.numel() * self.protect_ratio)))
                    k = min(k, wg.numel())
                    topk = torch.topk(ig, k=k, largest=True).indices
                    # ==========================================
                    # TODO 3: 标记本组中需要保护的敏感通道
                    # 提示: topk 是通道下标，把这些位置在 mask 中置为 True
                    # ==========================================
                    mask[topk] = True
                    protected_mask[row, start:end] = mask
                    protected_weight[row, start:end] = wg * mask.to(wg.dtype)

                base = wg[~mask]
                if base.numel() == 0:
                    base = wg
                # ==========================================
                # TODO 4: 为未保护的普通通道计算分组 scale
                # 提示: 对称量化 scale = absmax / qmax，并用 eps 避免除零
                # ==========================================
                scale = (base.abs().max() / self.qmax).clamp_min(self.eps)

                q_group = torch.zeros_like(wg, dtype=torch.int8)
                q_group[~mask] = torch.clamp(torch.round(wg[~mask] / scale), -self.qmax, self.qmax).to(torch.int8)
                qweight[row, start:end] = q_group
                scales[row, g] = scale

        self.qweight = qweight
        self.scales = scales
        self.protected_weight = protected_weight
        self.protected_mask = protected_mask
        return self

    def dequantize(self) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before dequantize().")

        out_features, in_features = self.weight_shape
        n_groups = self.scales.size(1)
        weight = torch.zeros((out_features, in_features), dtype=self.scales.dtype, device=self.scales.device)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                scale = self.scales[row, g]
                q_group = self.qweight[row, start:end].to(self.scales.dtype)
                # ==========================================
                # TODO 5: 将整数权重反量化回浮点近似值
                # 提示: 量化时除以 scale，恢复时乘回 scale
                # ==========================================
                dequant = q_group * scale
                protected = self.protected_mask[row, start:end]
                if protected.any():
                    dequant = dequant.clone()
                    dequant[protected] = self.protected_weight[row, start:end][protected]
                weight[row, start:end] = dequant

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before forward().")
        weight = self.dequantize().to(x.dtype)
        return F.linear(x, weight)

    def mse(self, weight: torch.Tensor) -> torch.Tensor:
        recon = self.dequantize().to(weight.dtype)
        # ==========================================
        # TODO 6: 计算原始权重和恢复权重之间的均方误差
        # 提示: 先相减、平方，再求平均
        # ==========================================
        error = torch.mean((weight.float() - recon.float()) ** 2)
        return error

```

### 解析

**1. TODO 1: 统计通道重要性**
- **实现方式**：`importance = act.pow(2).mean(dim=reduce_dims).sqrt()`
- **关键点**：最后一维对应输入通道，其他维度是 batch 或序列维度，需要被聚合掉
- **技术细节**：这里用 RMS 近似衡量通道激活强度；激活越大的通道，权重误差越容易影响输出

**2. TODO 2: 计算分组数量**
- **实现方式**：`n_groups = (in_features + self.group_size - 1) // self.group_size`
- **关键点**：分组数要向上取整，因为最后一组可能不足 `group_size`
- **技术细节**：分组量化让每组拥有独立 scale，比整层共享一个 scale 更能适应局部数值范围

**3. TODO 3: 标记 AWQ 敏感通道**
- **实现方式**：`mask[topk] = True`
- **关键点**：`topk` 来自本组内 importance 最大的通道，这些位置会被 `protected_mask` 记录
- **技术细节**：本节用“保留原始浮点权重”模拟 AWQ 的敏感通道保护，真实实现通常会采用更细的 scale 搜索和重缩放策略

**4. TODO 4: 计算分组 scale**
- **实现方式**：`scale = (base.abs().max() / self.qmax).clamp_min(self.eps)`
- **关键点**：对称量化用本组绝对最大值确定动态范围，并用 `eps` 避免全零分组除零
- **技术细节**：`qmax = 2 ** (bits - 1) - 1`，4-bit 对称量化时有效正向上限是 7

**5. TODO 5: 反量化恢复权重**
- **实现方式**：`dequant = q_group * scale`
- **关键点**：量化时是 `round(w / scale)`，恢复时就乘回同一个 scale
- **技术细节**：如果当前位置被 `protected_mask` 标记，反量化结果会被原始 `protected_weight` 覆盖

**6. TODO 6: 计算重构误差**
- **实现方式**：`error = torch.mean((weight.float() - recon.float()) ** 2)`
- **关键点**：MSE 用来衡量量化恢复权重和原始权重之间的平均平方偏差
- **技术细节**：这个误差只检查权重重构，不等价于最终模型精度；真实评估还要看校准集或下游任务指标

**GPTQ / AWQ 核心机制**
- **GPTQ 直觉**：利用校准数据估计量化对层输出的影响，让低比特权重尽量维持原始层行为
- **AWQ 直觉**：激活越强的通道越敏感，少量通道需要更保守地量化或直接保护
- **分组量化**：按 group 计算 scale，可以减少极端值对整层量化范围的支配

**工程优化要点**
- **存储收益**：4-bit 权重量化能显著降低模型权重显存和加载带宽
- **元数据成本**：分组越细，scale 越多，精度通常更好，但元数据开销也更大
- **部署实践**：真实 GPTQ / AWQ 还涉及校准集选择、kernel 支持、group size、zero point、packing 格式和端到端精度评估
