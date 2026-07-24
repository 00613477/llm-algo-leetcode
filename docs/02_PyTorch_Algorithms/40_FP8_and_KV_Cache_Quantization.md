# 40. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化
**难度：** Hard | **环境：** GPU required | **标签：** `量化`, `FP8`, `KV Cache` | **目标人群：** 推理部署与系统工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/40_FP8_and_KV_Cache_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

第 39 节关注的是权重量化：把模型参数压得更小，降低加载和访存成本。但推理阶段的压力不只来自权重。长上下文生成时，KV Cache 会随着序列长度和并发请求持续增长；同时，部分激活或中间张量也会带来带宽压力。只压权重，不能完全解决长上下文推理的显存和带宽瓶颈。

本节用一个极简 `FP8KVCacheSim` 模拟两类推理量化：用对称低精度量化近似 FP8 张量，用分组 scale 量化 KV Cache。学完后，你应该能看清“量化值、scale、反量化、误差检查”这条闭环，以及为什么 KV Cache 通常需要按最后一维分组处理。

**关键词：** `FP8`, `KV cache quantization`, `deployment`

---

## 前置阅读

**导语：** 先看权重量化、PagedAttention 和显存模型，再看 FP8 与 KV Cache 量化会更容易。

- [39. GPTQ and AWQ Weight Quantization | GPTQ 与 AWQ 权重量化](../02_PyTorch_Algorithms/39_GPTQ_and_AWQ_Weight_Quantization.md)
- [25. Quantization W8A16 | W8A16 量化](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)

## 相关阅读

**导语：** FP8 与 KV Cache 量化之后，可以继续看 KV cache 调度和通信 profiling。

- [41. KV Cache Scheduling | KV Cache 调度](../02_PyTorch_Algorithms/41_KV_Cache_Scheduling.md)
- [42. Communication Profiling with NCCL | NCCL 通信性能剖析](../02_PyTorch_Algorithms/42_Communication_Profiling_with_NCCL.md)
- [P1: 14. FlashAttention Memory Model | FlashAttention 显存模型](../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)

### Step 1: 原理与痛点

> **为什么推理阶段还要关心 KV Cache 量化？**
>
> 因为生成式推理不是只跑一次前向。每生成一个 token，模型都会把新的 Key / Value 写入缓存；上下文越长、并发越高，KV Cache 占用越大。对于长上下文服务，KV Cache 往往会成为显存容量和带宽压力的重要来源。

FP8 和 KV Cache 量化解决的是推理过程中的不同对象：FP8 更常用于降低部分张量的计算/带宽成本，KV Cache 量化则直接压缩长上下文缓存。两者的共同点是都需要保存 scale，并在计算前恢复到可用的浮点近似值。

需要注意的是，本节并不复现真实硬件 FP8 格式（如 E4M3/E5M2），而是用对称 INT8 容器模拟“低精度浮点近似”的核心链路：先缩放、再取整、再用 scale 恢复。这样可以把教学重点放在量化闭环，而不是硬件编码细节上。

### Step 2: 代码实现框架

下面的代码会实现一个最小 `FP8KVCacheSim`。它包含两条链路：一条用于普通推理张量的 FP8 近似量化，另一条用于 KV Cache 的分组量化。

代码拆成七个关键动作：

| 动作 | 对应方法 / 变量 | 作用 |
|------|------------------|------|
| 对称量化 | `_sym_quantize` | 计算 absmax、scale，并把张量映射成 int8 |
| 对称反量化 | `_sym_dequantize` | 用 scale 把整数张量恢复成浮点近似值 |
| FP8 记录 | `quantize_fp8` | 保存 FP8 近似量化值、scale 和原始形状 |
| KV 分组 | `quantize_kv_cache` | 沿最后一维按 `kv_group_size` 切块量化 |
| KV 恢复 | `dequantize_kv_cache` | 使用每个 group 的 scale 恢复 KV Cache |
| 实验记录 | `fit` / `forward` | 跑通量化、恢复和前向返回 |
| 误差检查 | `mse` | 衡量原始张量和恢复张量之间的重构误差 |

这里最重要的是区分“全局 scale”和“分组 scale”。普通 hidden states 可以用一个全局 scale 做教学模拟；KV Cache 的最后一维跨度更大，因此按 group 保存 scale 更合理。

### Step 3: 核心机制

对称量化的基本公式是：

$$
scale = \frac{q_{max}}{\max(|X|)}
$$

量化时：

$$
Q = \mathrm{clamp}(\mathrm{round}(X \cdot scale), -q_{max}, q_{max})
$$

反量化时：

$$
\hat{X} = \frac{Q}{scale}
$$

KV Cache 分组量化只是把这个过程应用到多个小块上。假设最后一维被切成若干组，那么每一组都有自己的 $scale_g$。这样能避免一个极端值把整条 hidden dimension 的量化范围拉大，从而降低普通位置的精度损失。

### Step 4: 动手实战

**要求**：请补全下方 `FP8KVCacheSim`，跑通“对称量化 -> 反量化 -> KV 分组量化 -> KV 恢复 -> 误差检查”这条链路。你需要重点完成七个位置：absmax、scale、量化值、FP8 状态记录、KV group 数、KV group 恢复和 MSE 误差。

完成后观察测试结果：`fp8_q` 和 `kv_q` 应该使用 int8 容器保存低精度值，`fp8_scale` 和 `kv_scale` 负责恢复数值范围，恢复后的 hidden states 和 KV Cache 形状应与原始输入一致。


```python
import torch
import torch.nn as nn

```


```python
class FP8KVCacheSim(nn.Module):
    """极简版 FP8 与 KV Cache 量化模拟器。"""

    def __init__(self, fp8_qmax: int = 127, kv_group_size: int = 64, eps: float = 1e-8):
        super().__init__()
        if kv_group_size <= 0:
            raise ValueError("kv_group_size must be positive")
        self.fp8_qmax = fp8_qmax
        self.kv_group_size = kv_group_size
        self.eps = eps

        self.register_buffer("fp8_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("fp8_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("kv_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("kv_scale", torch.empty(0), persistent=False)
        self.fp8_shape = None
        self.kv_shape = None

    def _sym_quantize(self, x: torch.Tensor, qmax: int):
        x = x.detach().float()
        # ==========================================
        # TODO 1: 计算张量绝对最大值
        # 提示: 对 abs(x) 取最大值
        # ==========================================
        # absmax = ???
        absmax = TODO_ABSMAX
        # ==========================================
        # TODO 2: 计算对称量化 scale
        # 提示: scale = qmax / absmax，并用 eps 避免除零
        # ==========================================
        # scale = ???
        scale = TODO_SCALE
        # ==========================================
        # TODO 3: 完成 round + clamp + int8 转换
        # 提示: 先 x * scale，再 round，最后 clamp 到 [-qmax, qmax]
        # ==========================================
        # q = ???
        q = TODO_Q
        return q, scale

    def _sym_dequantize(self, q: torch.Tensor, scale: torch.Tensor):
        return q.to(scale.dtype) / scale.clamp_min(self.eps)

    def quantize_fp8(self, x: torch.Tensor):
        q, scale = self._sym_quantize(x, self.fp8_qmax)
        self.fp8_q = q
        self.fp8_scale = scale
        # ==========================================
        # TODO 4: 记录 FP8 近似张量的原始形状
        # 提示: 后续恢复或检查时需要知道输入 shape
        # ==========================================
        # self.fp8_shape = ???
        self.fp8_shape = TODO_FP8_SHAPE
        return q, scale

    def dequantize_fp8(self):
        if self.fp8_shape is None:
            raise RuntimeError("Call quantize_fp8() before dequantize_fp8().")
        return self._sym_dequantize(self.fp8_q, self.fp8_scale)

    def quantize_kv_cache(self, kv_cache: torch.Tensor):
        kv = kv_cache.detach().float()
        if kv.ndim < 2:
            raise ValueError("KV cache should have at least 2 dimensions.")

        last_dim = kv.size(-1)
        # ==========================================
        # TODO 5: 计算 KV cache 最后一维需要切成多少组
        # 提示: 使用向上取整，最后一组可以不足 kv_group_size
        # ==========================================
        # n_groups = ???
        n_groups = TODO_N_GROUPS
        qkv = torch.zeros_like(kv, dtype=torch.int8)
        scales = torch.zeros(kv.shape[:-1] + (n_groups,), dtype=kv.dtype, device=kv.device)

        flat = kv.reshape(-1, last_dim)
        flat_q = qkv.reshape(-1, last_dim)
        flat_scale = scales.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                chunk = flat[row, start:end]
                if chunk.numel() == 0:
                    continue
                q, scale = self._sym_quantize(chunk, self.fp8_qmax)
                flat_q[row, start:end] = q
                flat_scale[row, g] = scale

        self.kv_q = qkv
        self.kv_scale = scales
        self.kv_shape = tuple(kv.shape)
        return qkv, scales

    def dequantize_kv_cache(self):
        if self.kv_shape is None:
            raise RuntimeError("Call quantize_kv_cache() before dequantize_kv_cache().")

        kv = self.kv_q.to(self.kv_scale.dtype)
        last_dim = kv.size(-1)
        n_groups = self.kv_scale.size(-1)
        flat = kv.reshape(-1, last_dim)
        flat_out = torch.zeros_like(flat, dtype=self.kv_scale.dtype)
        flat_scale = self.kv_scale.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                scale = flat_scale[row, g]
                # ==========================================
                # TODO 6: 恢复当前 KV group 的浮点近似值
                # 提示: 复用 _sym_dequantize，并写回 flat_out 的对应区间
                # ==========================================
                # restored_chunk = ???
                restored_chunk = TODO_RESTORE
                flat_out[row, start:end] = restored_chunk

        return flat_out.reshape(self.kv_shape)

    def fit(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        self.quantize_fp8(hidden_states)
        if kv_cache is not None:
            self.quantize_kv_cache(kv_cache)
        return self

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        fp8_q, fp8_scale = self._sym_quantize(hidden_states, self.fp8_qmax)
        fp8_restored = self._sym_dequantize(fp8_q, fp8_scale)

        if kv_cache is None:
            return fp8_restored

        self.quantize_kv_cache(kv_cache)
        kv_restored = self.dequantize_kv_cache()
        return fp8_restored, kv_restored

    def mse(self, original: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 7: 计算原始张量与恢复张量之间的均方误差
        # 提示: 先转 float，相减平方，再求平均
        # ==========================================
        # error = ???
        error = TODO_MSE
        return error

```


```python
# 测试你的实现
def test_fp8_kv_cache_quantization():
    try:
        torch.manual_seed(0)
        sim = FP8KVCacheSim(fp8_qmax=127, kv_group_size=4)
        hidden = torch.randn(2, 8)
        kv = torch.randn(2, 3, 8)

        sim.fit(hidden, kv)
        hidden_restore = sim.dequantize_fp8()
        kv_restore = sim.dequantize_kv_cache()
        out_hidden, out_kv = sim.forward(hidden, kv)

        assert sim.fp8_q.dtype == torch.int8
        assert sim.kv_q.dtype == torch.int8
        assert sim.fp8_shape == tuple(hidden.shape)
        assert sim.kv_shape == tuple(kv.shape)
        assert sim.kv_scale.shape == (2, 3, 2)
        assert hidden_restore.shape == hidden.shape
        assert kv_restore.shape == kv.shape
        assert out_hidden.shape == hidden.shape
        assert out_kv.shape == kv.shape
        assert float(sim.mse(hidden, hidden_restore)) >= 0.0

        print('✅ FP8KVCacheSim 测试通过')
    except NotImplementedError as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except (AttributeError, NameError, TypeError, ValueError, RuntimeError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_fp8_kv_cache_quantization()

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

class FP8KVCacheSim(nn.Module):
    """极简版 FP8 与 KV Cache 量化模拟器。"""

    def __init__(self, fp8_qmax: int = 127, kv_group_size: int = 64, eps: float = 1e-8):
        super().__init__()
        if kv_group_size <= 0:
            raise ValueError("kv_group_size must be positive")
        self.fp8_qmax = fp8_qmax
        self.kv_group_size = kv_group_size
        self.eps = eps

        self.register_buffer("fp8_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("fp8_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("kv_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("kv_scale", torch.empty(0), persistent=False)
        self.fp8_shape = None
        self.kv_shape = None

    def _sym_quantize(self, x: torch.Tensor, qmax: int):
        x = x.detach().float()
        # ==========================================
        # TODO 1: 计算张量绝对最大值
        # 提示: 对 abs(x) 取最大值
        # ==========================================
        # absmax = ???
        absmax = torch.max(torch.abs(x))
        # ==========================================
        # TODO 2: 计算对称量化 scale
        # 提示: scale = qmax / absmax，并用 eps 避免除零
        # ==========================================
        # scale = ???
        scale = qmax / absmax.clamp_min(self.eps)
        # ==========================================
        # TODO 3: 完成 round + clamp + int8 转换
        # 提示: 先 x * scale，再 round，最后 clamp 到 [-qmax, qmax]
        # ==========================================
        # q = ???
        q = torch.clamp(torch.round(x * scale), -qmax, qmax).to(torch.int8)
        return q, scale

    def _sym_dequantize(self, q: torch.Tensor, scale: torch.Tensor):
        return q.to(scale.dtype) / scale.clamp_min(self.eps)

    def quantize_fp8(self, x: torch.Tensor):
        q, scale = self._sym_quantize(x, self.fp8_qmax)
        self.fp8_q = q
        self.fp8_scale = scale
        # ==========================================
        # TODO 4: 记录 FP8 近似张量的原始形状
        # 提示: 后续恢复或检查时需要知道输入 shape
        # ==========================================
        # self.fp8_shape = ???
        self.fp8_shape = tuple(x.shape)
        return q, scale

    def dequantize_fp8(self):
        if self.fp8_shape is None:
            raise RuntimeError("Call quantize_fp8() before dequantize_fp8().")
        return self._sym_dequantize(self.fp8_q, self.fp8_scale)

    def quantize_kv_cache(self, kv_cache: torch.Tensor):
        kv = kv_cache.detach().float()
        if kv.ndim < 2:
            raise ValueError("KV cache should have at least 2 dimensions.")

        last_dim = kv.size(-1)
        # ==========================================
        # TODO 5: 计算 KV cache 最后一维需要切成多少组
        # 提示: 使用向上取整，最后一组可以不足 kv_group_size
        # ==========================================
        # n_groups = ???
        n_groups = (last_dim + self.kv_group_size - 1) // self.kv_group_size
        qkv = torch.zeros_like(kv, dtype=torch.int8)
        scales = torch.zeros(kv.shape[:-1] + (n_groups,), dtype=kv.dtype, device=kv.device)

        flat = kv.reshape(-1, last_dim)
        flat_q = qkv.reshape(-1, last_dim)
        flat_scale = scales.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                chunk = flat[row, start:end]
                if chunk.numel() == 0:
                    continue
                q, scale = self._sym_quantize(chunk, self.fp8_qmax)
                flat_q[row, start:end] = q
                flat_scale[row, g] = scale

        self.kv_q = qkv
        self.kv_scale = scales
        self.kv_shape = tuple(kv.shape)
        return qkv, scales

    def dequantize_kv_cache(self):
        if self.kv_shape is None:
            raise RuntimeError("Call quantize_kv_cache() before dequantize_kv_cache().")

        kv = self.kv_q.to(self.kv_scale.dtype)
        last_dim = kv.size(-1)
        n_groups = self.kv_scale.size(-1)
        flat = kv.reshape(-1, last_dim)
        flat_out = torch.zeros_like(flat, dtype=self.kv_scale.dtype)
        flat_scale = self.kv_scale.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                scale = flat_scale[row, g]
                # ==========================================
                # TODO 6: 恢复当前 KV group 的浮点近似值
                # 提示: 复用 _sym_dequantize，并写回 flat_out 的对应区间
                # ==========================================
                # restored_chunk = ???
                restored_chunk = self._sym_dequantize(flat[row, start:end], scale)
                flat_out[row, start:end] = restored_chunk

        return flat_out.reshape(self.kv_shape)

    def fit(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        self.quantize_fp8(hidden_states)
        if kv_cache is not None:
            self.quantize_kv_cache(kv_cache)
        return self

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        fp8_q, fp8_scale = self._sym_quantize(hidden_states, self.fp8_qmax)
        fp8_restored = self._sym_dequantize(fp8_q, fp8_scale)

        if kv_cache is None:
            return fp8_restored

        self.quantize_kv_cache(kv_cache)
        kv_restored = self.dequantize_kv_cache()
        return fp8_restored, kv_restored

    def mse(self, original: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 7: 计算原始张量与恢复张量之间的均方误差
        # 提示: 先转 float，相减平方，再求平均
        # ==========================================
        # error = ???
        error = torch.mean((original.float() - restored.float()) ** 2)
        return error

```

### 解析

**1. TODO 1: 计算绝对最大值**
- **实现方式**：`absmax = torch.max(torch.abs(x))`
- **关键点**：对称量化需要先确定张量的动态范围，绝对最大值决定正负两侧的缩放尺度
- **技术细节**：后续会用 `clamp_min(self.eps)` 避免全零张量导致除零

**2. TODO 2: 计算量化 scale**
- **实现方式**：`scale = qmax / absmax.clamp_min(self.eps)`
- **关键点**：这里的 scale 表示浮点值乘以多少后能映射到整数区间
- **技术细节**：本节采用 `q = round(x * scale)` 和 `x_hat = q / scale` 这一组互逆写法

**3. TODO 3: 生成 int8 量化值**
- **实现方式**：`q = torch.clamp(torch.round(x * scale), -qmax, qmax).to(torch.int8)`
- **关键点**：`round` 产生整数近似，`clamp` 防止越界，`int8` 容器保存低精度值
- **技术细节**：这里是教学模拟，不等价于真实硬件 FP8 编码，但保留了低精度存储和 scale 恢复的核心链路

**4. TODO 4: 记录 FP8 张量形状**
- **实现方式**：`self.fp8_shape = tuple(x.shape)`
- **关键点**：量化状态不仅包括整数值和 scale，也包括原始张量的结构信息
- **技术细节**：本节的 FP8 近似使用全局 scale，因此 shape 主要用于状态检查和教学可读性

**5. TODO 5: 计算 KV 分组数量**
- **实现方式**：`n_groups = (last_dim + self.kv_group_size - 1) // self.kv_group_size`
- **关键点**：KV Cache 沿最后一维分组，最后一组可以不足 `kv_group_size`
- **技术细节**：分组 scale 可以降低局部极端值对整条 hidden dimension 的影响

**6. TODO 6: 恢复 KV group**
- **实现方式**：`restored_chunk = self._sym_dequantize(flat[row, start:end], scale)`
- **关键点**：每个 group 必须使用自己的 scale 恢复，不能混用其他 group 的 scale
- **技术细节**：代码先把 KV Cache flatten 成二维，再按最后一维切 group，最后 reshape 回原始形状

**7. TODO 7: 计算重构误差**
- **实现方式**：`error = torch.mean((original.float() - restored.float()) ** 2)`
- **关键点**：MSE 用来衡量量化恢复后的数值偏差
- **技术细节**：误差越小不一定代表端到端效果越好，真实部署仍需要结合任务指标和吞吐/显存收益评估

**FP8 与 KV Cache 量化核心机制**
- **FP8 近似**：用低精度值和 scale 保存张量，降低带宽和存储压力
- **KV Cache 分组**：对最后一维分组保存 scale，使长上下文缓存可以更细粒度地压缩和恢复
- **量化闭环**：任何推理量化都要同时记录低精度值、scale、shape 和恢复误差

**工程优化要点**
- **硬件格式**：真实 FP8 通常涉及 E4M3 / E5M2、Tensor Core 支持和 kernel 路径，本节只模拟核心思想
- **缓存收益**：KV Cache 量化对长上下文和高并发更有价值，因为缓存大小会随序列长度线性增长
- **精度边界**：KV Cache 参与后续 attention，过度压缩可能影响生成质量，需要结合 perplexity、任务指标和在线效果验证
