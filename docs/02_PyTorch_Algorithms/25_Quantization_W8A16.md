# 25. Quantization W8A16 | W8A16 量化
**难度：** Medium | **环境：** CPU-first | **标签：** `量化`, `推理优化`, `Linear` | **目标人群：** 推理优化与模型压缩

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

模型变大以后，推理时最先遇到的问题往往不是“代码能不能跑”，而是权重太大、显存占用高、每一步都要从显存里读很多数据。前面的推理内容已经把这条压力线铺开了，本节开始进入量化：先不改完整推理框架，只尝试把最稳定的一部分——权重——压小。

这一节会把这个思路落到一个最小 `Linear` 层里：先把一块浮点权重压成 8-bit 存储，再在前向时把它接回普通矩阵乘法。学完后，你应该能看懂 Weight-only 量化为什么能省显存、它和真正低精度计算有什么区别，以及后面的 4-bit / QLoRA 为什么是在这个基础上继续往前走。

**关键词：** `W8A16`, `INT8`, `quantization`

---

## 前置阅读

**导语：** 先补齐精度格式、硬件吞吐和基本量化直觉，再看本节会更顺：这里关注的不是把整条推理链路都改成低精度，而是先解决权重太大、读取太贵的问题。

- [P1: 01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)


## 相关阅读

**导语：** 完成 W8A16 后，可以继续看两件事：一是用 profiling 验证瓶颈有没有变化，二是进入更激进的 4-bit 和部署级量化方案。

- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](./26_QLoRA_and_4bit_Quantization.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [P1: 24. SRAM Optimization Techniques | SRAM 优化技术](../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md)

---

### Step 1: 核心思想与概念

> **什么是量化？**
> 将高精度（如 FP32/FP16，分别占用 4/2 个字节）的浮点数，映射到低精度（如 INT8，占用 1 个字节）的整数上。这样不仅能让显存占用直接缩小至原来的 1/2 到 1/4，还能利用硬件的整数计算单元（如 INT8 Tensor Core）加速计算。

> **为什么本节只做 Weight-only Quantization？**
> 推理时，显存的大头通常来自权重本身。把权重量化到 INT8，就能立刻将静态参数显存压到原来的 1/4。相比之下，激活值往往是动态变化的，是否量化要看具体场景，所以这里先聚焦最稳定、收益最直接的权重量化。

> **PTQ 与 QAT 的区别：**
> - **PTQ (Post-Training Quantization，训练后量化)**：模型已经训练好了。对 Weight-only Quantization 而言，权重的数值范围是确定的，直接对权重计算绝对最大值即可算出缩放因子（Scale），不需要额外的校准数据。若涉及激活值量化，才需要校准数据来统计激活值分布以确定其动态范围。
> - **QAT (Quantization-Aware Training，量化感知训练)**：在训练时，正向传播模拟量化的误差，反向传播用“直通估计器 (STE)”更新原始的高精度权重。成本极高，但精度损失最小。

### Step 2: 代码实现框架

要实现“压缩权重显存、缓解带宽压力。”这个目标，选择 `Absmax` 对称量化 + `W8A16`，是因为它能用最少的实现复杂度，直接把权重显存和读取带宽压下来，同时又不需要改写低精度算子。看两个关键的实现组件：`absmax_quantize` 负责把单个张量转成 INT8，`W8A16Linear` 负责把量化后的权重装进模型并在前向时反量化。

> **组件一：量化函数 `absmax_quantize`**
> 负责将浮点权重按 absmax 对称量化为 INT8，并返回对应的缩放因子。注意边界处理：若 `absmax == 0`（全零张量），将 `absmax` 设为 `1e-8` 以避免除零。
> 量化流程（细节见 Step 3）：
> - `absmax`：找到张量的绝对最大值
> - `scale`：计算缩放因子
> - `round + clamp`：把浮点映射到 INT8
> - `int8`：得到最终存储结果



> **组件二：W8A16 量化线性层 `W8A16Linear`**
> 负责保存 INT8 权重和缩放因子，在前向传播时将权重反量化到与输入一致的 `dtype`，再调用 `F.linear` 完成计算。
> 它和 `absmax_quantize` 的关系很直接：
> - `absmax_quantize` 负责“单个张量怎么量化”
> - `W8A16Linear` 负责“量化后的权重怎么在模型里使用” 。
>
下面进入 Step 3，看这套流程在数学上为什么成立。
### Step 3: 数学公式：绝对最大值量化（Absmax Quantization）

这里给出 `absmax_quantize` 的数学定义。目标是把浮点张量映射到 INT8，并在需要时反量化回浮点域。

**第 1 步：计算绝对最大值**

找到张量中绝对值最大的元素，用于确定动态范围：

$$
m = \max(|X|)
$$

**第 2 步：计算缩放因子**

缩放因子为：

$$
S = \frac{127}{m}
$$

其中 $S$ 是一个标量张量，表示“1 个单位的 INT8 对应多少个单位的浮点数”。若 $m = 0$（全零张量），令 $S = 1$ 以避免除零。

> **关于为什么用 127 而不是 128**：这是对称量化，0 点严格对齐，正负范围对称。`-128` 是 INT8 数据类型的硬边界，在对称量化中通常不被使用。严格对称量化通常写作 `clamp(-127, 127)`；本文代码实现保留 `-128` 作为下限以利用 INT8 的完整表示范围。两者差别极小，实践中均可接受。

**第 3 步：量化（浮点数 → INT8）**

将张量乘以缩放因子，再四舍五入取整，最后截断到 INT8 的可表示范围内：

$$
X_{\text{int8}} = \text{clamp}\bigl(\text{round}(X \cdot S),\; -127,\; 127\bigr)
$$

其中：

- $\text{round}(\cdot)$：四舍五入到最近整数
- $\text{clamp}(\cdot, -127, 127)$：截断到 INT8 范围内

**第 4 步：反量化（INT8 → 浮点数）**

在 W8A16 的前向传播中，需要将 INT8 权重恢复为浮点数才能参与矩阵乘法。反量化就是量化的逆操作：

$$
X_{\text{dequant}} = \frac{X_{\text{int8}}}{S} = X_{\text{int8}} \cdot \frac{m}{127}
$$

> 注意：反量化后的值不会完全等于原始浮点数，量化误差来自 `round` 带来的舍入损失。误差大小取决于张量的数值分布和缩放因子的精度。

### Step 4: 动手实战

**要求**：
1. 补全 `absmax_quantize` 函数，实现权重的 INT8 转换并返回 `scale`。
2. 补全 `W8A16Linear` 的 `forward` 方法。W8A16 意味着权重是 INT8，但激活值保持输入 `dtype`。计算时需要实时反量化。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
def absmax_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将浮点张量 X 量化为 INT8，并返回缩放因子。
    
    Args:
        x: 浮点类型的张量
    Returns:
        x_quant: dtype 为 torch.int8 的量化张量
        scale: 标量张量形式的缩放因子
    """
    # ==========================================
    # TODO 1: 计算张量的绝对最大值 absmax
    # ==========================================
    # absmax = ???
    
    # 防除零保护：全零张量时 absmax 为 0，设为一个非零值避免除以 0
    if absmax == 0:
        absmax = torch.tensor(1.0, device=x.device)  # 保持设备一致性 
        
    # ==========================================
    # TODO 2: 计算缩放因子 scale (对称量化，基于 127 计算)
    # ==========================================
    # scale = ???
    
    # ==========================================
    # TODO 3: 量化过程
    # 1. 乘以 scale
    # 2. round 到最近整数
    # 3. clamp 到 INT8 可表示范围 [-128, 127]
    # ==========================================
    # x_scaled = ???
    # x_quant = ???
    return x_quant, scale

class W8A16Linear(nn.Module):
    """
    Weight-only INT8 量化线性层。
    在内存中，我们存储的是非常微小的 INT8 权重。
    在计算时，我们将权重反量化回与输入一致的浮点类型（如 FP16/FP32），再进行矩阵乘法。
    这种方式虽然没有加速计算，但极大地缓解了从内存读取权重的带宽压力（相比 FP32 可降至 1/4，相比 FP16 可降至 1/2）。
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def from_float(self, linear_layer: nn.Linear):
        """
        从高精度的 Linear 层中吸收权重并进行 PTQ 量化
        """
        with torch.no_grad():
            w_quant, scale = absmax_quantize(linear_layer.weight)
            self.weight_int8.copy_(w_quant)
            self.scale.copy_(scale)
            if linear_layer.bias is not None:
                self.bias.copy_(linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 4: 反量化与前向传播
        # 1. 将 weight_int8 转换回与输入 x 相同的类型 (如 float32/float16)
        # 2. 除以 self.scale 恢复其数值范围
        # 3. 使用 F.linear 进行标准的矩阵乘法
        # ==========================================
        
        # w_fp = ???
        # w_dequant = ???
        
        # out = ???
        return out

```


```python
# 测试你的实现
def test_quantization():
    try:
        torch.manual_seed(42)

        # 1. 测试 absmax_quantize 的基础边界
        zero_q, zero_scale = absmax_quantize(torch.zeros(5))
        assert zero_q.dtype == torch.int8, "量化后的张量必须是 int8 类型！"
        assert torch.count_nonzero(zero_q) == 0, "全 0 张量量化后仍应保持全 0！"
        assert torch.isfinite(torch.as_tensor(zero_scale)).item(), "Scale 不能是 NaN/Inf！"

        # 继续沿用带符号样本验证 scale 和 round 行为
        x_fp = torch.tensor([-0.8, 1.5, -3.0, 2.5, 0.0])
        # 绝对最大值是 3.0。Scale = 127 / 3.0 = 42.333
        # 2.5 * 42.333 = 105.8 -> 106
        x_q, scale = absmax_quantize(x_fp)
        assert x_q.dtype == torch.int8, "量化后的张量必须是 int8 类型！"
        assert torch.allclose(scale, torch.tensor(127.0 / 3.0)), "Scale 计算不正确！"
        assert x_q[3].item() == 106, "量化后的四舍五入数值计算不正确！"
        print("✅ absmax_quantize 核心算法测试通过！")

        # 2. 测试 W8A16 线性层
        in_dim, out_dim = 128, 64
        batch, seq = 2, 10

        fp_linear = nn.Linear(in_dim, out_dim)
        q_linear = W8A16Linear(in_dim, out_dim)
        q_linear.from_float(fp_linear)

        fp_bytes = fp_linear.weight.element_size() * fp_linear.weight.numel()
        q_bytes = q_linear.weight_int8.element_size() * q_linear.weight_int8.numel()
        #assert q_bytes == fp_bytes // 4, "INT8 权重的内存占用必须是 FP32 的四分之一！"
        expected_ratio = fp_linear.weight.element_size() // q_linear.weight_int8.element_size()
        assert q_bytes * expected_ratio == fp_bytes, \
            f"INT8 权重内存占用应为原大小的 1/{expected_ratio}，实际比例为 {fp_bytes / q_bytes:.1f}"

        x_input = torch.randn(batch, seq, in_dim)
        out_fp = fp_linear(x_input)
        out_q = q_linear(x_input)
        cos_sim = F.cosine_similarity(out_fp.flatten(), out_q.flatten(), dim=0)
        # 余弦相似度 > 0.99 表示量化前后输出方向几乎完全一致，精度损失极小
        assert cos_sim > 0.99, f"反量化计算出的张量与原始张量差异过大，相似度仅为: {cos_sim.item():.4f}"

        # 3. 用一个确定性小矩阵，直接验证“量化权重 -> 反量化 -> 线性层”的公式链路
        fp_linear_small = nn.Linear(4, 3)
        with torch.no_grad():
            fp_linear_small.weight.copy_(torch.tensor([
                [1.0, -2.0, 3.0, -4.0],
                [0.5, 0.25, -0.75, 1.5],
                [-1.0, 0.0, 1.0, -2.0],
            ]))
            fp_linear_small.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

        q_linear_small = W8A16Linear(4, 3)
        q_linear_small.from_float(fp_linear_small)
        x_small = torch.tensor([[1.0, -1.0, 0.5, 2.0], [0.0, 1.0, -1.0, 3.0]])
        out_small = q_linear_small(x_small)
        w_dequant = q_linear_small.weight_int8.to(x_small.dtype) / q_linear_small.scale
        out_ref = F.linear(x_small, w_dequant, q_linear_small.bias)
        assert torch.allclose(out_small, out_ref, atol=1e-6), "小矩阵下的反量化前向公式不正确！"

        print(f"✅ W8A16Linear 测试通过！输出相似度极高 (Cosine Sim: {cos_sim.item():.4f})，且权重内存缩小 4 倍。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成导致变量属性错误。")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义。")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误。")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误。")
        elif isinstance(e, AssertionError):
            print(f"❌ 测试失败: {e}")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误。")
        else:
            print("代码可能未完成，导致了断言失败。")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_quantization()

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
def absmax_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将浮点张量 X 量化为 INT8，并返回缩放因子。
    """
    # TODO 1: 计算张量的绝对最大值 absmax
    absmax = torch.max(torch.abs(x))
    
    # 防除零保护：全零张量时 absmax 为 0，设为一个非零值避免除以 0
    if absmax == 0:
        absmax = torch.tensor(1.0, device=x.device)  # 保持设备一致性  
        
    # TODO 2: 计算缩放因子 scale  (对称量化，基于 127 计算)
    scale = 127.0 / absmax
    
    # TODO 3: 量化过程
    x_scaled = x * scale
    x_quant = torch.clamp(torch.round(x_scaled), -128, 127).to(torch.int8)
    
    return x_quant, scale

class W8A16Linear(nn.Module):
    """
    Weight-only INT8 量化线性层。
    在内存中存储 INT8 权重，前向时反量化到与输入一致的浮点类型进行计算。
    这种方式主要缓解从内存读取权重的带宽压力，而非将计算链路改为纯 INT8。
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def from_float(self, linear_layer: nn.Linear):
        """
        从高精度的 Linear 层中吸收权重并进行 PTQ 量化
        """
        with torch.no_grad():
            w_quant, scale = absmax_quantize(linear_layer.weight)
            self.weight_int8.copy_(w_quant)
            self.scale.copy_(scale)
            if linear_layer.bias is not None:
                self.bias.copy_(linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 4: 反量化与前向传播
        # 1. 将 weight_int8 转换回与输入 x 相同的类型
        w_fp = self.weight_int8.to(x.dtype)
        
        # 2. 除以 self.scale 恢复其数值范围
        w_dequant = w_fp / self.scale
        
        # 3. 使用 F.linear 进行标准的矩阵乘法
        out = F.linear(x, w_dequant, self.bias)
        return out
```

### 解析

**1. TODO 1（计算绝对最大值）**
- `absmax = torch.max(torch.abs(x))` 找到张量中最“极端”的值，用它来确定量化动态范围。
- 如果 `absmax` 为 0，需要先做保护，避免除零。

**2. TODO 2（计算缩放因子）**
- `scale = 127.0 / absmax` 将浮点范围映射到 INT8 的对称区间。
- **关于 `-127` 和 `-128`**：数学上对称量化描述为 `[-127, 127]`，因为 0 点严格对齐。代码中 `clamp` 使用 `-128` 只是利用 INT8 数据类型的完整存储范围。由于 `scale` 基于 127 计算，`-128` 这个边界值在实际量化中极少被用到，两者精度差别极小。

**3. TODO 3（量化过程）**
- 先执行 `x_scaled = x * scale`，再 `torch.round`，最后 `torch.clamp` 到可用区间。
- 这一步的本质是把连续浮点数离散化成有限的 INT8 取值。
- `torch.int8` 是最终存储格式，能直接把权重显存压到更低。

**4. TODO 4（反量化与前向传播）**
- 反量化时先把 `weight_int8` 转回与输入一致的数据类型。
- 再除以 `scale` 恢复近似的浮点值范围。
- 最后用 `F.linear` 完成标准前向传播。

**5. `register_buffer` 的作用**
- `weight_int8` 和 `scale` 不是可训练参数，但它们需要和模型一起保存、加载和迁移设备，所以适合注册为 buffer。
- 它们会随 `state_dict()` 保存，并在 `model.to(device)` 时自动迁移，但不会被优化器更新。这正是量化权重所需要的——属于模型状态，但不参与训练。

**6. 进阶思考**
- 本页实现的是 `per-tensor` 量化，工业界常见更细粒度的 `per-channel` 量化。
- `W8A16` 主要压缩的是权重显存，激活仍保持高精度，以平衡收益与精度。
- 本节实现的 W8A16 方案，收益主要在于压缩权重显存和降低带宽压力，计算仍在浮点域完成。若进一步将激活也量化为 INT8（W8A8），则可以充分利用 INT8 Tensor Core 获得计算加速。
