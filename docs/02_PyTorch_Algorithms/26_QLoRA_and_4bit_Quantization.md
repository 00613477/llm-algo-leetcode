# 26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化

**难度：** Hard | **环境：** GPU required | **标签：** `微调`, `QLoRA`, `量化` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

第 25 节已经说明：只压权重，就能明显降低显存和读取压力。但微调大模型时，问题会更尖锐：底座模型很大，训练又需要额外保存梯度和优化器状态，单靠 8-bit 还不够省。QLoRA 继续往前走一步：把底座尽量压到 4-bit，同时把学习能力留给一条很小的可训练旁路。

这一节不复现工业库里的高性能内核，而是用纯 PyTorch 搭一个教学模拟：低精度基础权重负责存储和前向，高精度 LoRA 旁路负责训练更新。学完后，你应该能看清“底座压缩、旁路训练、计算前还原”这条主线，再理解真实 QLoRA 为什么能把大模型微调的显存门槛大幅压低。

**关键词：** `QLoRA`, `NF4`, `LoRA`

---

## 前置阅读

**导语：** 这一节同时承接量化和微调两条线：先理解为什么要压缩底座权重，再理解为什么只训练 LoRA 旁路。
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
- [25. Quantization W8A16 | W8A16 量化](./25_Quantization_W8A16.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)

## 相关阅读

**导语：** 学完 QLoRA 后，可以继续沿项目线看 LoRA 微调如何交付，也可以沿部署线看量化后如何真正用于推理。
- [30. LoRA Fine-Tuning Project | LoRA 微调项目](./30_LoRA_Fine_Tuning_Project.md)
- [35. Quantized Inference and Deployment | 量化推理与部署](./35_Quantized_Inference_and_Deployment.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 24. SRAM Optimization Techniques | SRAM 优化技术](../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md)

---

### Step 1: 核心机制

> **为什么普通的 4-bit 均匀量化不适合微调大模型？**
> 
> 关键在于权重的统计特性与均匀量化不匹配。神经网络的权重通常服从正态分布（钟形曲线），中间多，两头少。但普通的 4-bit 均匀量化（16 个等间隔码点）是均匀分布的。这会导致大量的精度浪费。

> **NF4 (NormalFloat 4-bit) 的本质：**
> 
> 我们根据标准正态分布的累积分布函数（CDF）划分出 16 个等概率区间，并取每个区间的分位点作为对应码点。这样得到的 16 个 NF4 码点在 0 附近更密集、在尾部更稀疏，因此更贴合权重的分布特性。它们在存储时只用 4 个 bit 表示索引 0 到 15，但对应的真实数值是预先定义好的浮点码点。

> **QLoRA 的训练流：**
> 1. 基础权重（Base Weights）以 NF4 索引的形式存储（每个参数占 4 位），并在微调过程中冻结，不参与梯度更新。
> 2. 前向传播时，先查表把 NF4 索引还原成高精度权重，再交给线性层计算。
> 3. LoRA 旁路保持高精度并参与训练。
> 4. 反向传播时，梯度主要更新 LoRA 旁路参数；底座权重保持冻结，只负责提供稳定的量化存储。**一句话总结** QLoRA 的核心就是：底座权重用 NF4 压缩显存，LoRA 旁路保持高精度以保证微调效果。两者分工明确，互不干扰。

理解了 NF4 在 QLoRA 中的角色之后，下一步我们来看 NF4 的码点具体是怎么算出来的。

### Step 2: 4-bit NormalFloat (NF4) 原理
NF4 的核心是一个预计算的 16 码点 lookup table。它基于标准正态分布的 CDF / 分位数函数（quantile function）构造，使码点在 0 附近更密集、在尾部更稀疏，因此比均匀 4-bit 更贴合神经网络权重的统计特性。

从直观上看，INT4 是"均匀铺点"，而 NF4 是"按概率密度聚集铺点"：权重出现概率高的区域（靠近 0）码点更密，尾部区域码点更疏。

其码点构造可概括为：

$$
q_i = \Phi^{-1}(p_i), \quad p_i = \frac{i - 0.5}{16}, \quad i = 1,2,\dots,16
$$

其中 $\Phi$ 表示标准正态分布的累积分布函数（CDF），$\Phi^{-1}$ 是其反函数（分位数函数）。实际实现中，这些码点会预先计算并存为 lookup table。

NF4 解决了基础权重的极致压缩问题，而 QLoRA 的可训练能力来自 LoRA 策略。LoRA 旁路的具体形式是：在 QLoRA 中，$A$ 负责将输入投影到低秩空间，$B$ 再将低秩特征映射回输出维度，二者共同构成权重更新 $\Delta W = B \cdot A$。基础权重 $W$ 保持冻结，可训练参数从完整矩阵 $W$ 降为两个低秩矩阵 $A$和$B$。QLoRA 还配合 Double Quantization，对 NF4 量化过程中产生的 scale 等元数据再做一次量化，进一步压缩其存储开销。

### Step 3: 代码实现框架
本节我们将模拟 QLoRA 的前向传播链路。这里使用纯 PyTorch 演示 NF4 的核心逻辑，而不是调用真实的 bitsandbytes C++/CUDA 内核；两者的核心思想一致，都是先通过查表完成 NF4 反量化，再进行后续计算。

本节的代码会拆成两步：
- NF4 反量化：通过查表（Lookup Table）将 4-bit 索引还原为高精度浮点权重
- 前向融合：将反量化后的基础权重计算结果与 LoRA 旁路输出相加。这样就能把“存储用 4-bit、计算用高精度”这条核心思路落实到代码实现中。LoRA 旁路的计算可写为：

$$
(x A^\top) B^\top \cdot \mathrm{scaling}
$$

在 Step 4 中，我们将把这两步落到一个完整的 `QLoRALinearSim` 类里，逐行补全 NF4 查表和前向融合的实现。

### Step 4: 动手实战

**要求**：请补全下方 `QLoRALinearSim` 类。为了不引入复杂的 C++ BitsAndBytes 底层实现，我们将用纯 PyTorch 模拟查表反量化和前向传播。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
def create_nf4_lookup_table() -> torch.Tensor:
    """
    创建 4-bit NormalFloat (NF4) 的查表 (共 16 个离散的浮点值)。
    为了教学，这里提供论文中给出的标准 NF4 分位点数值的近似版本。
    """
    nf4_values = [
        -1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
        0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0
    ]
    return torch.tensor(nf4_values)

class QLoRALinearSim(nn.Module):
    """
    模拟 QLoRA 的 Linear 层。
    真实的 QLoRA 会把 weight 存为 uint8，两个 4-bit 挤在一个字节里。
    为了只演示原理，我们这里用 torch.int8 存储 0-15 的索引。
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        
        # 1. 冻结的低精度基础权重 (保存 0~15 的索引)
        self.register_buffer("weight_nf4_indices", torch.randint(0, 16, (out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.tensor(1.0)) # 简化的单缩放因子
        self.register_buffer("nf4_table", create_nf4_lookup_table())
        
        # 2. 活跃的高精度 LoRA 适配器
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.register_buffer("scaling", torch.tensor(alpha / r))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 1: 基础权重反量化（查表还原）
        # 提示：将 weight_nf4_indices 转为 long 类型作为索引，
        #   从 nf4_table 中取值后乘以 weight_scale
        # ==========================================
        # indices = ???
        # dequantized_base_weight = ???

        # ==========================================
        # TODO 2: 计算基础分支和 LoRA 旁路分支
        # 基础分支: F.linear(x, dequantized_base_weight)
        # LoRA 分支: (x @ lora_A.T) @ lora_B.T * scaling
        # ==========================================
        # base_out = ???
        # lora_out = ???

        return base_out + lora_out

```


```python
# 测试你的实现
def test_qlora():
    try:
        torch.manual_seed(42)

        # 使用一个更小、可精确对照的配置，直接验证 NF4 查表 + LoRA 旁路的公式链路
        batch, seq, in_dim, out_dim, r = 1, 2, 4, 3, 2
        x = torch.tensor([[[0.1, -0.2, 0.3, -0.4], [0.5, 0.6, -0.7, 0.8]]], requires_grad=True)
        layer = QLoRALinearSim(in_features=in_dim, out_features=out_dim, r=r, alpha=8.0)

        with torch.no_grad():
            layer.weight_nf4_indices.copy_(torch.tensor([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ], dtype=torch.int8))
            layer.weight_scale.copy_(torch.tensor(0.5))
            layer.lora_A.copy_(torch.tensor([
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ], dtype=layer.lora_A.dtype))
            layer.lora_B.copy_(torch.tensor([
                [0.9, -0.1],
                [0.2, 0.3],
                [-0.4, 0.7],
            ], dtype=layer.lora_B.dtype))

        out = layer(x)
        assert out.shape == (batch, seq, out_dim), "输出形状不正确！"

        indices_ref = layer.weight_nf4_indices.long()
        dequantized_ref = layer.nf4_table[indices_ref] * layer.weight_scale
        base_out_ref = F.linear(x, dequantized_ref)
        lora_out_ref = (x @ layer.lora_A.T) @ layer.lora_B.T * layer.scaling
        out_ref = base_out_ref + lora_out_ref
        assert torch.allclose(out, out_ref, atol=1e-5), "输出数值不正确！查表反量化或 LoRA 计算有误。"
        assert not torch.allclose(out, base_out_ref, atol=1e-6), "LoRA 旁路应该参与输出，不能退化为纯基础分支！"

        # 2. 验证反向传播时的梯度断点机制 (QLoRA 的灵魂)
        out.sum().backward()
        assert x.grad is not None, "输入 x 没有获得梯度！"
        assert layer.lora_A.grad is not None, "LoRA_A 没有更新梯度！"
        assert layer.lora_B.grad is not None, "LoRA_B 没有更新梯度！"
        assert not layer.weight_nf4_indices.requires_grad, "基础权重的索引不应该有梯度！"
        assert layer.weight_nf4_indices.grad is None, "冻结的基础权重不应该产生梯度！"
        assert layer.weight_scale.grad is None, "冻结的缩放因子不应该产生梯度！"

        print("✅ 查表反量化逻辑正确！")
        print("✅ 梯度流向正确：低精度冻结，高精度更新！")
        print("\n ✅ QLoRA 核心模拟测试通过！(真实生产环境需使用 bitsandbytes 等优化库)")

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
            print("代码可能未完成，导致了断言失败")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_qlora()

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
def create_nf4_lookup_table() -> torch.Tensor:
    """
    创建 4-bit NormalFloat (NF4) 的查表 (共 16 个离散的浮点值)。
    """
    nf4_values = [
        -1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
        0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0
    ]
    return torch.tensor(nf4_values)

class QLoRALinearSim(nn.Module):
    """
    模拟 QLoRA 的 Linear 层。
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        
        # 1. 冻结的低精度基础权重 (保存 0~15 的索引)
        self.register_buffer("weight_nf4_indices", torch.randint(0, 16, (out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("nf4_table", create_nf4_lookup_table())
        
        # 2. 活跃的高精度 LoRA 适配器
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.register_buffer("scaling", torch.tensor(alpha / r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 1: 基础权重反量化（查表还原）
        # 1. 将 weight_nf4_indices 转换为长整型 (long)，以作为查表的索引
        indices = self.weight_nf4_indices.long()
        
        # 2. 从 nf4_table 中取出对应的浮点数值
        # 3. 乘以 weight_scale 恢复范围
        dequantized_base_weight = self.nf4_table[indices] * self.weight_scale
        
        # TODO 2: 分别计算基础分支和 LoRA 旁路分支
        base_out = F.linear(x, dequantized_base_weight)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        
        return base_out + lora_out
```

### 解析

**1. TODO 1: 基础权重反量化**
- **实现方式**：`indices = self.weight_nf4_indices.long()`，`dequantized_base_weight = self.nf4_table[indices] * self.weight_scale`
- **关键点**：通过查表将 4-bit 索引（0-15）映射到 NF4 浮点值
- **技术细节**：NF4 查表包含 16 个根据正态分布分位点设计的浮点值，密度集中在 0 附近

**2. TODO 2: 分别计算基础前向和 LoRA 旁路**
- **实现方式**：`base_out = F.linear(x, dequantized_base_weight)`，`lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling`
- **关键点**：基础权重冻结（不更新梯度），LoRA 权重可训练
- **技术细节**：LoRA 输出需要乘以 scaling 因子（alpha / r）来平衡贡献

**NF4 量化原理**
- **INT4 的局限**：INT4 的 16 个码点在数轴上均匀分布，但神经网络权重通常集中在 0 附近（正态分布），两者分布特性不匹配。均匀码点导致尾部区域分配了过多码点（浪费），而 0 附近区域码点不足（精度损失）。
- **NF4 解决方案**：根据标准正态分布的累积分布函数（CDF）计算 16 个分位点
- **信息密度**：在 0 附近分配更多的量化点，在尾部分配更少的点
- **查表机制**：4-bit 索引 → NF4 浮点值 → 乘以 scale 恢复原始范围

**工程优化要点**
- **显存节省**：基础权重从 FP16（2 bytes）降至 NF4（0.5 bytes），节省 75% 显存
- **双重量化**：对 scale 参数本身也进行量化，进一步节省显存
- **关于 scaling 的存储方式**：scaling = alpha / r 在代码中被注册为 buffer（而非 Python float），以确保它随模型一起保存、加载和设备迁移。这是区分"模型状态"与"普通变量"的工程习惯。
- **分块量化**：NF4 将权重分成若干小块（如每 64 个参数一块），每块独立计算一个 scale，以适应不同区域的数值范围。这些 scale 本身再用 Double Quantization 进一步压缩，避免元数据开销过大。
- **梯度流向**：基础权重冻结，梯度只更新 LoRA 参数，避免量化误差累积
- **训练效率**：虽然反量化增加计算开销，但显存节省允许更大的 batch size
- **工业实践**：QLoRA 使 33B 模型可在单张 24GB 显卡上微调，65B 模型可在单张 48GB 显卡上微调