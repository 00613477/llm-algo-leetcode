# LLM Algo LeetCode 全作业完成版设计

## 目标

在 `00613477/llm-algo-leetcode` 的 `solution/all-homework` 分支上，将仓库中可判定为“题目区”的 Notebook 作业全部补全，并保持原教程结构、讲解、测试与参考答案区不被破坏。

## 范围

### 纳入

- `00_Prerequisites`
- `01_Hardware_Math_and_Systems`
- `02_PyTorch_Algorithms`
- `03_Triton_Kernels`
- `04_CUDA_and_System_Optimization`
- 上述目录中 Notebook 题目区里的 `TODO`、`???`、`NotImplementedError`、明确要求补全的函数/类/算子实现。
- 与题目直接绑定的测试代码所要求的必要实现。

### 不纳入

- `05_CUDA_Rust`：当前上游仍为预留模块，不把占位内容伪装成已完成作业。
- `topic_discussion`、`team_study`：属于专题/共学材料，不作为主线 Notebook 作业处理，除非其中存在明确的独立题目与可验证答案。
- 教材示例代码、纯理论说明、链接页和导航页不做无关修改。

## 实现原则

1. **以题目区为修改目标**：只补齐学习者应完成的代码，不删除题目、不隐藏测试、不改写题意。
2. **以仓库官方参考实现为主要基准**：优先使用同一 Notebook 的“参考代码与解析”区域，避免引入与课程口径不一致的自定义实现。
3. **以测试行为为最终约束**：如果参考代码与测试约束存在细节差异，以当前仓库测试可验证行为为准，并保持实现语义正确。
4. **最小修改**：只改需要完成的代码单元格及少量必要注释，不进行与作业无关的重构。
5. **保留学习价值**：完成代码应可读，关键算法步骤保留简洁注释，但不把答案区的长篇解析复制进题目区。
6. **不伪造 GPU 验证**：CPU 环境无法运行的 Triton/CUDA/GPU 题目，只能报告结构验证或仓库脚本的 skip 结果；不能声称运行级测试通过。

## 作业识别规则

按以下优先级识别：

1. Markdown 中明确出现“动手实战 / 要求 / 请补全 / TODO / 实现”；
2. 对应代码单元格包含 `TODO`、`???`、`NotImplementedError`、空函数体等占位；
3. Notebook 在 `STOP HERE` 之前属于题目区，且仓库 `tools/test_notebook_answers.py` 能抽取并测试；
4. 若某 Notebook 是纯理论章节，则只保留理论内容，不人为增加实现题。

## 完成策略

对每个目标 Notebook：

1. 读取完整 Notebook；
2. 定位题目区和“参考代码与解析”区；
3. 将参考实现映射回对应题目函数/类，而不是直接替换整段 Notebook；
4. 清理遗留的实现占位符；
5. 保留测试单元格；
6. 运行或静态检查该 Notebook；
7. 再进入下一题。

## 验证标准

### 仓库级

执行：

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py part4 --no-build
```

### 题目区级

对支持的 Notebook 使用：

```bash
python tools/test_notebook_answers.py --all --dir <目录> --mode both
```

期望：

- 题目区：原先应为 `expected_fail` 的实现题，在完成后应变为 `pass`；
- 答案区：保持 `pass`；
- GPU 不可用时：只接受仓库脚本明确给出的 `skip`，并在最终结果中列出未做运行级验证的章节。

## 提交组织

分支：`solution/all-homework`

建议按 Part 提交，便于回滚和复习：

- `complete part0-1 homework`
- `complete part2 homework`
- `complete part3 homework`
- `complete part4 homework`
- `add homework completion report`

## 最终交付

1. 一个完成全部主线作业的 GitHub 分支；
2. 保留所有原始 Notebook 教学结构；
3. 一份完成情况报告，列出：
   - 完成的 Notebook 数；
   - 完成的 TODO/题目数量；
   - CPU 通过项；
   - GPU/Triton/CUDA 因环境限制仅做结构验证的项；
   - 仍属于上游预留/建设中的内容；
4. 如条件允许，创建 Draft PR 方便浏览完整 diff。

## 成功判定

只有同时满足以下条件，才称为“全作业完成版”：

- Part 0-4 所有明确实现型题目均无未完成占位符；
- 可在当前环境执行的官方验证全部通过；
- 无 GPU 的部分被明确标记为未做运行级验证，而非错误宣称通过；
- 原参考答案区和教程结构未被破坏；
- Part 5 预留内容未被错误计入已完成。
