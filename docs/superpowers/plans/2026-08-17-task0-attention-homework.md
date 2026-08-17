# Task 0 Attention Homework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 `02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb` 题目区并取得可核验的测试通过结果，用于 GitHub issue #84 打卡。

**Architecture:** 保持 Notebook 原教学结构不变，只将题目区中的实现占位符按同 Notebook 官方参考实现补齐。验收直接使用仓库 `tools/test_notebook_answers.py` 的 question/answer 双模式，避免自定义测试口径。

**Tech Stack:** Python 3, PyTorch, Jupyter Notebook, nbformat, repository verification scripts

## Global Constraints

- 只修改 Task 0 的 `02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb` 及打卡证据文件。
- 不删除题目、测试或参考答案区。
- 不伪造测试结果；只有实际命令退出码成功时才生成“测试通过”证据。
- GitHub issue #84 打卡必须包含测试通过截图/图片证据。

---

### Task 1: 完成 Attention MHA/GQA 题目区

**Files:**
- Modify: `02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb`

**Interfaces:**
- Consumes: Notebook 中现有题目区、参考答案区和测试函数。
- Produces: 可被 `tools/test_notebook_answers.py` 的 question 模式执行通过的题目区实现。

- [ ] **Step 1: 基线运行题目区测试**

Run:
```bash
python tools/test_notebook_answers.py 02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb --mode question
```
Expected: 当前未完成题目应失败或出现未实现占位符。

- [ ] **Step 2: 定位题目占位与对应参考实现**

读取 Notebook JSON，识别 `STOP HERE` 前的 `TODO` / `???` / `NotImplementedError`，并映射到“参考代码与解析”区域的对应类、函数和张量运算。

- [ ] **Step 3: 回填最小正确实现**

只修改题目区对应代码 cell，保持函数/类签名、测试代码和教学文本不变。

- [ ] **Step 4: 运行 question + answer 双测试**

Run:
```bash
python tools/test_notebook_answers.py 02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb --mode both
```
Expected: 题目区 `pass`，答案区 `pass`；若仓库明确因无 GPU 返回 `skip`，记录该状态而不宣称运行通过。

### Task 2: 生成并提交打卡证据

**Files:**
- Create: `homework/task0-attention-test-pass.png`

**Interfaces:**
- Consumes: Task 1 实际测试命令输出。
- Produces: issue #84 可直接展示的测试结果图片。

- [ ] **Step 1: 将实际测试输出保存为证据**

保存完整命令、Notebook 路径、题目区和答案区最终状态。

- [ ] **Step 2: 从真实测试输出生成 PNG**

图片必须显示 Task 0 文件名、测试命令和 PASS 状态；不得手工填写未经运行验证的结果。

- [ ] **Step 3: 将 Notebook 和 PNG 提交到 `solution/all-homework`**

提交信息：
```text
complete task0 attention homework
```

### Task 3: GitHub Issue #84 打卡

**Files:** none

**Interfaces:**
- Consumes: GitHub ID `00613477`、测试证据图片链接。
- Produces: `datawhalechina/llm-algo-leetcode#84` 下的打卡评论。

- [ ] **Step 1: 提交 issue 评论**

评论包含 Task 0 完成说明、GitHub ID、测试通过证据图片。

- [ ] **Step 2: 读取 issue 评论确认提交成功**

确认评论出现在 #84，且图片链接可以从评论 Markdown 中解析。
