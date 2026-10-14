# EHR-Learning: 电子病历时序表征学习

## EHR = Electronic Health Record（电子健康记录）

**核心问题：如何从异步、缺失、多频率的临床时序数据中，学习到有意义的患者状态表征？**

EHR 数据不同于普通时序数据：
- **异步采样**：监护仪每分钟采一次，化验每6-12小时采一次
- **大量缺失**：不是所有指标都同时测量
- **多模态**：结构化数据（生命体征、化验）+ 非结构化数据（病历文本）
- **时间语义复杂**：同样是"过去24小时"，对稳定患者和对急剧恶化的患者意义完全不同

---

## 核心假设：临床时间语义不是固定的

```
患者轨迹 ──► 时序编码器 ──► 当前临床状态
                                  │
                                  ▼
                         任务条件时序学习器
                                  │
                                  ▼
                    什么样的历史重要？
                    什么时间尺度重要？
                    什么证据重要？
```

**关键假设**：临床时间语义随着**患者状态**、**病程阶段**和**下游任务**而改变。

| 患者类型 | 状态 | 最佳时间窗口 |
|----------|------|-------------|
| 稳定 ICU 患者 | 生命体征平稳 | 12-24h（长期趋势） |
| 急剧恶化患者 | 休克、器官衰竭 | 1-3h（急性动态） |
| 恢复期患者 | 逐渐好转 | 6-12h（恢复轨迹） |

> **不存在 universally optimal 的固定时间窗口。**

---

## 研究方法：自适应时间核（Adaptive Temporal Kernel）

### 传统固定窗口 vs 自适应核

```
传统方法：
  if |t_i - t_j| < 5h:    # 固定阈值
      attend(t_i, t_j)

我们的方法：
  σ_t = f(patient_state, task)     # 学习得到的时间带宽
  K(Δt) = exp(-Δt² / (2σ_t²))     # 高斯核函数
```

**含义**：模型自己学习"当前患者，我应该看多远以前的信息"。

### 三个研究问题

| RQ | 问题 | 假设 |
|----|------|------|
| **RQ1** | 固定时间粒度是否限制了 EHR 表征？ | 不存在 universally optimal window |
| **RQ2** | 时间相关性是否依赖患者状态？ | 学到的 σ_t 与 SOFA 变化、生命体征不稳定相关 |
| **RQ3** | 下游任务是否应该重塑时间表征？ | 不同任务需要不同的时间证据 |

### 实验设计

**数据集**：MIMIC-IV（可扩展 MIMIC-CXR, PhysioNet）

**任务**（覆盖急性到慢性全谱）：
- **Decompensation**（急性恶化）：证明短时间动态重要
- **Mortality**（死亡风险）：证明长期累积重要
- **Phenotyping**（表型分类）：证明疾病模式重要
- **LOS**（住院时长）：证明整体轨迹理解重要

**Baseline**：
1. 固定窗口：1h / 3h / 5h / 12h
2. 经典 EHR 模型：GRU-D, RETAIN, Transformer-EHR
3. LLM 方法：LLM4EHR 类

**消融实验**：
```
A1: Fixed window → A2: Multi-scale → A3: Patient adaptive
→ A4: Task adaptive → A5: Full model
```

**关键评估**（不只是 AUROC）：
- **时间行为评估**：打乱事件顺序，看模型是否改变 evidence selection
- **学习到的 σ_t 可视化**：患者时间线上展示不同阶段的 σ_t 变化
- **σ_t 与临床指标相关性**：σ_t 是否与 SOFA 变化率相关

---

## 论文1：GRU-D — 处理缺失数据的循环神经网络

**论文**：*Recurrent Neural Networks for Multivariate Timeseries with Missing Values*, Scientific Reports, 2018

**核心问题**：临床时序数据大量缺失，传统方法（均值填充、前向填充）会引入偏差。

**GRU-D 的核心创新：衰减机制（Decay）**

两种衰减：
1. **输入衰减**：观测值随时间衰减趋向均值
2. **隐藏状态衰减**：隐藏状态随时间衰减趋向默认值

```
时间间隔 δ_t = 距上次观测的时间

输入衰减：
  x̃_t^i = x_t^i                    (如果 t 时刻有观测)
  x̃_t^i = γ_x · x̃_{t-1}^i + (1-γ_x) · mean_i  (如果无观测)

  γ_x = exp(-max(0, W_x δ_t + b_x))   [学习到的衰减率]

隐藏状态衰减：
  h_{t-1} = γ_h ⊙ h_{t-1} + (1-γ_h) ⊙ h_mean
```

**与我们工作的关系**：
- GRU-D 使用**固定**的衰减函数，只基于时间间隔
- 我们的方法使用**自适应**的时间带宽，基于患者状态和任务
- 两者都认识到"时间信息很重要"，但我们更进一步：不仅建模数据缺失，还建模**临床相关性**

**关键图**：GRU-D Cell 结构（待补充）

---

## 论文2：RETAIN — 可解释的 EHR 预测模型

**论文**：*RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention*, NeurIPS 2016

**核心思想**：用**逆时间顺序**的注意力机制，识别历史中哪些就诊记录对当前预测最重要。

**两层注意力**：
1. **就诊级注意力**：哪些历史就诊最相关？
2. **事件级注意力**：每次就诊中哪些事件（诊断、操作、用药）最重要？

```
时间方向：◄──────────────────────────────────────
          v_1    v_2    v_3    ...    v_T

RETAIN 逆时间注意力：
          从 v_T 开始，逐步回溯到 v_1
          模拟临床推理：从最近开始，向前看
```

**与我们工作的关系**：
- RETAIN 的注意力权重是**给定输入后固定的**
- 我们的自适应核使这些权重**依赖患者状态和任务**
- RETAIN 没有显式的时间衰减；我们通过学习 σ_t 来加权

**关键图**：RETAIN 注意力可视化（待补充）

---

## 论文3：LLM4EHR — 大语言模型用于临床预测

**论文**：*Prompting Large Language Models for Zero-Shot Clinical Prediction with Structured Longitudinal Electronic Health Record Data*, 2024

**核心问题**：能否用 LLM 直接处理 EHR 数据做零样本/少样本预测？

**方法**：将结构化 EHR 转换为文本 prompt：

```
Patient Demographics: 65-year-old male
Admission Diagnosis: Pneumonia
Vital Signs:
  - Hour 1: HR 85, BP 120/80, SpO2 95%
  - Hour 2: HR 90, BP 118/78, SpO2 94%
Lab Results:
  - Hour 6: WBC 12.5, CRP 45
Medications:
  - Hour 4: Antibiotics started
```

**与我们工作的关系**：
- LLM4EHR 把时间信息当作**文本**，不是**时间**
- LLM 必须从文本中推断时间模式
- 我们的方法**显式学习时间语义**：模型知道"看多远"
- 潜在整合：自适应时间编码器 → LLM，让 LLM 基于时间感知的 embedding 做预测

**关键图**：Prompt 模板结构（待补充）

---

## 论文4：Transformer-based EHR Models

**代表工作**：Med-BERT, BEHRT, G-BERT, EHR-Transformer

**共同模式**：
- 标准 Transformer + 位置编码 → EHR Transformer + 时间编码
- 多头自注意力捕捉事件间依赖

**共同局限**：
- **固定位置编码**（正弦/学习得到，但不自适应）
- **固定注意力模式**（对所有患者相同）
- **固定时间粒度**

**与我们工作的关系**：
- 自适应时间核可以在事件进入 Transformer 之前**预过滤**
- 使 Transformer 的注意力更高效、临床更有意义
- 结合了 Transformer 全局建模 + 自适应时间局部建模

---

## 我们的研究 Idea：Adaptive Temporal Semantics Learning

### 标题
**Task- and Patient-Conditioned Temporal Representation Learning for Asynchronous EHR**

### 架构

```
EHR Events ──► [时序编码器] ──► 患者状态 h_t
                                    │
                                    ▼
                          ┌─────────────────┐
                          │ σ_t = f(h_t,    │
                          │   task_emb)     │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ K(Δt) =         │
                          │ exp(-Δt²/2σ_t²) │
                          └────────┬────────┘
                                   │
                                   ▼
                          加权事件聚合
                                   │
                                   ▼
                          [任务特定预测头]
```

### 方法细节

**自适应带宽网络**：
```python
class AdaptiveBandwidth(nn.Module):
    def __init__(self, state_dim, task_dim, hidden_dim=64):
        self.net = nn.Sequential(
            nn.Linear(state_dim + task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保 σ_t > 0
        )

    def forward(self, patient_state, task_embedding):
        combined = torch.cat([patient_state, task_embedding], dim=-1)
        sigma_t = self.net(combined)
        return sigma_t
```

**自适应核聚合**：
```python
def adaptive_kernel_aggregation(events, timestamps, sigma_t):
    current_time = timestamps[-1:]
    delta_t = current_time - timestamps

    # 高斯核权重
    weights = torch.exp(-delta_t**2 / (2 * sigma_t**2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    # 加权聚合
    aggregated = torch.sum(weights.unsqueeze(-1) * events, dim=1)
    return aggregated, weights
```

### 学习到的 σ_t 可视化

```
患者时间线：
0h        20h       40h       60h
stable    stable    shock     recovery
σ_t=12h   σ_t=12h   σ_t=2h    σ_t=6h

解读：模型"知道"稳定时看远期历史，休克时关注近期事件
```

### 论文 Outline

```
1. Introduction
   - 临床动机（稳定 vs 恶化患者）
   - 固定时间窗口的局限
   - 我们的假设：自适应时间语义

2. Related Work
   - EHR 表征学习
   - 医疗时序建模
   - 注意力机制
   - LLM 在医疗中的应用

3. Method: ATSL
   - 问题形式化
   - 自适应时间核
   - 患者条件带宽
   - 任务条件适应

4. Experiments
   - 设置：MIMIC-IV, 4个任务, baselines
   - RQ1: 固定窗口对比
   - RQ2: 患者依赖分析
   - RQ3: 任务依赖分析
   - 消融实验
   - 时间行为评估

5. Discussion
   - 适应什么时候最有用？
   - 局限与未来工作
   - 临床意义

6. Conclusion
```

---

## 仓库结构

```
EHR-learning/
├── README.md                    # 本文档
├── Fig/                         # 关键图例
├── Paper/                       # 论文 PDF
│   └── LLM4EHR-2607.15447v1.pdf
├── ideas/                       # 研究想法详细文档
│   ├── README.md
│   ├── adaptive_temporal_semantics.md
│   ├── research_questions.md
│   ├── method_design.md
│   └── experiment_design.md
└── docs/                        # 辅助文档
    ├── glossary.md              # 术语表
    ├── dataset_guide.md         # MIMIC-IV 数据指南
    └── meeting_template.md      # 组会 PPT 模板
```

---

## Coming soon...

- 更多论文笔记（Transformer EHR, Survey 等）
- 实验代码实现
- 结果可视化

---

* Built for weekly group meeting presentations and research development.*
*EHR Temporal Representation Learning Group*
