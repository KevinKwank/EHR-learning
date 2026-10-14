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

## 论文精讲

每篇论文都有独立的详细分析文档（含原文图例、框架解析、与我们研究的关联）：

| # | 论文 | 核心主题 | 详细分析 |
|---|------|---------|---------|
| 01 | Autoregressive EHR Foundation Models | 自回归 EHR 基础模型 + 多模态 | [分析](papers/01_autoregressive_ehr_foundation_models_analysis.md) |
| 02 | M2G-LLM | GNN + LLM 对齐 + 残差注入 | [分析](papers/02_m2g_llm_analysis.md) |
| 03 | GDP (Generative Deep Patient) | 多模态生成式患者模型 | [分析](papers/03_gdp_multimodal_generative_model_analysis.md) |
| 04 | Unified Multimodal Learners | LLM 统一多模态学习 | [分析](papers/04_unified_multimodal_learners_analysis.md) |
| 07 | KAMELEON | 知识增强的多模态 EHR 预测 | [分析](papers/07_kameleON_analysis.md) |
| 08 | Fusion is not one-size-fits-all | 多模态融合策略研究 | [分析](papers/08_fusion_analysis.md) |
| 11 | Optical Compression | 光学压缩多模态临床数据 | [分析](papers/11_optical_compression_analysis.md) |
| 12 | RL over Patient Trajectories | 强化学习患者轨迹建模 | [分析](papers/12_rl_trajectories_analysis.md) |
| 13 | EHRWorld | EHR 知识图谱世界模型 | [分析](papers/13_ehrworld_analysis.md) |
| 14 | Computable Longitudinal Journeys | 可计算纵向患者旅程 | [分析](papers/14_longitudinal_journeys_analysis.md) |
| - | LLM4EHR | LLM 零样本临床预测 | [分析](papers/LLM4EHR_analysis.md) |

### 经典 Baseline 论文

**GRU-D**（2018, Scientific Reports）：处理缺失数据的循环神经网络，核心创新是**衰减机制（Decay）**——输入和隐藏状态随时间间隔衰减。与我们工作的关系：GRU-D 使用固定衰减函数，我们使用自适应时间带宽。

**RETAIN**（NeurIPS 2016）：逆时间注意力机制，可解释 EHR 预测。两层注意力：就诊级 + 事件级。与我们工作的关系：RETAIN 注意力权重固定，我们的权重依赖患者状态和任务。

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
