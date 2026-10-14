# Method Design: Adaptive Temporal Semantics Learning

> **Detailed technical design for the ATSL model**

---

## 1. Problem Formulation

### Input
- Patient EHR sequence: `{(e_i, t_i)}_{i=1}^N` where `e_i` is the event features and `t_i` is the timestamp
- Prediction time: `t_pred`
- Task identifier: `task ∈ {mortality, decompensation, phenotyping, LOS}`

### Output
- Prediction: `ŷ = model(EHR_sequence, t_pred, task)`
- Learned temporal bandwidth: `σ_t` (for interpretability)

### Objective
```
minimize: L_pred(ŷ, y) + λ · L_reg(σ_t)

where:
  L_pred = prediction loss (cross-entropy, MSE, etc.)
  L_reg = regularization to prevent σ_t collapse
```

---

## 2. Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  EHR Events ──► [Temporal Encoder] ──► Patient State h_t       │
│                                              │                  │
│                                              ▼                  │
│                                    ┌──────────────────┐         │
│                                    │ Adaptive Bandwidth│         │
│                                    │ σ_t = f(h_t,τ)  │         │
│                                    └────────┬─────────┘         │
│                                             │                   │
│                                             ▼                   │
│                                    ┌──────────────────┐         │
│                                    │ Adaptive Kernel  │         │
│                                    │ K(Δt;σ_t)       │         │
│                                    └────────┬─────────┘         │
│                                             │                   │
│                                             ▼                   │
│                                    Weighted Aggregation         │
│                                             │                   │
│                                             ▼                   │
│                                    [Prediction Head]            │
│                                             │                   │
│                                             ▼                   │
│                                        ŷ (prediction)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 2.1 Temporal Encoder

**Purpose**: Convert irregular EHR events into a dense patient state representation.

**Choice**: GRU-D (for missing data handling) or Transformer (for long-range dependencies)

**GRU-D Option**:
```python
class GRUDEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.gru_d = GRUD(input_dim, hidden_dim)

    def forward(self, events, timestamps, masks):
        # events: [batch, seq_len, input_dim]
        # timestamps: [batch, seq_len]
        # masks: [batch, seq_len] — 1 if observed, 0 if missing
        h = self.gru_d(events, timestamps, masks)
        return h  # [batch, hidden_dim]
```

**Transformer Option**:
```python
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.time_encoding = TimeEncoding(hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers=3
        )

    def forward(self, events, timestamps):
        x = self.embedding(events) + self.time_encoding(timestamps)
        h = self.transformer(x)
        return h[:, -1, :]  # last timestep
```

#### 2.2 Adaptive Bandwidth Network

**Purpose**: Learn σ_t as a function of patient state and task.

```python
class AdaptiveBandwidth(nn.Module):
    def __init__(self, state_dim, task_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + task_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # ensure σ_t > 0
        )

    def forward(self, patient_state, task_embedding):
        combined = torch.cat([patient_state, task_embedding], dim=-1)
        sigma_t = self.net(combined)
        return sigma_t  # [batch, 1]
```

#### 2.3 Adaptive Kernel Aggregation

**Purpose**: Weight historical events by temporal relevance.

```python
class AdaptiveKernelAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, events, timestamps, sigma_t):
        """
        events: [batch, seq_len, event_dim]
        timestamps: [batch, seq_len]
        sigma_t: [batch, 1]
        """
        current_time = timestamps[:, -1:]  # [batch, 1]
        delta_t = current_time - timestamps  # [batch, seq_len]

        # Gaussian kernel
        weights = torch.exp(-delta_t**2 / (2 * sigma_t**2))  # [batch, seq_len]

        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted aggregation
        aggregated = torch.sum(weights.unsqueeze(-1) * events, dim=1)
        return aggregated, weights
```

#### 2.4 Task Embedding

```python
class TaskConditioning(nn.Module):
    def __init__(self, num_tasks, task_dim):
        super().__init__()
        self.task_embeddings = nn.Embedding(num_tasks, task_dim)

    def forward(self, task_ids):
        return self.task_embeddings(task_ids)  # [batch, task_dim]
```

#### 2.5 Prediction Head

```python
class PredictionHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(x)
```

---

## 3. Full Model

```python
class ATSLModel(nn.Module):
    def __init__(self, event_dim, hidden_dim, task_dim, num_tasks, num_classes):
        super().__init__()
        self.encoder = GRUDEncoder(event_dim, hidden_dim)
        self.task_conditioning = TaskConditioning(num_tasks, task_dim)
        self.bandwidth_net = AdaptiveBandwidth(hidden_dim, task_dim)
        self.kernel_agg = AdaptiveKernelAggregation()
        self.prediction_head = PredictionHead(hidden_dim, num_classes)

    def forward(self, events, timestamps, masks, task_ids):
        # Encode patient state
        h_t = self.encoder(events, timestamps, masks)  # [batch, hidden_dim]

        # Get task embedding
        task_emb = self.task_conditioning(task_ids)  # [batch, task_dim]

        # Learn adaptive bandwidth
        sigma_t = self.bandwidth_net(h_t, task_emb)  # [batch, 1]

        # Aggregate with adaptive kernel
        aggregated, weights = self.kernel_agg(
            events, timestamps, sigma_t
        )  # [batch, hidden_dim], [batch, seq_len]

        # Predict
        logits = self.prediction_head(aggregated)  # [batch, num_classes]

        return logits, sigma_t, weights
```

---

## 4. Loss Function

```python
def compute_loss(logits, labels, sigma_t, lambda_reg=0.01):
    # Prediction loss
    pred_loss = F.cross_entropy(logits, labels)

    # Regularization: prevent σ_t from collapsing to 0 or ∞
    # Option 1: L2 regularization toward a target value
    target_sigma = 6.0  # hours
    reg_loss = lambda_reg * torch.mean((sigma_t - target_sigma)**2)

    # Option 2: Entropy regularization (prevent collapse)
    # reg_loss = -lambda_reg * torch.mean(
    #     sigma_t * torch.log(sigma_t + 1e-8)
    # )

    total_loss = pred_loss + reg_loss
    return total_loss, pred_loss, reg_loss
```

---

## 5. Training Strategy

### Two-Stage Training

**Stage 1: Pre-training**
- Train the temporal encoder on a general task (e.g., next-event prediction)
- Learn general-purpose patient state representations

**Stage 2: Fine-tuning**
- Freeze or fine-tune the encoder
- Train the bandwidth network and prediction head on each task
- Learn task-specific temporal semantics

### Curriculum Learning

```
Epoch 1-10:  Train with fixed σ_t = 6h (warm-up)
Epoch 11-20: Gradually introduce adaptive σ_t
Epoch 21+:   Full adaptive training
```

---

## 6. Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_dim | 128 | Patient state dimension |
| task_dim | 32 | Task embedding dimension |
| learning_rate | 1e-3 | Adam optimizer |
| batch_size | 64 | |
| max_seq_len | 100 | Maximum events per patient |
| dropout | 0.1-0.2 | Regularization |
| lambda_reg | 0.01 | σ_t regularization weight |
| sigma_init | 6.0 | Initial bandwidth (hours) |

---

## 7. Implementation Notes

### Handling Different Time Scales
- Normalize timestamps to hours since admission
- Clip extreme time gaps (e.g., > 72h) to prevent numerical issues

### Numerical Stability
- Add epsilon (1e-8) to denominator in kernel computation
- Use log-space computation for very small/large σ_t

### Interpretability
- Log σ_t values during training
- Visualize σ_t distribution across patients and tasks
- Correlate σ_t with clinical indicators (SOFA, etc.)
