
# Method Proposal

## Framework

Input:
- structured EHR time series
- clinical events
- timestamps
- patient context

Encoder:
TS encoder produces dynamic state representation.

Adaptive temporal module:
learns temporal relevance weights.

Example:

attention(t, tau, task) =
P(history at tau contributes to prediction at t)

Instead of:
fixed 5 hour window

learn:
continuous temporal receptive field.

## Possible modules

### 1. Adaptive temporal kernel

Learn bandwidth sigma(t):

K(delta_t)=exp(-delta_t^2/(2 sigma(t)^2))

### 2. Multi-scale temporal routing

Candidate scales:
1h, 3h, 6h, 12h, 24h

Learn patient/task-specific weights.

### 3. Downstream-supervised temporal adaptation

Pretraining learns general representation.
Fine-tuning learns task-specific temporal semantics.
