
# Proposed paper story

## Core hypothesis

Fixed temporal representation is suboptimal.

Clinical temporal relevance depends on:
- patient state;
- physiological dynamics;
- event density;
- prediction task.

Therefore:

Patient state + task objective -> adaptive temporal semantics.

## Story

Previous EHR models:
- fixed windows;
- fixed sampling;
- task-agnostic temporal encoder.

Our hypothesis:
The downstream task should reshape temporal representation.

The model should learn:
1. what historical period is relevant;
2. what temporal resolution is useful;
3. which clinical events should align with current physiology.

## Proposed title

Task- and Patient-Conditioned Temporal Representation Learning for Asynchronous EHR Modeling
