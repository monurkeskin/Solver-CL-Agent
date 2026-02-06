# Data Repository for "Personalized Affective Perception for Negotiation Agents (CLIFER)"

This repository contains the anonymized experimental data (N=66) used to reproduce the findings in the paper *Personalized Affective Perception for Negotiation Agents (CLIFER)*.

## Experiment Overview

- **Design**: Within-subjects user study (N=66) comparing two conditions:
  1. **CLIFER (CL)**: A continual learning-enabled Solver agent that personalizes affect perception in real-time.
  2. **FaceChannel (FC)**: A static baseline agent using a pre-trained generalized affect model.
- **Sample Size**: 66 paired sessions (Total 132 sessions).
- **Data Points**: 6,342 round-level affective predictions.
- **Statistical Power**: N=66 provides power ≥0.80 for detecting effects as small as d=0.35.

## Dataset Files

### 1. Subject-Level Outcomes (`experiment_subject_data.csv`)

Contains aggregated metrics for RQ1-RQ4.
**Key columns:**

- `Condition`: **CL** (CLIFER) vs **FC** (FaceChannel).
- `Agent_Utility`: Normalized utility score (0.0 – 1.0) achieved by the agent.
- `User_Utility`: Normalized utility score (0.0 – 1.0) achieved by the human participant.
- `Agreement_Rounds`: Efficiency metric (number of rounds to agreement).
- `Nash_Distance`: **Normalized Product Score**, metric for joint optimality (dist. to Pareto frontier).
- `Concession_Rate`: Frequency of concession moves (Stepwise Analysis). (p=0.013, d=0.32).
- `Selfish_Rate`, `Unfortunate_Rate`, `Fortunate_Rate`, `Nice_Rate`, `Silent_Rate`: Other behavioral move categories.
- `Q1_Prefs` ... `Q5_Time`: Subjective questionnaire responses (1-5 Likert).

### 2. Round-Level Affect (`experiment_round_data.csv`)

Contains round-by-round affective predictions for RQ5.
**Key columns:**

- `Condition`: **CL** vs **FC**.
- `Arousal`: Normalized prediction in range **[-1.0, 1.0]**.
- `Valence`: Normalized prediction in range **[-1.0, 1.0]**.
- **Note on Subject IDs**: Subject IDs in this file are assigned alphabetically to ensure round-level anonymity. Aggregation should be performed by grouping `Condition` and `Subject_ID`.

## Reproduction Instructions

The provided script `verify_benchmarks.py` implements the statistical pipeline described in Section V of the paper:

1. **Normality Check**: Uses Shapiro-Wilk test (implied).
2. **Hypothesis Testing**:
    - **Paired t-test** for normally distributed metrics (e.g., Utility, Concession).
    - **Wilcoxon Signed-Rank Test** for non-normal behavioral distributions (e.g., Unfortunate Moves).
3. **Effect Size**: Cohen's d calculations.

### Run Verification

```bash
python3 verify_benchmarks.py
```

## Key Results Verified

- **Agent Utility**: CL (0.75) > FC (0.73), p=0.017, d=0.31. (Significant)
- **Efficiency**: CL (7.6 rounds) < FC (8.6 rounds), p=0.13. (Trend, not significant)
- **Concession Rate**: CL (0.37) > FC (0.28), p=0.013, d=0.32. (Significant)
- **Arousal Sensitivity**: CL (μ=0.15) > FC (μ=0.01), d=0.88. (Round-level)
- **Generalization**: 76% of subjects showed improved arousal sensitivity with CL.
