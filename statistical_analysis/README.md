# Statistical Analysis Archive

This folder contains all materials for the statistical analysis supporting the paper:

**"Personalized Affective Perception via Continual Learning Promotes Mutual Cooperation in Human–Agent Negotiation"**

## Folder Structure

```
statistical_analysis/
├── data/                               # Source data files (4 files)
│   ├── experiment_subject_data.csv     # Subject-level outcomes & moves (N=66)
│   ├── final_evaluation_results.csv    # Round-level affect data (N=6,342)
│   ├── affective_behavioral_merged.csv # Round-level merged data
│   └── move_affect_comparison.csv      # Move × Affect interaction data
├── code/                               # Analysis scripts (8 files)
│   ├── comprehensive_lmm_analysis.py   # Main LMM & statistical tests
│   ├── mixed_effects_analysis.py       # Mixed-effects modeling
│   ├── mixed_effects_figures.py        # Figure generation
│   ├── move_affect_circumplex_final.py # Circumplex visualization
│   ├── move_affect_raincloud.py        # Raincloud plots
│   ├── move_affect_causal.py           # Causal pathway analysis
│   ├── move_affect_coherence.py        # A-V coherence analysis
│   └── raw_data_verification.py        # Data verification scripts
├── reports/                            # Final report
│   ├── MixedEffects_Report.tex         # LaTeX source (838 lines)
│   └── MixedEffects_Report.pdf         # Compiled PDF (15 pages)
└── README.md                           # This file
```

## Data Sources

| File | Description | Level | Key Variables |
|------|-------------|-------|---------------|
| `experiment_subject_data.csv` | Main subject outcomes | Subject (N=66) | Agent_Utility, User_Utility, Agreement_Rounds, Nash_Distance, Move rates |
| `final_evaluation_results.csv` | Affect predictions | Round (N=6,342) | Norm_Arousal, Norm_Valence, Subject, Condition, Round |
| `affective_behavioral_merged.csv` | Merged affect+moves | Round | Move_Type, Arousal, Valence |
| `move_affect_comparison.csv` | Move×Condition stats | Aggregated | Move type means, effect sizes |

## Key Findings (Verified)

| Metric | FC | CL | p | d | Status |
|--------|----|----|---|---|--------|
| **Arousal** (Subject) | 0.02±0.07 | **0.14±0.16** | <.001 | **0.80** | ✅ SIG*** |
| **Agent Utility** | 0.73±0.08 | **0.76±0.06** | .005 | **0.36** | ✅ SIG** |
| **Agreement Rounds** | 8.74±5.04 | **7.30±3.54** | .032 | **-0.27** | ✅ SIG* |
| **Concession Rate** | 0.28±0.20 | **0.38±0.27** | .009 | **0.33** | ✅ SIG** |
| **A-V Correlation** | r = -0.27 | **r = 0.88** | <.001 | -- | ✅ SIG*** |

## Report Contents (10 Analyses)

1. **Statistical Methodology** – Power analysis, normality tests, test selection
2. **Affect Perception** – Arousal and valence by condition
3. **A-V Correlations** – Circumplex coherence analysis
4. **Negotiation Outcomes** – Utility, rounds, Nash distance
5. **Behavioral Moves** – Move type proportions
6. **Move × Condition** – Affect during different moves
7. **Causal Pathway** – Affect → Behavior → Outcomes
8. **Temporal Evolution** – Effect across negotiation timeline
9. **Counterbalancing** – Order effects check (p = .167, n.s.)
10. **Individual Differences** – Responder subgroup analysis
11. **Mediation** – Concession as potential mediator (no evidence)

## Statistical Methodology

1. **Normality**: Shapiro-Wilk tests on difference scores
2. **Normal data**: Paired t-tests with Cohen's d
3. **Non-normal data**: Wilcoxon Signed-Rank with Rank-Biserial r
4. **Hierarchical data**: Linear Mixed-Effects Models (ICC = 0.24)
5. **Power**: N=66, α=0.05, power=0.80 → detects d≥0.35

## Reproducibility

```bash
# Run comprehensive analysis
cd code/
python comprehensive_lmm_analysis.py

# Generate figures
python move_affect_circumplex_final.py
python move_affect_raincloud.py

# Verify raw data
python raw_data_verification.py
```

## Contact

For questions about this analysis, contact the paper authors.

---
*Generated: February 2026*
*Report: 15 pages, 236KB*
