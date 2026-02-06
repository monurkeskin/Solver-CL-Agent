import pandas as pd
import numpy as np
import scipy.stats as stats

print("=== CLIFER Reproducibility Benchmark ===")

# 1. Load Data
df_subj = pd.read_csv('experiment_subject_data.csv')
df_rnd = pd.read_csv('experiment_round_data.csv')

print(f"Loaded Subject Data: {len(df_subj)} rows (N={len(df_subj['Subject_ID'].unique())})")
print(f"Loaded Round Data: {len(df_rnd)} rows (N={len(df_rnd['Subject_ID'].unique())})")

# 2. Report Function
def report_stat(metric_name, col_name, df_src, paired=True, claim_cl=None):
    # Filter
    cl = df_src[df_src['Condition']=='CL'].sort_values('Subject_ID')
    fc = df_src[df_src['Condition']=='FC'].sort_values('Subject_ID')
    
    # Align
    if paired:
        valid = set(cl['Subject_ID']).intersection(set(fc['Subject_ID']))
        cl = cl[cl['Subject_ID'].isin(valid)].sort_values('Subject_ID')
        fc = fc[fc['Subject_ID'].isin(valid)].sort_values('Subject_ID')
    
    vals_cl = cl[col_name]
    vals_fc = fc[col_name]
    
    m_cl = vals_cl.mean()
    m_fc = vals_fc.mean()
    
    if paired:
        # T-test Paired
        t, p = stats.ttest_rel(vals_cl, vals_fc)
        # Cohen d (Paired diff std)
        d = (vals_cl.values - vals_fc.values).mean() / (vals_cl.values - vals_fc.values).std()
    else:
        # Independent (e.g. Rounds)
        t, p = stats.ttest_ind(vals_cl, vals_fc)
        # Pooled SD
        n1, n2 = len(vals_cl), len(vals_fc)
        s1, s2 = vals_cl.std(), vals_fc.std()
        sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
        d = (m_cl - m_fc) / sp
        
    print(f"| {metric_name:<20} | {m_fc:.3f} | {m_cl:.3f} | {p:.4f} | {abs(d):.3f} |")

print("\n--- Negotiation Outcomes (Table II) ---")
print(f"| {'Metric':<20} | {'FC':<5} | {'CL':<5} | {'p':<6} | {'d':<5} |")
print("-" * 60)
report_stat("Agent Utility", "Agent_Utility", df_subj)
report_stat("User Utility", "Human_Utility", df_subj)
report_stat("Agree. Rounds", "Agreement_Rounds", df_subj)

print("\n--- Behavioral Influence (Table III) ---")
report_stat("Concession Rate", "Concession_Rate", df_subj)
report_stat("Selfish Rate", "Selfish_Rate", df_subj)

print("\n--- Affect Perception (Fig 10) ---")
# Round Level Analysis
cl_rnd = df_rnd[df_rnd['Condition']=='CL']
fc_rnd = df_rnd[df_rnd['Condition']=='FC']

ar_cl = cl_rnd['Norm_Arousal'].mean()
ar_fc = fc_rnd['Norm_Arousal'].mean()
# Effect Size (Pooled)
n1, n2 = len(cl_rnd), len(fc_rnd)
s1, s2 = cl_rnd['Norm_Arousal'].std(), fc_rnd['Norm_Arousal'].std()
sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
d_ar = (ar_cl - ar_fc) / sp
t_ar, p_ar = stats.ttest_ind(cl_rnd['Norm_Arousal'], fc_rnd['Norm_Arousal'])

print(f"| {'Arousal (Round)':<20} | {ar_fc:.3f} | {ar_cl:.3f} | {p_ar:.4f} | {abs(d_ar):.3f} |")

# Correlation
r_cl, _ = stats.pearsonr(cl_rnd['Norm_Arousal'], cl_rnd['Norm_Valence'])
r_fc, _ = stats.pearsonr(fc_rnd['Norm_Arousal'], fc_rnd['Norm_Valence'])
print(f"| {'Correlation':<20} | {r_fc:.3f} | {r_cl:.3f} | -      | -     |")

print("\nVerification Complete.")
