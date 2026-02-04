#!/usr/bin/env python3
"""
Generate NegotiationOutcomes and BehavioralAnalysis figures from experiment_subject_data.csv
Uses deduplicated subject-level data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'experiment_subject_data.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'assets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors matching paper
FC_COLOR = '#5B9BD5'
CL_COLOR = '#ED7D31'

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from experiment_subject_data.csv")
    return df

def generate_negotiation_outcomes(df):
    """Generate NegotiationOutcomes.png - Bar chart comparing FC vs CL outcomes"""
    
    fc = df[df['Condition'] == 'FC']
    cl = df[df['Condition'] == 'CL']
    
    metrics = ['Agent_Utility', 'Human_Utility', 'Agreement_Rounds']
    labels = ['Agent Utility', 'Human Utility', 'Agreement Rounds']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        
        fc_vals = fc[metric].dropna()
        cl_vals = cl[metric].dropna()
        
        # Bar positions
        x = [0, 1]
        means = [fc_vals.mean(), cl_vals.mean()]
        sems = [fc_vals.std() / np.sqrt(len(fc_vals)), cl_vals.std() / np.sqrt(len(cl_vals))]
        
        bars = ax.bar(x, means, yerr=sems, capsize=5, 
                     color=[FC_COLOR, CL_COLOR], edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['FC', 'CL'], fontweight='bold', fontsize=14)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=14, fontweight='bold')
        
        # Add statistical annotation
        # Paired t-test for subject-level
        fc_subj = df.pivot_table(index='Subject_ID', columns='Condition', values=metric)
        if 'FC' in fc_subj.columns and 'CL' in fc_subj.columns:
            fc_paired = fc_subj['FC'].dropna()
            cl_paired = fc_subj['CL'].dropna()
            common = fc_paired.index.intersection(cl_paired.index)
            if len(common) > 1:
                t, p = stats.ttest_rel(cl_paired[common], fc_paired[common])
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                ax.text(0.5, 0.95, f'p = {p:.3f} {sig}', transform=ax.transAxes, 
                       ha='center', fontsize=11, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sems[i] + 0.01,
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'NegotiationOutcomes.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'NegotiationOutcomes.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: NegotiationOutcomes.png")

def generate_behavioral_analysis(df):
    """Generate BehavioralAnalysis.png - Move type proportions by condition"""
    
    move_cols = ['Concession_Rate', 'Selfish_Rate', 'Fortunate_Rate', 'Nice_Rate', 'Unfortunate_Rate']
    move_labels = ['Concession', 'Selfish', 'Fortunate', 'Nice', 'Unfortunate']
    
    fc = df[df['Condition'] == 'FC']
    cl = df[df['Condition'] == 'CL']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(move_cols))
    width = 0.35
    
    fc_means = [fc[col].mean() for col in move_cols]
    cl_means = [cl[col].mean() for col in move_cols]
    fc_sems = [fc[col].std() / np.sqrt(len(fc)) for col in move_cols]
    cl_sems = [cl[col].std() / np.sqrt(len(cl)) for col in move_cols]
    
    bars_fc = ax.bar(x - width/2, fc_means, width, yerr=fc_sems, capsize=4,
                     label='FC', color=FC_COLOR, edgecolor='black', linewidth=1)
    bars_cl = ax.bar(x + width/2, cl_means, width, yerr=cl_sems, capsize=4,
                     label='CL', color=CL_COLOR, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Move Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Behavioral Move Proportions by Condition', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(move_labels, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(0, max(max(fc_means), max(cl_means)) * 1.3)
    
    # Add significance markers
    for i, col in enumerate(move_cols):
        pivot = df.pivot_table(index='Subject_ID', columns='Condition', values=col)
        if 'FC' in pivot.columns and 'CL' in pivot.columns:
            fc_v = pivot['FC'].dropna()
            cl_v = pivot['CL'].dropna()
            common = fc_v.index.intersection(cl_v.index)
            if len(common) > 1:
                t, p = stats.ttest_rel(cl_v[common], fc_v[common])
                if p < 0.05:
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                    max_val = max(fc_means[i] + fc_sems[i], cl_means[i] + cl_sems[i])
                    ax.text(i, max_val + 0.02, sig, ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'BehavioralAnalysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'BehavioralAnalysis.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: BehavioralAnalysis.png")

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING NEGOTIATION OUTCOMES & BEHAVIORAL ANALYSIS FIGURES")
    print("=" * 60)
    
    df = load_data()
    
    print(f"\nFC: {len(df[df['Condition'] == 'FC'])} sessions")
    print(f"CL: {len(df[df['Condition'] == 'CL'])} sessions")
    
    generate_negotiation_outcomes(df)
    generate_behavioral_analysis(df)
    
    print("\n" + "=" * 60)
    print("FIGURES COMPLETE")
    print("=" * 60)
