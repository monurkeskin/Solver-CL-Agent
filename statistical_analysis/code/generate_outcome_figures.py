#!/usr/bin/env python3
"""
Generate NegotiationOutcomes and BehavioralAnalysis figures from experiment_subject_data.csv
Uses violin plots with paired subject lines (matching paper style).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'experiment_subject_data.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'assets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors matching paper
FC_COLOR = '#5B9BD5'  # Blue
CL_COLOR = '#ED7D31'  # Orange

# Font sizes
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 11

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from experiment_subject_data.csv")
    return df

def draw_paired_violin(ax, fc_vals, cl_vals, fc_subj_ids, cl_subj_ids, ylabel, title, show_ylabel=True):
    """
    Draw a paired violin plot with boxplot overlay, scatter points, and paired subject lines.
    """
    pos_fc, pos_cl = 0, 1
    
    # Skip if insufficient data
    if len(fc_vals) < 3 or len(cl_vals) < 3:
        ax.text(0.5, 0.5, f'{title}\nInsufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
        return
    
    # FC violin (left half)
    parts_fc = ax.violinplot([fc_vals], positions=[pos_fc], showmeans=False, showmedians=False, widths=0.6)
    for pc in parts_fc['bodies']:
        pc.set_facecolor(FC_COLOR)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
    for p in ['cbars', 'cmins', 'cmaxes']:
        if p in parts_fc:
            parts_fc[p].set_visible(False)
    
    # CL violin (right half)
    parts_cl = ax.violinplot([cl_vals], positions=[pos_cl], showmeans=False, showmedians=False, widths=0.6)
    for pc in parts_cl['bodies']:
        pc.set_facecolor(CL_COLOR)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
    for p in ['cbars', 'cmins', 'cmaxes']:
        if p in parts_cl:
            parts_cl[p].set_visible(False)
    
    # Boxplots for quartiles
    for vals, pos in [(fc_vals, pos_fc), (cl_vals, pos_cl)]:
        bp = ax.boxplot([vals], positions=[pos], widths=0.15, patch_artist=True, manage_ticks=False)
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
        for med in bp['medians']:
            med.set_color('black')
            med.set_linewidth(2)
    
    # Scatter points (jittered)
    np.random.seed(42)  # For reproducibility
    ax.scatter(pos_fc + np.random.uniform(-0.08, 0.02, len(fc_vals)), fc_vals, c='black', s=15, alpha=0.5, zorder=5)
    ax.scatter(pos_cl + np.random.uniform(-0.02, 0.08, len(cl_vals)), cl_vals, c='black', s=15, alpha=0.5, zorder=5)
    
    # Paired subject lines
    fc_dict = dict(zip(fc_subj_ids, fc_vals))
    cl_dict = dict(zip(cl_subj_ids, cl_vals))
    common_subjects = set(fc_subj_ids) & set(cl_subj_ids)
    
    for subj in common_subjects:
        fc_v = fc_dict[subj]
        cl_v = cl_dict[subj]
        ax.plot([pos_fc + 0.08, pos_cl - 0.08], [fc_v, cl_v], color='gray', alpha=0.3, lw=0.5, zorder=1)
    
    # Paired t-test and significance bracket
    if len(common_subjects) >= 5:
        fc_paired = [fc_dict[s] for s in common_subjects]
        cl_paired = [cl_dict[s] for s in common_subjects]
        _, p_val = stats.ttest_rel(fc_paired, cl_paired)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        if sig:
            y_max = max(max(fc_vals), max(cl_vals)) * 1.05
            ax.plot([pos_fc, pos_fc, pos_cl, pos_cl], [y_max, y_max + 0.02 * y_max, y_max + 0.02 * y_max, y_max], 'k-', lw=1.5)
            ax.text(0.5, y_max + 0.04 * y_max, sig, ha='center', fontsize=12, fontweight='bold')
    
    # Formatting
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks([pos_fc, pos_cl])
    ax.set_xticklabels(['FC', 'CL'], fontsize=TICK_SIZE, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, axis='y', alpha=0.3)

def generate_negotiation_outcomes(df):
    """Generate NegotiationOutcomes.png - 2x2 violin plots with paired lines"""
    
    # Pivot to get subject-level data
    fc = df[df['Condition'] == 'FC'].set_index('Subject_ID')
    cl = df[df['Condition'] == 'CL'].set_index('Subject_ID')
    
    # Using available columns (no Nash_Distance in this dataset)
    metrics = [
        ('Agent_Utility', 'Utility ($U$)', 'Agent'),
        ('Human_Utility', 'Utility ($U$)', 'Participant'),
        ('Agreement_Rounds', 'Rounds', 'Agreement Rounds'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx, (col, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        
        fc_data = fc[col].dropna()
        cl_data = cl[col].dropna()
        
        draw_paired_violin(
            ax, 
            fc_data.values, cl_data.values,
            fc_data.index.tolist(), cl_data.index.tolist(),
            ylabel, title,
            show_ylabel=(idx % 2 == 0)  # Only left column gets y-label
        )
    
    # Hide unused subplot (4th panel)
    axes[3].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'NegotiationOutcomes.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'NegotiationOutcomes.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: NegotiationOutcomes.png (violin plot with paired lines)")

def generate_behavioral_analysis(df):
    """Generate BehavioralAnalysis.png - 2x3 violin plots with paired lines"""
    
    fc = df[df['Condition'] == 'FC'].set_index('Subject_ID')
    cl = df[df['Condition'] == 'CL'].set_index('Subject_ID')
    
    move_cols = [
        ('Concession_Rate', 'Concession'),
        ('Nice_Rate', 'Nice'),
        ('Fortunate_Rate', 'Fortunate'),
        ('Unfortunate_Rate', 'Unfortunate'),
        ('Selfish_Rate', 'Selfish'),
        ('Silent_Rate', 'Silent')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (col, title) in enumerate(move_cols):
        ax = axes[idx]
        
        fc_data = fc[col].dropna()
        cl_data = cl[col].dropna()
        
        draw_paired_violin(
            ax, 
            fc_data.values, cl_data.values,
            fc_data.index.tolist(), cl_data.index.tolist(),
            'Ratio', title,
            show_ylabel=(idx % 3 == 0)  # Only first column gets y-label
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'BehavioralAnalysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'BehavioralAnalysis.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: BehavioralAnalysis.png (violin plot with paired lines)")

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING VIOLIN PLOTS FOR OUTCOMES & BEHAVIORAL ANALYSIS")
    print("=" * 60)
    
    df = load_data()
    
    print(f"\nFC: {len(df[df['Condition'] == 'FC'])} sessions")
    print(f"CL: {len(df[df['Condition'] == 'CL'])} sessions")
    
    generate_negotiation_outcomes(df)
    generate_behavioral_analysis(df)
    
    print("\n" + "=" * 60)
    print("FIGURES COMPLETE")
    print("=" * 60)
