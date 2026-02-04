#!/usr/bin/env python3
"""
Raw Data Verification Script
Computes all paper statistics directly from session logs to verify accuracy.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import glob
from collections import defaultdict

# Paths
SESSION_LOGS_BASE = '# PATH REMOVED FOR ANONYMITY/Session_Logs'
OUTPUT_DIR = 'os.path.dirname(os.path.abspath(__file__))/analysis/output'

def get_session_condition(subject_name, session_num):
    """
    Determine condition based on session.
    Session 1 can be either FC or CL depending on counterbalancing.
    We need to check the Valid Experiments Excels naming to determine this.
    """
    # Check if CL file exists for this session
    cl_base = '# PATH REMOVED FOR ANONYMITY/Valid Experiments Excels'
    cl_file = os.path.join(cl_base, f'cl_{subject_name}_Session_{session_num}.csv')
    fc_file = os.path.join(cl_base, f'face_channel_{subject_name}_Session_{session_num}.csv')
    
    # Check file sizes - the actual session has data (>100 bytes)
    if os.path.exists(cl_file) and os.path.getsize(cl_file) > 100:
        return 'CL'
    elif os.path.exists(fc_file) and os.path.getsize(fc_file) > 100:
        return 'FC'
    return None

def load_all_sessions():
    """Load all session logs and determine condition."""
    all_data = []
    
    for session_num in [1, 2]:
        session_dir = os.path.join(SESSION_LOGS_BASE, f'Session Logs Session {session_num}')
        if not os.path.exists(session_dir):
            continue
            
        for file in os.listdir(session_dir):
            if file.endswith('.xlsx'):
                # Extract subject name
                subject = file.replace(f'_Session_{session_num}_negotiation_logs.xlsx', '')
                
                # Determine condition
                condition = get_session_condition(subject, session_num)
                if condition is None:
                    continue
                
                # Load file
                try:
                    filepath = os.path.join(session_dir, file)
                    df = pd.read_excel(filepath)
                    df['Subject'] = subject
                    df['Session'] = session_num
                    df['Condition'] = condition
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined

def analyze_negotiation_outcomes(df):
    """Analyze Agent Utility by condition."""
    print("\n" + "="*60)
    print("NEGOTIATION OUTCOMES (from raw logs)")
    print("="*60)
    
    # Get final round per session (last offer = agreement)
    final_outcomes = df.groupby(['Subject', 'Condition', 'Session']).agg({
        'Agent Utility': 'last',
        'Human Utility': 'last'
    }).reset_index()
    
    # Subject-level means
    subj_means = final_outcomes.groupby(['Subject', 'Condition']).agg({
        'Agent Utility': 'mean',
        'Human Utility': 'mean'
    }).reset_index()
    
    print("\n--- Final Agent Utility ---")
    for cond in ['FC', 'CL']:
        vals = subj_means[subj_means['Condition'] == cond]['Agent Utility'].dropna()
        print(f"  {cond}: {vals.mean():.3f} ± {vals.std():.3f} (N={len(vals)})")
    
    # Paired t-test
    fc = subj_means[subj_means['Condition'] == 'FC']['Agent Utility'].values
    cl = subj_means[subj_means['Condition'] == 'CL']['Agent Utility'].values
    
    if len(fc) > 0 and len(cl) > 0:
        t, p = stats.ttest_rel(cl, fc)
        d = (cl.mean() - fc.mean()) / np.std(cl - fc) if np.std(cl - fc) > 0 else 0
        print(f"  Paired t-test: t = {t:.3f}, p = {p:.4f}, d = {d:.3f}")
    
    return final_outcomes

def analyze_move_types(df):
    """Analyze move type proportions by condition."""
    print("\n" + "="*60)
    print("BEHAVIORAL MOVES (from raw logs)")
    print("="*60)
    
    # Filter human moves only (agent rows don't have moves from human)
    df_human = df[df['Bidder'] == 'Human'].copy()
    df_human['Move'] = df_human['Move'].str.lower().str.strip()
    
    # Get move proportions per subject-condition
    move_counts = df_human.groupby(['Subject', 'Condition', 'Move']).size().reset_index(name='Count')
    totals = df_human.groupby(['Subject', 'Condition']).size().reset_index(name='Total')
    
    move_props = move_counts.merge(totals, on=['Subject', 'Condition'])
    move_props['Proportion'] = move_props['Count'] / move_props['Total']
    
    print("\n--- Move Proportions (Paper Table 2) ---")
    
    moves = ['concession', 'fortunate', 'selfish', 'nice', 'unfortunate']
    results = {}
    
    for move in moves:
        move_data = move_props[move_props['Move'] == move]
        pivot = move_data.pivot_table(index='Subject', columns='Condition', values='Proportion', fill_value=0)
        
        if 'FC' in pivot.columns and 'CL' in pivot.columns:
            fc = pivot['FC'].values
            cl = pivot['CL'].values
            
            t, p = stats.ttest_rel(cl, fc)
            d = (cl.mean() - fc.mean()) / np.std(cl - fc) if np.std(cl - fc) > 0 else 0
            
            print(f"\n  {move.title()}:")
            print(f"    FC: {fc.mean()*100:.1f}% ± {fc.std()*100:.1f}%")
            print(f"    CL: {cl.mean()*100:.1f}% ± {cl.std()*100:.1f}%")
            print(f"    t = {t:.2f}, p = {p:.4f}, d = {d:.2f}")
            
            results[move] = {'fc': fc.mean(), 'cl': cl.mean(), 't': t, 'p': p, 'd': d}
    
    return results

def analyze_rounds_to_agreement(df):
    """Count rounds per session."""
    print("\n" + "="*60)
    print("ROUNDS TO AGREEMENT")
    print("="*60)
    
    # Count rounds per session
    rounds = df.groupby(['Subject', 'Condition', 'Session']).size().reset_index(name='Rounds')
    
    # Subject means
    subj_rounds = rounds.groupby(['Subject', 'Condition'])['Rounds'].mean().reset_index()
    
    print("\n--- Rounds per Session ---")
    for cond in ['FC', 'CL']:
        vals = subj_rounds[subj_rounds['Condition'] == cond]['Rounds'].values
        print(f"  {cond}: {vals.mean():.1f} ± {vals.std():.1f}")
    
    fc = subj_rounds[subj_rounds['Condition'] == 'FC']['Rounds'].values
    cl = subj_rounds[subj_rounds['Condition'] == 'CL']['Rounds'].values
    
    if len(fc) > 0 and len(cl) > 0:
        t, p = stats.ttest_rel(cl, fc)
        d = (cl.mean() - fc.mean()) / np.std(cl - fc) if np.std(cl - fc) > 0 else 0
        print(f"  Paired t-test: t = {t:.3f}, p = {p:.4f}, d = {d:.3f}")

def main():
    print("="*60)
    print("RAW DATA VERIFICATION")
    print("Processing all session logs from source")
    print("="*60)
    
    # Load all session data
    print("\nLoading session logs...")
    df = load_all_sessions()
    
    if df is None or len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"\n  Total rows: {len(df)}")
    print(f"  Unique subjects: {df['Subject'].nunique()}")
    print(f"  Conditions: {df['Condition'].value_counts().to_dict()}")
    
    # Analysis 1: Negotiation Outcomes
    analyze_negotiation_outcomes(df)
    
    # Analysis 2: Move Types
    analyze_move_types(df)
    
    # Analysis 3: Rounds to Agreement
    analyze_rounds_to_agreement(df)
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
