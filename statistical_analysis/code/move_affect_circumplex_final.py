#!/usr/bin/env python3
"""
Move-Affect Circumplex - Final Publication Version v2
=======================================================
Refinements:
- Nice moves in foreground (biggest change)
- CL legend forced to lower-right
- Better point visibility for Selfish/Concession in FC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Professional fonts - LARGER text (+2pt)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1.0

# ============================================================================
# WONG COLOR PALETTE (matching LaTeX definitions)
# ============================================================================

MOVE_COLORS = {
    'Fortunate': '#009E73',     # wongGreen RGB(0, 158, 115)
    'Concession': '#56B4E9',    # wongBlue RGB(86, 180, 233)
    'Selfish': '#D55E00',       # wongRed RGB(213, 94, 0)
    'Nice': '#F0E442',          # wongYellow RGB(240, 228, 66)
    'Unfortunate': '#C8C8C8',   # wongGray RGB(200, 200, 200)
    'Silent': '#CC79A7',        # wongPink (other move type)
}

# Z-order priority (higher = more in front)
# Concession/Selfish on top to be visible over background blobs
MOVE_ZORDER = {
    'Unfortunate': 1,  # Back
    'Silent': 2,
    'Fortunate': 3,
    'Nice': 4,
    'Selfish': 5,
    'Concession': 6,   # Front (visible over others)
}

# Alpha per move type (lower for background moves)
MOVE_ALPHA = {
    'Fortunate': 0.20,
    'Concession': 0.22,
    'Selfish': 0.22,
    'Nice': 0.25,
    'Unfortunate': 0.12,  # Reduced - shows through less
    'Silent': 0.12,       # Reduced - shows through less
}

# ============================================================================
# LOAD DATA
# ============================================================================
# NOTE: This visualization shows affect by Move Type. 
# - Round 1 is excluded (no previous offer to calculate move delta)
# - Only 6 scientific move types are included (Concession, Selfish, etc.)
# - For overall arousal/valence analysis, use final_evaluation_results.csv
# ============================================================================

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'affective_behavioral_merged.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'assets')

df = pd.read_csv(DATA_PATH)
print(f"Total rounds in dataset: {len(df)} (FC: {len(df[df['Condition'] == 'FC'])} | CL: {len(df[df['Condition'] == 'CL'])})")

# Exclude Round 1 (no previous offer to calculate delta)
df = df[df['Round'] != 1]
print(f"After excluding Round 1: {len(df)} (FC: {len(df[df['Condition'] == 'FC'])} | CL: {len(df[df['Condition'] == 'CL'])})")

df_valid = df[df['Move_Type'].notna() & (df['Move_Type'] != '')].copy()
df_valid['Move_Type'] = df_valid['Move_Type'].str.strip().str.title()
print(f"With Move_Type data: {len(df_valid)} ({len(df) - len(df_valid)} missing delta values)")

all_moves = ['Concession', 'Selfish', 'Fortunate', 'Unfortunate', 'Nice', 'Silent']
df_main = df_valid[df_valid['Move_Type'].isin(all_moves)]
excluded_moves = df_valid[~df_valid['Move_Type'].isin(all_moves)]
print(f"6 scientific moves: {len(df_main)} (FC: {len(df_main[df_main['Condition'] == 'FC'])} | CL: {len(df_main[df_main['Condition'] == 'CL'])})")
print(f"  Excluded Pa_Gain/Other: {len(excluded_moves)}")
print()

# Global limits
x_min = df_main['Norm_Valence'].quantile(0.001)
x_max = df_main['Norm_Valence'].quantile(0.999)
y_min = df_main['Norm_Arousal'].quantile(0.001)
y_max = df_main['Norm_Arousal'].quantile(0.999)
x_pad = (x_max - x_min) * 0.1
y_pad = (y_max - y_min) * 0.1
XLIM = (x_min - x_pad, x_max + x_pad)
YLIM = (y_min - y_pad, y_max + y_pad)

def add_quadrant_labels(ax):
    ax.text(XLIM[1] - 0.02, YLIM[1] - 0.02, 'Engagement\n(+A, +V)',
            fontsize=10, ha='right', va='top', alpha=0.5, style='italic', color='#555555')
    ax.text(XLIM[0] + 0.02, YLIM[1] - 0.02, 'Unpleasant\n(+A, −V)',
            fontsize=10, ha='left', va='top', alpha=0.5, style='italic', color='#555555')
    ax.text(XLIM[0] + 0.02, YLIM[0] + 0.02, 'Boredom\n(−A, −V)',
            fontsize=10, ha='left', va='bottom', alpha=0.5, style='italic', color='#555555')
    ax.text(XLIM[1] - 0.02, YLIM[0] + 0.02, 'Relaxation\n(−A, +V)',
            fontsize=10, ha='right', va='bottom', alpha=0.5, style='italic', color='#555555')

def format_axes(ax, title):
    ax.axhline(y=0, color='#BBBBBB', linestyle='-', alpha=0.8, linewidth=0.8, zorder=1)
    ax.axvline(x=0, color='#BBBBBB', linestyle='-', alpha=0.8, linewidth=0.8, zorder=1)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xlabel('Valence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Arousal', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.tick_params(labelsize=12, direction='out', width=0.8)
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(1.0)
    add_quadrant_labels(ax)

# ============================================================================
# AMORPHOUS BLOB FUNCTION
# ============================================================================

def create_amorphous_blob(ax, x, y, color, alpha=0.22, grid_size=120, zorder=2, 
                          min_region_points=20, skip_region=None):
    """
    Creates cohesive blob per move type.
    Filters out isolated regions with fewer than min_region_points.
    skip_region: tuple (valence_min, valence_max, arousal_min, arousal_max) to skip
    """
    n = len(x)
    if n < 10:  # Minimum for blob
        return
    
    x_range = (XLIM[0] - 0.1, XLIM[1] + 0.1)
    y_range = (YLIM[0] - 0.1, YLIM[1] + 0.1)
    
    H, xedges, yedges = np.histogram2d(x, y, bins=grid_size, range=[x_range, y_range])
    
    # Find connected regions
    binary_mask = H > 0
    dilated = ndimage.binary_dilation(binary_mask, iterations=3)
    labeled, num_features = ndimage.label(dilated)
    
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
    
    # Process each connected region separately
    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        region_points = H[region_mask].sum()
        
        # Skip regions below the minimum threshold
        if region_points < min_region_points:
            continue
        
        # Skip regions in the specified spatial area
        if skip_region is not None:
            v_min, v_max, a_min, a_max = skip_region
            # Get centroid of this region
            region_X = X.T[region_mask]
            region_Y = Y.T[region_mask]
            region_weights = H[region_mask]
            if region_weights.sum() > 0:
                centroid_x = np.average(region_X, weights=region_weights)
                centroid_y = np.average(region_Y, weights=region_weights)
                # Skip if centroid is in the skip region
                if v_min <= centroid_x <= v_max and a_min <= centroid_y <= a_max:
                    continue
        
        # Create histogram for this region only
        H_region = np.where(region_mask, H, 0)
        
        # Dynamic sigma based on region size
        if region_points < 50:
            sigma = 3.0  # Moderate for small regions
        else:
            sigma = 3.5 + 2.0 * (np.log10(region_points) / np.log10(1500))
            sigma = np.clip(sigma, 3.5, 5.5)
        
        # Dynamic threshold based on region size
        if region_points < 50:
            threshold = 0.20  # Higher threshold = tighter blob
        else:
            threshold = 0.25 - 0.15 * (np.log10(region_points) / np.log10(1500))
            threshold = np.clip(threshold, 0.10, 0.25)
        
        H_smooth = ndimage.gaussian_filter(H_region, sigma=sigma)
        if H_smooth.max() > 0:
            H_smooth = H_smooth / H_smooth.max()
        
        ax.contourf(X, Y, H_smooth.T, levels=[threshold, 1.0],
                    colors=[to_rgba(color, alpha)], zorder=zorder)
        ax.contour(X, Y, H_smooth.T, levels=[threshold],
                   colors=[to_rgba(color, 0.6)], linewidths=1.5, zorder=zorder+1)

# ============================================================================
# MAIN FIGURE
# ============================================================================

def create_final_circumplex():
    """Final publication-ready circumplex."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor='white')
    
    for idx, (condition, title) in enumerate([
        ('FC', 'FC Rounds'),
        ('CL', 'CL Rounds')
    ]):
        ax = axes[idx]
        ax.set_facecolor('white')
        cond_data = df_main[df_main['Condition'] == condition]
        
        legend_handles = []
        
        # Calculate cluster sizes for dynamic thresholds
        move_sizes = {}
        for move_type in all_moves:
            move_data = cond_data[cond_data['Move_Type'] == move_type]
            move_sizes[move_type] = len(move_data)
        
        # FIXED order for consistency between FC and CL
        # Largest types in back, smallest in front (same across both conditions)
        fixed_order = ['Selfish', 'Concession', 'Fortunate', 'Unfortunate', 'Nice', 'Silent']
        
        # Draw blobs - use fixed order for visual consistency
        for i, move_type in enumerate(fixed_order):
            move_data = cond_data[cond_data['Move_Type'] == move_type]
            n = len(move_data)
            if n < 10:
                continue
            
            x = move_data['Norm_Valence'].values
            y = move_data['Norm_Arousal'].values
            color = MOVE_COLORS[move_type]
            zorder = 2 + i * 2
            
            # Dynamic per-region threshold based on cluster size
            if n < 50:
                min_pts = 10
            elif n < 200:
                min_pts = 20
            else:
                min_pts = 30
            
            create_amorphous_blob(ax, x, y, color, alpha=MOVE_ALPHA.get(move_type, 0.20), 
                                  zorder=zorder, min_region_points=min_pts)
        
        # Draw scatter points - use fixed order for consistency
        for i, move_type in enumerate(fixed_order):
            move_data = cond_data[cond_data['Move_Type'] == move_type]
            n = len(move_data)
            if n < 3:
                continue
            
            x = move_data['Norm_Valence'].values
            y = move_data['Norm_Arousal'].values
            color = MOVE_COLORS[move_type]
            # Larger z-order gaps: smallest cluster gets zorder=100+
            zorder = 20 + i * 15  # 15 gap instead of 2
            
            # Slightly larger and more opaque points for better visibility
            ax.scatter(x, y, c=color, alpha=0.65, s=18, edgecolors='none', zorder=zorder)
        
        # Legend entries in display order
        for move_type in all_moves:
            n = len(cond_data[cond_data['Move_Type'] == move_type])
            if n < 3:
                continue
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=MOVE_COLORS[move_type],
                       markersize=9, markeredgewidth=0, label=f'{move_type} (n={n})')
            )
        
        # Draw centroids - ALWAYS on top (highest z-order)
        for i, move_type in enumerate(fixed_order):
            move_data = cond_data[cond_data['Move_Type'] == move_type]
            n = len(move_data)
            if n < 3:  # Show centroid if at least 3 points
                continue
            
            cx = move_data['Norm_Valence'].mean()
            cy = move_data['Norm_Arousal'].mean()
            color = MOVE_COLORS[move_type]
            # Centroids are HIGHEST z-order (200+) - always in front
            zorder = 200 + i * 2
            
            ax.scatter(cx, cy, s=280, c=color, edgecolors='black', linewidths=1.5, zorder=zorder)
        
        # Legend placement: FC = lower left, CL = lower right
        if condition == 'FC':
            legend_loc = 'lower left'
        else:
            legend_loc = 'lower right'
        
        ax.legend(handles=legend_handles, loc=legend_loc, fontsize=11,
                  framealpha=0.95, handletextpad=0.4, borderpad=0.5,
                  edgecolor='#CCCCCC', fancybox=False)
        
        format_axes(ax, title)
    
    # Panel labels
    axes[0].text(-0.08, 1.02, 'A', transform=axes[0].transAxes, 
                 fontsize=18, fontweight='bold', va='bottom')
    axes[1].text(-0.08, 1.02, 'B', transform=axes[1].transAxes,
                 fontsize=18, fontweight='bold', va='bottom')
    
    plt.tight_layout(w_pad=3)
    
    plt.savefig(f'{OUTPUT_DIR}/MoveAffect_Circumplex_Final.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/MoveAffect_Circumplex_Final.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR}/MoveAffect_Circumplex_Final.png")
    print(f"Saved: {OUTPUT_DIR}/MoveAffect_Circumplex_Final.pdf")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CREATING FINAL PUBLICATION CIRCUMPLEX v2")
    print("=" * 60)
    
    create_final_circumplex()
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
