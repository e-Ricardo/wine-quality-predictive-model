"""
utils/helpers.py
Shared utility functions used across all pipeline steps.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Directory setup ────────────────────────────────────────────────────────

def make_dirs():
    """Create all required output directories."""
    for d in ['data', 'figures', 'models']:
        os.makedirs(d, exist_ok=True)

# ── Plotting helpers ───────────────────────────────────────────────────────

def save_fig(name, dpi=150):
    """Save current matplotlib figure to figures/ folder."""
    os.makedirs('figures', exist_ok=True)
    path = f'figures/{name}.png'
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Saved: {path}')


def plot_training_history(history, model_name):
    """Plot and save loss + accuracy curves for one model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{model_name} — Training History', fontsize=13, fontweight='bold')

    # Loss
    ax1.plot(history.history['loss'],     label='Train Loss', color='steelblue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss',   color='orange',    linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend(); ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(history.history['accuracy'],     label='Train Acc', color='green',  linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Acc',   color='crimson', linewidth=2, linestyle='--')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_fig(f'training_{model_name.lower().replace(" ", "_")}')


# ── Label helpers ──────────────────────────────────────────────────────────

CLASS_NAMES = ['Low (≤5)', 'Medium (6)', 'High (≥7)']

def map_quality_to_class(score):
    """Map raw quality score (0-10) → 3-class label (0/1/2)."""
    if score <= 5:   return 0   # Low
    elif score == 6: return 1   # Medium
    else:            return 2   # High

# ── Metrics summary ────────────────────────────────────────────────────────

def print_banner(title):
    width = 60
    print('\n' + '=' * width)
    print(f'  {title}')
    print('=' * width)
