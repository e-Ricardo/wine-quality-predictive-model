"""
src/01_eda.py
Step 1 — Exploratory Data Analysis
Run: python src/01_eda.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import save_fig, make_dirs, print_banner

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_raw_data():
    """Download and combine red + white wine datasets."""
    make_dirs()
    print("📥 Loading UCI Wine Quality dataset...")

    try:
        from ucimlrepo import fetch_ucirepo
        wine = fetch_ucirepo(id=186)
        X = wine.data.features.copy()
        y = wine.data.targets.squeeze().copy()
        print(f"  ✅ Loaded via ucimlrepo: {len(X)} rows, {X.shape[1]} features")
    except Exception:
        print("  ⚠️  ucimlrepo failed — falling back to direct CSV download...")
        red   = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",   sep=';')
        white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
        red['wine_type']   = 0
        white['wine_type'] = 1
        df = pd.concat([red, white], ignore_index=True)
        X = df.drop('quality', axis=1)
        y = df['quality']
        print(f"  ✅ Loaded via CSV: {len(X)} rows, {X.shape[1]} features")

    # Save raw data
    df_full = X.copy()
    df_full['quality'] = y.values
    df_full.to_csv('data/wine_raw.csv', index=False)
    print("  💾 Saved: data/wine_raw.csv")
    return X, y

# ─────────────────────────────────────────────────────────────
# 2. BASIC STATS
# ─────────────────────────────────────────────────────────────

def print_stats(X, y):
    print_banner("BASIC STATISTICS")
    df = X.copy(); df['quality'] = y.values
    print(f"\n  Shape        : {df.shape}")
    print(f"  Missing vals : {df.isnull().sum().sum()}")
    print(f"\n  Quality distribution:\n{y.value_counts().sort_index().to_string()}")
    print(f"\n  Feature statistics:\n{df.describe().round(3).to_string()}")

# ─────────────────────────────────────────────────────────────
# 3. EDA PLOTS
# ─────────────────────────────────────────────────────────────

def plot_eda(X, y):
    print_banner("GENERATING EDA PLOTS")
    df = X.copy(); df['quality'] = y.values

    # ── Plot 1: Quality distribution + class pie ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Wine Quality — Distribution', fontsize=14, fontweight='bold')

    counts = y.value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color='steelblue', edgecolor='black', width=0.6)
    axes[0].set_xlabel('Quality Score'); axes[0].set_ylabel('Count')
    axes[0].set_title('Raw Quality Score Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    for i, (q, c) in enumerate(counts.items()):
        axes[0].text(q, c + 20, str(c), ha='center', fontsize=9)

    class_labels = ['Low (≤5)', 'Medium (6)', 'High (≥7)']
    class_counts = [
        (y <= 5).sum(),
        (y == 6).sum(),
        (y >= 7).sum(),
    ]
    axes[1].pie(class_counts, labels=class_labels, autopct='%1.1f%%',
                colors=['#d62728', '#1f77b4', '#2ca02c'], startangle=140)
    axes[1].set_title('3-Class Distribution (Low / Medium / High)')
    plt.tight_layout()
    save_fig('01_quality_distribution')

    # ── Plot 2: Correlation heatmap ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap='RdYlGn', center=0,
                annot=True, fmt='.2f', linewidths=0.4, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('02_correlation_heatmap')

    # ── Plot 3: Feature correlation with quality (bar) ────────
    fig, ax = plt.subplots(figsize=(9, 6))
    corr_q = corr['quality'].drop('quality').sort_values()
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in corr_q]
    corr_q.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title('Feature Correlation with Quality Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pearson Correlation Coefficient')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig('03_quality_correlations')

    # ── Plot 4: Key features vs quality (boxplots) ────────────
    key_features = ['alcohol', 'volatile_acidity', 'sulphates', 'citric_acid',
                'density', 'chlorides']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Key Features by Quality Score', fontsize=14, fontweight='bold')
    for ax, feat in zip(axes.flatten(), key_features):
        df.boxplot(column=feat, by='quality', ax=ax, notch=False)
        ax.set_title(feat.title())
        ax.set_xlabel('Quality Score')
        ax.set_ylabel(feat)
    plt.suptitle('Key Features by Quality Score', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('04_features_vs_quality')

    # ── Plot 5: Histograms for all features ───────────────────
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
    for i, (ax, col) in enumerate(zip(axes.flatten(), df.columns)):
        ax.hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)
    # hide unused axes
    for ax in axes.flatten()[len(df.columns):]:
        ax.set_visible(False)
    plt.tight_layout()
    save_fig('05_feature_histograms')

    # ── Plot 6: Pairplot of top 5 features ────────────────────
    top5 = corr['quality'].drop('quality').abs().nlargest(4).index.tolist() + ['quality']
    pair_df = df[top5].copy()
    pair_df['quality_class'] = pd.cut(df['quality'], bins=[0,5,6,10],
                                       labels=['Low','Medium','High'])
    g = sns.pairplot(pair_df, hue='quality_class',
                     palette={'Low':'#d62728','Medium':'#1f77b4','High':'#2ca02c'},
                     plot_kws={'alpha': 0.5, 's': 20})
    g.figure.suptitle('Pairplot — Top 4 Features vs Quality', y=1.02, fontsize=13)
    save_fig('06_pairplot_top_features')

    print("\n  ✅ All 6 EDA figures saved to figures/")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    X, y = load_raw_data()
    print_stats(X, y)
    plot_eda(X, y)
    print("\n✅ Step 1 — EDA complete!\n")
