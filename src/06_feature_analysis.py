"""
src/06_feature_analysis.py
Step 6 — Feature Importance Analysis
PCA + SHAP (optional) + correlation analysis
Run: python src/06_feature_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import joblib

from utils.helpers import save_fig, print_banner, CLASS_NAMES

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv('data/wine_raw.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test  = np.load('data/X_test.npy')
    y_test  = np.load('data/y_test.npy')
    return X, y, X_train, y_train, X_test, y_test

# ─────────────────────────────────────────────────────────────
# 1. PCA ANALYSIS
# ─────────────────────────────────────────────────────────────

def plot_pca(X_train, feature_names):
    print("  Running PCA...")
    pca = PCA()
    pca.fit(X_train)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cumvar >= 0.95) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('PCA Analysis', fontsize=13, fontweight='bold')

    # Scree plot
    ax = axes[0]
    ax.bar(range(1, len(pca.explained_variance_ratio_)+1),
           pca.explained_variance_ratio_, color='steelblue', edgecolor='black', alpha=0.8)
    ax.plot(range(1, len(cumvar)+1), cumvar, 'r-o', markersize=5, label='Cumulative variance')
    ax.axhline(0.95, color='green', linestyle='--', alpha=0.7, label=f'95% threshold → {n_95} PCs')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'Scree Plot  (95% variance with {n_95} PCs)')
    ax.legend(); ax.grid(alpha=0.3)

    # PC1 vs PC2 colored by quality class
    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_train)
    y_train = np.load('data/y_train.npy')
    colors = {0:'#d62728', 1:'#1f77b4', 2:'#2ca02c'}
    ax = axes[1]
    for cls, name in enumerate(CLASS_NAMES):
        mask = y_train == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[cls], label=name, alpha=0.3, s=10)
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA — 2D Projection by Quality Class')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig('12_pca_analysis')
    print(f"  ✅ PCA: {n_95} components explain 95% of variance")


# ─────────────────────────────────────────────────────────────
# 2. RANDOM FOREST FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def plot_rf_importance(X_train, y_train, X_test, y_test, feature_names):
    print("  Fitting Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    print(f"  RF Test Accuracy: {acc:.4f}")

    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
    importances.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
    ax.set_title('Random Forest — Feature Importances', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Decrease in Impurity')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig('13_rf_feature_importance')

    joblib.dump(rf, 'models/random_forest.pkl')
    print("  💾 RF model saved: models/random_forest.pkl")
    return rf


# ─────────────────────────────────────────────────────────────
# 3. SHAP VALUES (optional — skip gracefully if not installed)
# ─────────────────────────────────────────────────────────────

def plot_shap(rf, X_train, feature_names):
    try:
        import shap
        print("  Computing SHAP values...")
        explainer = shap.TreeExplainer(rf)
        # Use a sample for speed
        idx = np.random.choice(len(X_train), size=min(500, len(X_train)), replace=False)
        shap_values = explainer.shap_values(X_train[idx])

        # Summary plot (class 2 = High quality)
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values[2], X_train[idx],
            feature_names=feature_names,
            show=False, plot_type='dot'
        )
        plt.title('SHAP Summary — High Quality Class', fontsize=13, fontweight='bold')
        save_fig('14_shap_summary')
        print("  ✅ SHAP plot saved")

    except ImportError:
        print("  ⚠️  shap not installed — skipping SHAP analysis")
        print("     Install with: pip install shap")


# ─────────────────────────────────────────────────────────────
# 4. CORRELATION WITH QUALITY (from raw data)
# ─────────────────────────────────────────────────────────────

def plot_feature_quality_violin(X_raw, y_raw):
    print("  Plotting violin plots...")
    df = X_raw.copy()
    df['quality_class'] = pd.cut(y_raw, bins=[0,5,6,10],
                                  labels=['Low','Medium','High'])
    top4 = X_raw.corr() if False else None
    features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Feature Distributions by Quality Class (Violin)', fontsize=13, fontweight='bold')
    palette = {'Low':'#d62728','Medium':'#1f77b4','High':'#2ca02c'}

    for ax, feat in zip(axes, features):
        sns.violinplot(data=df, x='quality_class', y=feat, ax=ax,
                       palette=palette, order=['Low','Medium','High'])
        ax.set_title(feat.title())
        ax.set_xlabel('Quality Class')

    plt.tight_layout()
    save_fig('15_violin_plots')


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print_banner("FEATURE IMPORTANCE ANALYSIS")
    X_raw, y_raw, X_train, y_train, X_test, y_test = load_data()
    feature_names = list(X_raw.columns)

    plot_pca(X_train, feature_names)
    rf = plot_rf_importance(X_train, y_train, X_test, y_test, feature_names)
    plot_shap(rf, X_train, feature_names)
    plot_feature_quality_violin(X_raw, y_raw)

    print("\n✅ Step 6 — Feature analysis complete!\n")
