"""
src/02_preprocessing.py
Step 2 — Data Preprocessing
Handles: normalization, class mapping, SMOTE, train/val/test split
Run: python src/02_preprocessing.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
import joblib

from utils.helpers import make_dirs, save_fig, print_banner, map_quality_to_class, CLASS_NAMES

# ─────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────────────────────

def load_raw():
    make_dirs()
    df = pd.read_csv('data/wine_raw.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y


# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess(X, y):
    print_banner("PREPROCESSING PIPELINE")

    # 2a. Map quality → 3 classes
    y_class = y.map(map_quality_to_class)
    print(f"\n  Class distribution (BEFORE SMOTE):")
    for i, name in enumerate(CLASS_NAMES):
        n = (y_class == i).sum()
        print(f"    {name}: {n} ({100*n/len(y_class):.1f}%)")

    # 2b. Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"\n  ✅ StandardScaler fitted & saved → models/scaler.pkl")

    # 2c. SMOTE oversampling (on full data before split to show effect)
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_scaled, y_class)
    print(f"\n  Class distribution (AFTER SMOTE):")
    for i, name in enumerate(CLASS_NAMES):
        n = (y_bal == i).sum()
        print(f"    {name}: {n} ({100*n/len(y_bal):.1f}%)")

    # 2d. Train / Val / Test  (70 / 15 / 15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_bal, y_bal, test_size=0.15, random_state=42, stratify=y_bal)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

    print(f"\n  Dataset splits:")
    print(f"    Train : {X_train.shape[0]} samples")
    print(f"    Val   : {X_val.shape[0]}  samples")
    print(f"    Test  : {X_test.shape[0]}  samples")

    # 2e. One-hot encode labels
    n_classes = 3
    y_train_oh = to_categorical(y_train, n_classes)
    y_val_oh   = to_categorical(y_val,   n_classes)
    y_test_oh  = to_categorical(y_test,  n_classes)

    # 2f. Reshape for CNN / LSTM  (add channel dim)
    n_feat = X_train.shape[1]
    X_train_3d = X_train.reshape(-1, n_feat, 1)
    X_val_3d   = X_val.reshape(-1,   n_feat, 1)
    X_test_3d  = X_test.reshape(-1,  n_feat, 1)

    # 2g. Save everything
    np.save('data/X_train.npy',    X_train);    np.save('data/y_train.npy',    y_train)
    np.save('data/X_val.npy',      X_val);      np.save('data/y_val.npy',      y_val)
    np.save('data/X_test.npy',     X_test);     np.save('data/y_test.npy',     y_test)
    np.save('data/X_train_3d.npy', X_train_3d)
    np.save('data/X_val_3d.npy',   X_val_3d)
    np.save('data/X_test_3d.npy',  X_test_3d)
    np.save('data/y_train_oh.npy', y_train_oh)
    np.save('data/y_val_oh.npy',   y_val_oh)
    np.save('data/y_test_oh.npy',  y_test_oh)
    print("\n  💾 All split arrays saved to data/")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'X_train_3d': X_train_3d, 'X_val_3d': X_val_3d, 'X_test_3d': X_test_3d,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_oh': y_train_oh, 'y_val_oh': y_val_oh, 'y_test_oh': y_test_oh,
        'n_features': n_feat, 'n_classes': n_classes, 'scaler': scaler,
        'feature_names': list(X.columns),
    }


# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_preprocessing(X, y, data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Preprocessing Results', fontsize=14, fontweight='bold')

    # Before SMOTE
    y_class = y.map(map_quality_to_class)
    counts_before = [( y_class == i).sum() for i in range(3)]
    axes[0].bar(CLASS_NAMES, counts_before, color=['#d62728','#1f77b4','#2ca02c'], edgecolor='black')
    axes[0].set_title('Class Distribution — Before SMOTE')
    axes[0].set_ylabel('Count')
    for i, c in enumerate(counts_before):
        axes[0].text(i, c + 10, str(c), ha='center')

    # After SMOTE
    y_bal = np.load('data/y_train.npy')
    y_val = np.load('data/y_val.npy')
    y_test_arr = np.load('data/y_test.npy')
    y_all = np.concatenate([y_bal, y_val, y_test_arr])
    counts_after = [(y_all == i).sum() for i in range(3)]
    axes[1].bar(CLASS_NAMES, counts_after, color=['#d62728','#1f77b4','#2ca02c'], edgecolor='black')
    axes[1].set_title('Class Distribution — After SMOTE')
    axes[1].set_ylabel('Count')
    for i, c in enumerate(counts_after):
        axes[1].text(i, c + 10, str(c), ha='center')

    # Feature distributions before/after scaling
    X_raw_vals  = X.values[:, 0]           # first feature (fixed acidity)
    X_scal_vals = data['X_train'][:, 0]
    axes[2].hist(X_raw_vals,  bins=40, alpha=0.6, label='Original',   color='orange')
    axes[2].hist(X_scal_vals, bins=40, alpha=0.6, label='Normalized', color='steelblue')
    axes[2].set_title(f'Fixed Acidity — Before vs After Normalization')
    axes[2].set_xlabel('Value'); axes[2].set_ylabel('Count')
    axes[2].legend()

    plt.tight_layout()
    save_fig('07_preprocessing_results')


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    X, y = load_raw()
    data = preprocess(X, y)
    plot_preprocessing(X, y, data)
    print("\n✅ Step 2 — Preprocessing complete!\n")
