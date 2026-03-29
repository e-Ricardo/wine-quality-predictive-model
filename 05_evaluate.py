"""
src/05_evaluate.py
Step 5 — Model Evaluation
Loads saved models, generates all evaluation metrics & plots.
Run: python src/05_evaluate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

from utils.helpers import save_fig, print_banner, CLASS_NAMES

# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────────────────────

def load_all():
    print("📂 Loading test data & saved models...")
    data = {
        'X_test':    np.load('data/X_test.npy'),
        'X_test_3d': np.load('data/X_test_3d.npy'),
        'y_test':    np.load('data/y_test.npy'),
        'y_test_oh': np.load('data/y_test_oh.npy'),
    }
    models = {}
    for name in ['MLP', 'CNN', 'LSTM']:
        path = f'models/{name}_final.keras'
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)
            print(f"  ✅ Loaded {name}")
        else:
            print(f"  ⚠️  {path} not found — run 04_train.py first")
    return data, models

# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

def predict_all(models, data):
    results = {}
    for name, model in models.items():
        X = data['X_test_3d'] if name in ['CNN', 'LSTM'] else data['X_test']
        y_prob = model.predict(X, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        results[name] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'acc':    accuracy_score(data['y_test'], y_pred),
            'f1':     f1_score(data['y_test'], y_pred, average='weighted'),
        }
    return results

# ─────────────────────────────────────────────────────────────
# PRINT METRICS
# ─────────────────────────────────────────────────────────────

def print_metrics(results, y_test):
    print_banner("EVALUATION RESULTS")
    for name, res in results.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        print(f"  Test Accuracy : {res['acc']:.4f}")
        print(f"  Weighted F1   : {res['f1']:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, res['y_pred'], target_names=CLASS_NAMES))

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    fig.suptitle('Confusion Matrices — Test Set', fontsize=14, fontweight='bold')
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
        )
        ax.set_title(f"{name}\nAcc: {res['acc']:.4f}  |  F1: {res['f1']:.4f}")
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

    plt.tight_layout()
    save_fig('08_confusion_matrices')


def plot_model_comparison(results):
    names = list(results.keys())
    accs  = [results[n]['acc'] for n in names]
    f1s   = [results[n]['f1']  for n in names]

    x = np.arange(len(names))
    width = 0.35
    colors_acc = ['#4C72B0', '#DD8452', '#55A868']
    colors_f1  = ['#64A5D3', '#EBA878', '#7DBD8E']

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color=colors_acc, edgecolor='black')
    bars2 = ax.bar(x + width/2, f1s,  width, label='Weighted F1', color=colors_f1, edgecolor='black')

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Test Set Performance', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_fig('09_model_comparison')


def plot_all_training_curves():
    """Overlay all models' val_accuracy on one chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Training History — All Models', fontsize=13, fontweight='bold')
    colors = {'MLP': 'steelblue', 'CNN': 'orange', 'LSTM': 'green'}

    for name in ['MLP', 'CNN', 'LSTM']:
        path = f'models/{name}_history.json'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            h = json.load(f)
        ax1.plot(h['val_loss'],     color=colors[name], linewidth=2, label=f'{name} val loss')
        ax1.plot(h['loss'],         color=colors[name], linewidth=1, linestyle='--', alpha=0.5)
        ax2.plot(h['val_accuracy'], color=colors[name], linewidth=2, label=f'{name} val acc')
        ax2.plot(h['accuracy'],     color=colors[name], linewidth=1, linestyle='--', alpha=0.5)

    for ax, title, ylabel in [(ax1,'Validation Loss','Loss'),(ax2,'Validation Accuracy','Accuracy')]:
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig('10_all_training_curves')


def plot_class_accuracy(results, y_test):
    """Per-class accuracy for each model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        ax.bar(x + i * width, per_class_acc, width, label=name, edgecolor='black')

    ax.set_xticks(x + width); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.2); ax.set_ylabel('Recall (per-class accuracy)')
    ax.set_title('Per-Class Recall — All Models', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig('11_per_class_accuracy')


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    data, models = load_all()

    if not models:
        print("❌ No models found. Run src/04_train.py first.")
        sys.exit(1)

    results = predict_all(models, data)
    print_metrics(results, data['y_test'])
    plot_confusion_matrices(results, data['y_test'])
    plot_model_comparison(results)
    plot_all_training_curves()
    plot_class_accuracy(results, data['y_test'])

    print("\n✅ Step 5 — Evaluation complete! Check figures/\n")
