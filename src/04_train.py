"""
src/04_train.py
Step 4 — Model Training
Trains MLP, 1D-CNN, LSTM.  Saves trained models + histories.
Run: python src/04_train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import importlib.util, os as _os
_spec = importlib.util.spec_from_file_location("models", _os.path.join(_os.path.dirname(__file__), "03_models.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
build_mlp, build_cnn1d, build_lstm = _mod.build_mlp, _mod.build_cnn1d, _mod.build_lstm

from utils.helpers import make_dirs, print_banner, plot_training_history

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

EPOCHS     = 100
BATCH_SIZE = 64
LR         = 0.001

# ─────────────────────────────────────────────────────────────
# LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────────────────────

def load_data():
    print("📂 Loading preprocessed data...")
    return {
        'X_train':    np.load('data/X_train.npy'),
        'X_val':      np.load('data/X_val.npy'),
        'X_test':     np.load('data/X_test.npy'),
        'X_train_3d': np.load('data/X_train_3d.npy'),
        'X_val_3d':   np.load('data/X_val_3d.npy'),
        'X_test_3d':  np.load('data/X_test_3d.npy'),
        'y_train_oh': np.load('data/y_train_oh.npy'),
        'y_val_oh':   np.load('data/y_val_oh.npy'),
    }

# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

def get_callbacks(model_name: str):
    return [
        EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=f'models/{model_name}_best.keras',
            monitor='val_accuracy', save_best_only=True, verbose=0
        ),
    ]

# ─────────────────────────────────────────────────────────────
# TRAIN ONE MODEL
# ─────────────────────────────────────────────────────────────

def train_model(model, model_name, X_train, X_val, y_train_oh, y_val_oh):
    print(f"\n🔧 Training {model_name}...")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )
    # Save full model
    model.save(f'models/{model_name}_final.keras')
    print(f"  💾 Saved: models/{model_name}_final.keras")

    # Save history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(f'models/{model_name}_history.json', 'w') as f:
        json.dump(hist_dict, f, indent=2)

    best_val = max(history.history['val_accuracy'])
    print(f"  ✅ Best val accuracy: {best_val:.4f}")
    plot_training_history(history, model_name)

    return history

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    make_dirs()
    d = load_data()
    n_features = d['X_train'].shape[1]

    print_banner("MODEL TRAINING")

    histories = {}

    # ── MLP ──────────────────────────────────────────────────
    mlp = build_mlp(n_features, lr=LR)
    histories['MLP'] = train_model(
        mlp, 'MLP',
        d['X_train'], d['X_val'],
        d['y_train_oh'], d['y_val_oh']
    )

    # ── 1D-CNN ───────────────────────────────────────────────
    cnn = build_cnn1d(n_features, lr=LR)
    histories['CNN'] = train_model(
        cnn, 'CNN',
        d['X_train_3d'], d['X_val_3d'],
        d['y_train_oh'], d['y_val_oh']
    )

    # ── LSTM ─────────────────────────────────────────────────
    lstm = build_lstm(n_features, lr=LR)
    histories['LSTM'] = train_model(
        lstm, 'LSTM',
        d['X_train_3d'], d['X_val_3d'],
        d['y_train_oh'], d['y_val_oh']
    )

    print("\n✅ Step 4 — All models trained & saved!\n")
