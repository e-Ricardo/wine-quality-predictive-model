"""
src/03_models.py
Step 3 — Model Definitions
MLP, 1D-CNN, LSTM, XGBoost
Run: python src/03_models.py   (prints summaries)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten,
    LSTM, Input
)
from tensorflow.keras.optimizers import Adam
from utils.helpers import print_banner


# ─────────────────────────────────────────────────────────────
# MODEL 1 — MLP (Baseline Feedforward Network)
# ─────────────────────────────────────────────────────────────

def build_mlp(n_features: int, n_classes: int = 3, lr: float = 0.001) -> tf.keras.Model:
    """
    Feedforward MLP baseline.
    Input shape: (n_features,)
    """
    model = Sequential([
        Input(shape=(n_features,)),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(n_classes, activation='softmax'),
    ], name='MLP')

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────────────────────
# MODEL 2 — 1D-CNN
# ─────────────────────────────────────────────────────────────

def build_cnn1d(n_features: int, n_classes: int = 3, lr: float = 0.001) -> tf.keras.Model:
    """
    1D Convolutional Neural Network for local feature correlations.
    Input shape: (n_features, 1)
    """
    model = Sequential([
        Input(shape=(n_features, 1)),

        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(n_classes, activation='softmax'),
    ], name='CNN1D')

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────────────────────
# MODEL 3 — LSTM
# ─────────────────────────────────────────────────────────────

def build_lstm(n_features: int, n_classes: int = 3, lr: float = 0.001) -> tf.keras.Model:
    """
    LSTM network for modeling sequential feature dependencies.
    Input shape: (n_features, 1)
    """
    model = Sequential([
        Input(shape=(n_features, 1)),

        LSTM(128, return_sequences=True),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),

        Dense(n_classes, activation='softmax'),
    ], name='LSTM')

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────────────────────
# FACTORY — get model by name
# ─────────────────────────────────────────────────────────────

def get_model(name: str, n_features: int, n_classes: int = 3, lr: float = 0.001):
    """
    Factory function.
    Usage:
        model = get_model('MLP', n_features=11)
        model = get_model('CNN', n_features=11)
        model = get_model('LSTM', n_features=11)
    """
    builders = {
        'MLP':  build_mlp,
        'CNN':  build_cnn1d,
        'LSTM': build_lstm,
    }
    key = name.upper().replace('1D-', '').replace('1D_', '')
    if key not in builders:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(builders.keys())}")
    return builders[key](n_features, n_classes, lr)


# ─────────────────────────────────────────────────────────────
# MAIN — print summaries
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print_banner("MODEL ARCHITECTURES")

    n_feat = 11   # wine dataset features
    for name, build_fn in [('MLP', build_mlp), ('1D-CNN', build_cnn1d), ('LSTM', build_lstm)]:
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        m = build_fn(n_feat)
        m.summary()

    print("\n✅ Step 3 — Model definitions ready!\n")
