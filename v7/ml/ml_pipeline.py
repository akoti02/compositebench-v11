"""
V7 CompositeBench ML pipeline — trains surrogate models on 1000 CFRP crack-damage sims.

Run: python v7_ml_pipeline.py
Expects simulation_results_v7.csv alongside this file; figures land in ~/Downloads/figures_v7/.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import copy
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    KFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor, XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: catboost not installed. Install with: pip install catboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Install with: pip install shap")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed. Install with: pip install optuna")

try:
    from mapie.regression import CrossConformalRegressor
    HAS_MAPIE = True
except ImportError:
    HAS_MAPIE = False
    print("WARNING: mapie not installed. Install with: pip install mapie")


# config

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'simulation_results_v7.csv'
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K_FOLDS = 5
N_REPEATS = 3
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 30

OPTUNA_N_TRIALS = 50

PLATE_LENGTH = 100.0   # mm
PLATE_WIDTH = 50.0     # mm
MAX_DEFECTS = 5

FIG_DIR = os.path.join(os.path.expanduser('~'), 'Downloads', 'figures_v7')

REG_TARGETS = [
    'tsai_wu_index',
    'max_mises',
    'max_hashin_ft',
    'max_hashin_fc',
    'max_hashin_mt',
    'max_hashin_mc',
]

CLF_TARGETS = [
    'failed_tsai_wu',
    'failed_hashin',
]


# data loading

def load_data(filepath):
    print(f"Loading data from: {os.path.basename(filepath)}")
    print("-" * 50)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\nERROR: File not found: {filepath}\n"
            f"Please ensure simulation_results_v7.csv exists."
        )

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    for col in df.columns:
        if col != 'sim_id':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'solver_completed' in df.columns:
        n_before = len(df)
        df = df[df['solver_completed'] == 1].copy()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"Removed {n_removed} failed simulations (solver_completed=0)")

    original_count = len(df)
    key_cols = [c for c in REG_TARGETS + CLF_TARGETS if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols)
    removed = original_count - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with missing target values")

    print(f"\nFinal dataset: {len(df)} samples")
    return df


# feature engineering

def engineer_features(df):
    df = df.copy()
    n_engineered = 0

    for i in range(1, MAX_DEFECTS + 1):
        hl_col = f'defect{i}_half_length'
        if hl_col in df.columns:
            df[f'defect{i}_norm_length'] = df[hl_col] * 2.0 / PLATE_WIDTH
            n_engineered += 1

    for i in range(1, MAX_DEFECTS + 1):
        x_col = f'defect{i}_x'
        y_col = f'defect{i}_y'
        if x_col in df.columns and y_col in df.columns:
            df[f'defect{i}_norm_x'] = (df[x_col] - PLATE_LENGTH / 2) / PLATE_LENGTH
            df[f'defect{i}_norm_y'] = (df[y_col] - PLATE_WIDTH / 2) / PLATE_WIDTH
            df[f'defect{i}_dist_center'] = np.sqrt(
                df[f'defect{i}_norm_x']**2 + df[f'defect{i}_norm_y']**2
            )
            n_engineered += 3

    total_crack_area = pd.Series(0.0, index=df.index)
    for i in range(1, MAX_DEFECTS + 1):
        hl_col = f'defect{i}_half_length'
        w_col = f'defect{i}_width'
        if hl_col in df.columns and w_col in df.columns:
            total_crack_area += df[hl_col] * 2.0 * df[w_col]
    plate_area = PLATE_LENGTH * PLATE_WIDTH
    df['crack_area_ratio'] = total_crack_area / plate_area
    n_engineered += 1

    total_crack_len = pd.Series(0.0, index=df.index)
    for i in range(1, MAX_DEFECTS + 1):
        hl_col = f'defect{i}_half_length'
        if hl_col in df.columns:
            total_crack_len += df[hl_col] * 2.0
    df['total_crack_length_norm'] = total_crack_len / PLATE_WIDTH
    n_engineered += 1

    if 'pressure_x' in df.columns and 'pressure_y' in df.columns:
        df['load_ratio'] = df['pressure_y'] / (df['pressure_x'] + 1e-6)
        df['total_pressure'] = np.sqrt(df['pressure_x']**2 + df['pressure_y']**2)
        n_engineered += 2

    if 'total_thickness' in df.columns:
        df['thickness_aspect'] = df['total_thickness'] / PLATE_WIDTH
        n_engineered += 1

    max_crack_len = pd.Series(0.0, index=df.index)
    for i in range(1, MAX_DEFECTS + 1):
        hl_col = f'defect{i}_half_length'
        if hl_col in df.columns:
            max_crack_len = np.maximum(max_crack_len, df[hl_col] * 2.0)
    df['max_crack_width_ratio'] = max_crack_len / PLATE_WIDTH
    n_engineered += 1

    # crack-vs-load alignment per defect
    for i in range(1, MAX_DEFECTS + 1):
        ang_col = f'defect{i}_angle'
        if ang_col in df.columns and 'pressure_x' in df.columns and 'pressure_y' in df.columns:
            load_angle = np.degrees(np.arctan2(df['pressure_y'], df['pressure_x'] + 1e-6))
            angle_diff = np.abs(df[ang_col] - load_angle) % 180
            df[f'defect{i}_load_alignment'] = np.minimum(angle_diff, 180 - angle_diff) / 90.0
            n_engineered += 1

    print(f"  Engineered {n_engineered} new physics-based features")
    return df


# column detection

def detect_columns(df):
    all_cols = list(df.columns)
    skip_cols = {'sim_id', 'solver_completed'}

    reg_targets = [t for t in REG_TARGETS if t in all_cols]
    clf_targets = [t for t in CLF_TARGETS if t in all_cols]

    output_keywords = [
        'max_mises', 'max_s11', 'min_s11', 'max_s22', 'min_s22', 'max_s12',
        'tsai_wu', 'hashin', 'max_disp', 'n_elements', 'failed',
        'max_mises_defect', 'max_sdeg', 'n_damaged_elements',
        'solver_completed',
    ]

    known_outputs = set(reg_targets + clf_targets)
    output_cols = set()
    for col in all_cols:
        for kw in output_keywords:
            if kw in col and col not in skip_cols:
                output_cols.add(col)

    features = [
        col for col in all_cols
        if col not in skip_cols
        and col not in known_outputs
        and col not in output_cols
    ]

    detected = {
        'features': features,
        'reg_targets': reg_targets,
        'clf_targets': clf_targets,
        'output_cols': list(output_cols - known_outputs),
    }

    print(f"\n{'='*60}")
    print("COLUMN DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"  Total columns:              {len(all_cols)}")
    print(f"  Input features ({len(features):2d}):        {features}")
    print(f"  Regression targets ({len(reg_targets)}):    {reg_targets}")
    print(f"  Classification targets ({len(clf_targets)}): {clf_targets}")
    print(f"  Other outputs ({len(detected['output_cols'])}):         {detected['output_cols']}")
    print(f"{'='*60}\n")

    return detected


# preprocessing

def preprocess(df, feature_names, target_name, task='regression'):
    X = df[feature_names].values.astype(np.float64)
    y = df[target_name].values.astype(np.float64)
    if task == 'classification':
        y = y.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


# pytorch pieces

class FEADataset(Dataset):
    def __init__(self, X, y, task='regression'):
        self.X = torch.FloatTensor(X)
        if task == 'regression':
            if y.ndim == 1:
                self.y = torch.FloatTensor(y).unsqueeze(1)
            else:
                self.y = torch.FloatTensor(y)
        else:
            self.y = torch.LongTensor(y)
        self.task = task

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RegressionNet(nn.Module):
    def __init__(self, input_size=32, hidden_sizes=(128, 64, 32), dropout_rate=0.3):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout_rate)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiOutputNet(nn.Module):
    """Shared trunk + per-target heads."""
    def __init__(self, input_size=32, shared_sizes=(128, 64),
                 head_size=32, n_outputs=6, dropout_rate=0.3):
        super().__init__()
        shared_layers = []
        prev = input_size
        for h in shared_sizes:
            shared_layers += [nn.Linear(prev, h), nn.ReLU(),
                              nn.BatchNorm1d(h), nn.Dropout(dropout_rate)]
            prev = h
        self.shared = nn.Sequential(*shared_layers)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev, head_size), nn.ReLU(),
                nn.BatchNorm1d(head_size), nn.Dropout(dropout_rate * 0.5),
                nn.Linear(head_size, 1)
            )
            for _ in range(n_outputs)
        ])

    def forward(self, x):
        shared_out = self.shared(x)
        outputs = [head(shared_out) for head in self.heads]
        return torch.cat(outputs, dim=1)


class PhysicsInformedLoss(nn.Module):
    """MSE plus a penalty on negative predictions — stresses and indices are >= 0."""
    def __init__(self, physics_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.physics_weight = physics_weight

    def forward(self, pred, target):
        data_loss = self.mse(pred, target)
        neg_penalty = torch.mean(torch.relu(-pred))
        return data_loss + self.physics_weight * neg_penalty


class ClassificationNet(nn.Module):
    def __init__(self, input_size=32, hidden_sizes=(128, 64, 32), dropout_rate=0.3):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout_rate)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# training

def train_nn_regression(X_train, y_train, X_val, y_val, input_size,
                        epochs=EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE,
                        use_physics_loss=False, verbose=True):
    train_loader = DataLoader(FEADataset(X_train, y_train, 'regression'),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FEADataset(X_val, y_val, 'regression'),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = RegressionNet(input_size=input_size).to(DEVICE)
    criterion = PhysicsInformedLoss(physics_weight=0.1) if use_physics_loss else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    es = EarlyStopping(patience=patience)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        es(val_loss, model)
        if es.early_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            es.load_best(model)
            break

    return model, history


def train_multioutput_nn(X_train, y_train, X_val, y_val, input_size, n_outputs,
                         epochs=EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE,
                         verbose=True):
    train_loader = DataLoader(FEADataset(X_train, y_train, 'regression'),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FEADataset(X_val, y_val, 'regression'),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = MultiOutputNet(input_size=input_size, n_outputs=n_outputs).to(DEVICE)
    criterion = PhysicsInformedLoss(physics_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    es = EarlyStopping(patience=patience)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        es(val_loss, model)
        if es.early_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            es.load_best(model)
            break

    return model, history


def train_nn_classification(X_train, y_train, X_val, y_val, input_size,
                            epochs=EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE,
                            verbose=True):
    train_loader = DataLoader(FEADataset(X_train, y_train, 'classification'),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FEADataset(X_val, y_val, 'classification'),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = ClassificationNet(input_size=input_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    es = EarlyStopping(patience=patience)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                out = model(Xb)
                val_loss += criterion(out, yb).item() * Xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        es(val_loss, model)
        if es.early_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            es.load_best(model)
            break

    return model, history


def predict_nn(model, X, task='regression'):
    model.eval()
    loader = DataLoader(FEADataset(X, np.zeros(len(X)) if task == 'regression'
                                   else np.zeros(len(X), dtype=int), task),
                        batch_size=BATCH_SIZE, shuffle=False)
    preds, probs = [], []

    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(DEVICE)
            out = model(Xb)
            if task == 'regression':
                preds.extend(out.cpu().numpy().flatten())
            else:
                prob = torch.softmax(out, dim=1)
                preds.extend(out.argmax(1).cpu().numpy())
                probs.extend(prob[:, 1].cpu().numpy())

    if task == 'classification':
        return np.array(preds), np.array(probs)
    return np.array(preds)


def predict_multioutput_nn(model, X):
    model.eval()
    dummy_y = np.zeros((len(X), model.heads.__len__()))
    loader = DataLoader(FEADataset(X, dummy_y, 'regression'),
                        batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(DEVICE)
            out = model(Xb)
            all_preds.append(out.cpu().numpy())
    return np.vstack(all_preds)


def mc_dropout_predict(model, X, n_forward=50):
    """MC Dropout at inference — keeps dropout active, returns mean and std."""
    model.train()  # leave dropout on
    loader = DataLoader(FEADataset(X, np.zeros(len(X)), 'regression'),
                        batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    for _ in range(n_forward):
        batch_preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(DEVICE)
                out = model(Xb)
                batch_preds.extend(out.cpu().numpy().flatten())
        all_preds.append(batch_preds)

    all_preds = np.array(all_preds)
    model.eval()
    return all_preds.mean(axis=0), all_preds.std(axis=0)


# optuna tuning

def optuna_tune_xgboost(X, y, n_trials=OPTUNA_N_TRIALS):
    if not HAS_OPTUNA:
        print("  Optuna not available, using defaults")
        return XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                            random_state=RANDOM_SEED, verbosity=0)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_SEED,
            'verbosity': 0,
        }
        model = XGBRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for tr_idx, va_idx in kf.split(X):
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            scores.append(r2_score(y[va_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Optuna best R2: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    best_params = study.best_params
    best_params['random_state'] = RANDOM_SEED
    best_params['verbosity'] = 0
    return XGBRegressor(**best_params)


# repeated k-fold cv

def repeated_kfold_cv(X, y, feature_names, target_name, task='regression',
                      k=K_FOLDS, n_repeats=N_REPEATS):
    """Repeated k-fold for every model; returns summary and last-fold models."""
    print(f"\n{'='*60}")
    print(f"{n_repeats}x{k}-FOLD CV: {target_name} ({task})")
    print(f"{'='*60}")

    if task == 'classification':
        rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=RANDOM_SEED)
    else:
        rkf = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=RANDOM_SEED)
    input_size = X.shape[1]
    total_folds = k * n_repeats

    if task == 'regression':
        sklearn_models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=RANDOM_SEED),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=RANDOM_SEED),
            'XGBoost': XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=RANDOM_SEED, verbosity=0),
        }
        if HAS_CATBOOST:
            sklearn_models['CatBoost'] = CatBoostRegressor(
                iterations=200, depth=5, learning_rate=0.05,
                random_seed=RANDOM_SEED, verbose=0)
        metric_names = ['R2', 'RMSE', 'MAE']
    else:
        sklearn_models = {
            'Logistic Regression': LogisticRegression(
                random_state=RANDOM_SEED, max_iter=1000),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=RANDOM_SEED),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=RANDOM_SEED),
            'XGBoost': XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=RANDOM_SEED, verbosity=0, eval_metric='logloss'),
        }
        if HAS_CATBOOST:
            sklearn_models['CatBoost'] = CatBoostClassifier(
                iterations=200, depth=5, learning_rate=0.05,
                random_seed=RANDOM_SEED, verbose=0)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    if task == 'regression':
        model_names = list(sklearn_models.keys()) + ['Neural Network', 'PINN']
    else:
        model_names = list(sklearn_models.keys()) + ['Neural Network']
    all_results = {name: {m: [] for m in metric_names} for name in model_names}
    last_fold_models = {}

    for fold_i, (train_idx, val_idx) in enumerate(rkf.split(X, y)):
        rep = fold_i // k + 1
        fold = fold_i % k + 1
        if fold == 1:
            print(f"\n--- Repeat {rep}/{n_repeats} ---")

        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        for name, model_template in sklearn_models.items():
            m = copy.deepcopy(model_template)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)

            if task == 'regression':
                all_results[name]['R2'].append(r2_score(y_va, preds))
                all_results[name]['RMSE'].append(np.sqrt(mean_squared_error(y_va, preds)))
                all_results[name]['MAE'].append(mean_absolute_error(y_va, preds))
            else:
                all_results[name]['Accuracy'].append(accuracy_score(y_va, preds))
                all_results[name]['Precision'].append(precision_score(y_va, preds, zero_division=0))
                all_results[name]['Recall'].append(recall_score(y_va, preds, zero_division=0))
                all_results[name]['F1'].append(f1_score(y_va, preds, zero_division=0))

            if fold_i == total_folds - 1:
                last_fold_models[name] = m

        # regression gets both NN and PINN; classification only plain NN
        nn_split = int(len(X_tr) * 0.8)
        X_nn_tr, X_nn_va = X_tr[:nn_split], X_tr[nn_split:]
        y_nn_tr, y_nn_va = y_tr[:nn_split], y_tr[nn_split:]

        nn_variants = [('Neural Network', False), ('PINN', True)] if task == 'regression' else [('Neural Network', False)]
        for nn_name, use_physics in nn_variants:
            if task == 'regression':
                nn_model, _ = train_nn_regression(
                    X_nn_tr, y_nn_tr, X_nn_va, y_nn_va, input_size,
                    use_physics_loss=use_physics, verbose=False)
                preds = predict_nn(nn_model, X_va, 'regression')
                all_results[nn_name]['R2'].append(r2_score(y_va, preds))
                all_results[nn_name]['RMSE'].append(np.sqrt(mean_squared_error(y_va, preds)))
                all_results[nn_name]['MAE'].append(mean_absolute_error(y_va, preds))
            else:
                nn_model, _ = train_nn_classification(
                    X_nn_tr, y_nn_tr, X_nn_va, y_nn_va, input_size, verbose=False)
                preds, _ = predict_nn(nn_model, X_va, 'classification')
                all_results[nn_name]['Accuracy'].append(accuracy_score(y_va, preds))
                all_results[nn_name]['Precision'].append(precision_score(y_va, preds, zero_division=0))
                all_results[nn_name]['Recall'].append(recall_score(y_va, preds, zero_division=0))
                all_results[nn_name]['F1'].append(f1_score(y_va, preds, zero_division=0))

            if fold_i == total_folds - 1:
                last_fold_models[nn_name] = nn_model

        if fold == k:
            pm = metric_names[0]
            line = f"  Rep {rep} done: "
            line += " | ".join(
                f"{n}: {np.mean(all_results[n][pm][-k:]):.4f}"
                for n in list(sklearn_models.keys())[:3]
            )
            print(line)

    print(f"\n{'='*60}")
    print(f"CV SUMMARY — {target_name} ({n_repeats}x{k}-fold = {total_folds} total folds)")
    print(f"{'='*60}")
    summary = {}
    for name in model_names:
        summary[name] = {}
        for metric in metric_names:
            vals = all_results[name][metric]
            summary[name][metric] = {'mean': np.mean(vals), 'std': np.std(vals), 'values': vals}
        pm = metric_names[0]
        print(f"  {name:25s}: {pm}={summary[name][pm]['mean']:.4f} +/- {summary[name][pm]['std']:.4f}")

    return summary, last_fold_models


# gaussian process regression

def run_gpr(X_train, y_train, X_test, y_test, target_name):
    """GPR with uncertainty; returns predictions, std, metrics, fitted model."""
    print(f"\n  GPR for {target_name}...")

    # GPR is O(n^3) so cap training size
    max_gpr_samples = 500
    if len(X_train) > max_gpr_samples:
        idx = np.random.choice(len(X_train), max_gpr_samples, replace=False)
        X_gpr_train = X_train[idx]
        y_gpr_train = y_train[idx]
        print(f"  (Subsampled to {max_gpr_samples} for GPR — O(n^3) complexity)")
    else:
        X_gpr_train = X_train
        y_gpr_train = y_train

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, random_state=RANDOM_SEED,
        normalize_y=True
    )
    gpr.fit(X_gpr_train, y_gpr_train)

    y_pred, y_std = gpr.predict(X_test, return_std=True)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  GPR Results — R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"  Mean uncertainty (std): {y_std.mean():.4f}")

    return y_pred, y_std, {'R2': r2, 'RMSE': rmse, 'MAE': mae}, gpr


# conformal prediction (mapie)

def run_conformal_prediction(X_train, y_train, X_test, y_test, target_name):
    """MAPIE conformal intervals via CrossConformalRegressor (v1.3+ API)."""
    if not HAS_MAPIE:
        print(f"  MAPIE not available — skipping conformal prediction for {target_name}")
        return None, None, None

    print(f"\n  Conformal prediction for {target_name}...")

    base_model = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=RANDOM_SEED, verbosity=0)

    ccr = CrossConformalRegressor(
        estimator=base_model, cv=5, confidence_level=0.9)
    ccr.fit_conformalize(X_train, y_train)

    y_pred = ccr.predict(X_test)
    _, y_intervals = ccr.predict_interval(X_test)  # shape (n, 2, 1), lower/upper

    coverage = np.mean(
        (y_test >= y_intervals[:, 0, 0]) & (y_test <= y_intervals[:, 1, 0])
    )
    avg_width = np.mean(y_intervals[:, 1, 0] - y_intervals[:, 0, 0])

    print(f"  90% prediction interval coverage: {coverage:.3f}")
    print(f"  Average interval width: {avg_width:.4f}")

    return y_pred, y_intervals, {'coverage': coverage, 'avg_width': avg_width}


# plots (300 dpi)

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(fig, filename):
    ensure_fig_dir()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_data_overview(df, features, detected):
    reg_targets = detected['reg_targets']
    n_t = len(reg_targets)
    ncols = min(3, n_t)
    nrows = max(1, (n_t + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, target in enumerate(reg_targets):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.hist(df[target].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(target, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Distribution of {target}', fontsize=12)
        ax.grid(True, alpha=0.3)
    for i in range(n_t, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle('V7 Composite — Regression Target Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig_v7_target_distributions.png')

    clf_targets = detected['clf_targets']
    if clf_targets:
        fig, axes_clf = plt.subplots(1, len(clf_targets), figsize=(5 * len(clf_targets), 4))
        if len(clf_targets) == 1:
            axes_clf = [axes_clf]
        for ax, target in zip(axes_clf, clf_targets):
            counts = df[target].value_counts().sort_index()
            ax.bar(['Not Failed (0)', 'Failed (1)'],
                   [counts.get(0, 0), counts.get(1, 0)],
                   color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8)
            ax.set_title(f'{target}', fontsize=12)
            ax.set_ylabel('Count', fontsize=11)
            for j, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
                ax.text(j, v + 1, str(v), ha='center', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        fig.suptitle('V7 Composite — Failure Class Balance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_fig(fig, 'fig_v7_class_balance.png')

    fig, ax = plt.subplots(figsize=(16, 14))
    corr_cols = features + reg_targets
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax,
                annot=False, linewidths=0.5)
    ax.set_title('V7 Feature-Target Correlation Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    save_fig(fig, 'fig_v7_correlation_heatmap.png')

    if 'n_defects' in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df['n_defects'].value_counts().sort_index().plot(kind='bar', ax=ax,
            color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Number of Defects', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('V7 — Distribution of Defect Count', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_fig(fig, 'fig_v7_n_defects_distribution.png')


def plot_cv_comparison(summary, target_name, task='regression'):
    metric = 'R2' if task == 'regression' else 'F1'
    ylabel = 'R$^2$ Score' if task == 'regression' else 'F1 Score'

    models = list(summary.keys())
    means = [summary[m][metric]['mean'] for m in models]
    stds = [summary[m][metric]['std'] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.01,
                f'{mean:.3f}\n$\\pm${std:.3f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'V7 {N_REPEATS}x{K_FOLDS}-Fold CV — {target_name} ({ylabel})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    safe = target_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_v7_cv_{safe}_{metric.lower()}.png')


def plot_feature_importance(model, feature_names, target_name, model_name='XGBoost'):
    if not hasattr(model, 'feature_importances_'):
        return
    importance = model.feature_importances_
    idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.25)))
    ax.barh(range(len(idx)), importance[idx], color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'V7 Feature Importance — {target_name}\n({model_name})', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    for i, j in enumerate(idx[-10:]):
        ax.text(importance[j] + 0.002, len(idx) - 10 + i, f'{importance[j]:.3f}',
                va='center', fontsize=8)

    plt.tight_layout()
    safe_t = target_name.replace(' ', '_').lower()
    safe_m = model_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_v7_importance_{safe_t}_{safe_m}.png')


def plot_shap_analysis(model, X, feature_names, target_name, model_name='XGBoost'):
    if not HAS_SHAP:
        print(f"  SHAP not available — skipping for {target_name}")
        return

    print(f"  Computing SHAP values for {target_name} ({model_name})...")

    # subsample to keep this cheap
    max_shap = min(200, len(X))
    X_shap = X[:max_shap]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        fig, ax = plt.subplots(figsize=(12, max(6, len(feature_names) * 0.3)))
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                          show=False, max_display=20)
        plt.title(f'V7 SHAP Analysis — {target_name} ({model_name})', fontsize=13)
        plt.tight_layout()
        safe_t = target_name.replace(' ', '_').lower()
        save_fig(plt.gcf(), f'fig_v7_shap_{safe_t}.png')

        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.25)))
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                          plot_type='bar', show=False, max_display=20)
        plt.title(f'V7 SHAP Feature Importance — {target_name}', fontsize=13)
        plt.tight_layout()
        save_fig(plt.gcf(), f'fig_v7_shap_bar_{safe_t}.png')

    except Exception as e:
        print(f"  WARNING: SHAP analysis failed: {e}")


def plot_regression_scatter(y_true, y_pred, target_name, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5, s=40)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    axes[0].plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel(f'Actual {target_name}', fontsize=12)
    axes[0].set_ylabel(f'Predicted {target_name}', fontsize=12)
    axes[0].set_title(f'Actual vs Predicted — {model_name}', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    r2 = r2_score(y_true, y_pred)
    axes[0].text(0.05, 0.95, f'R$^2$ = {r2:.4f}', transform=axes[0].transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, alpha=0.6, edgecolors='black', linewidth=0.5, s=40)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel(f'Actual {target_name}', fontsize=12)
    axes[1].set_ylabel('Residual (Pred - Actual)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'V7 Composite — {target_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    safe_t = target_name.replace(' ', '_').lower()
    safe_m = model_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_v7_scatter_{safe_t}_{safe_m}.png')


def plot_gpr_uncertainty(y_true, y_pred, y_std, target_name):
    sort_idx = np.argsort(y_true)
    y_true_s = y_true[sort_idx]
    y_pred_s = y_pred[sort_idx]
    y_std_s = y_std[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(len(y_true_s))

    ax.scatter(x_axis, y_true_s, s=20, c='black', label='Actual', zorder=3, alpha=0.6)
    ax.plot(x_axis, y_pred_s, 'b-', linewidth=1.5, label='GPR Prediction', alpha=0.8)
    ax.fill_between(x_axis,
                     y_pred_s - 2 * y_std_s,
                     y_pred_s + 2 * y_std_s,
                     alpha=0.25, color='blue', label='95% confidence')
    ax.fill_between(x_axis,
                     y_pred_s - y_std_s,
                     y_pred_s + y_std_s,
                     alpha=0.35, color='blue', label='68% confidence')

    ax.set_xlabel('Test Sample (sorted by actual value)', fontsize=12)
    ax.set_ylabel(target_name, fontsize=12)
    ax.set_title(f'V7 GPR Uncertainty — {target_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_t = target_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_v7_gpr_uncertainty_{safe_t}.png')


def plot_conformal_intervals(y_true, y_pred, y_intervals, target_name):
    if y_intervals is None:
        return

    sort_idx = np.argsort(y_true)
    y_true_s = y_true[sort_idx]
    y_pred_s = y_pred[sort_idx]
    lower_s = y_intervals[sort_idx, 0, 0]
    upper_s = y_intervals[sort_idx, 1, 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(len(y_true_s))

    ax.scatter(x_axis, y_true_s, s=20, c='black', label='Actual', zorder=3, alpha=0.6)
    ax.plot(x_axis, y_pred_s, 'b-', linewidth=1.5, label='Prediction', alpha=0.8)
    ax.fill_between(x_axis, lower_s, upper_s,
                     alpha=0.3, color='orange', label='90% conformal interval')

    coverage = np.mean((y_true_s >= lower_s) & (y_true_s <= upper_s))
    ax.set_xlabel('Test Sample (sorted by actual value)', fontsize=12)
    ax.set_ylabel(target_name, fontsize=12)
    ax.set_title(f'V7 Conformal Prediction — {target_name}\n(Coverage: {coverage:.1%})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_t = target_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_v7_conformal_{safe_t}.png')


def plot_mc_dropout_uncertainty(y_true, y_mean, y_std, target_name):
    sort_idx = np.argsort(y_true)
    y_true_s = y_true[sort_idx]
    y_mean_s = y_mean[sort_idx]
    y_std_s = y_std[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(len(y_true_s))

    ax.scatter(x_axis, y_true_s, s=20, c='black', label='Actual', zorder=3, alpha=0.6)
    ax.plot(x_axis, y_mean_s, 'r-', linewidth=1.5, label='NN Mean Prediction', alpha=0.8)
    ax.fill_between(x_axis,
                     y_mean_s - 2 * y_std_s,
                     y_mean_s + 2 * y_std_s,
                     alpha=0.25, color='red', label='95% MC Dropout')

    ax.set_xlabel('Test Sample (sorted by actual value)', fontsize=12)
    ax.set_ylabel(target_name, fontsize=12)
    ax.set_title(f'V7 MC Dropout Uncertainty — {target_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_t = target_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_v7_mc_dropout_{safe_t}.png')


def plot_confusion_matrix_fig(y_true, y_pred, target_name, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Failed', 'Failed'],
                yticklabels=['Not Failed', 'Failed'],
                annot_kws={'size': 16}, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'V7 Confusion Matrix — {target_name}\n({model_name})', fontsize=13)
    plt.tight_layout()
    safe_t = target_name.replace(' ', '_').lower()
    safe_m = model_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_v7_cm_{safe_t}_{safe_m}.png')


def plot_roc_curves(y_true, probs_dict, target_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(probs_dict)))

    for (name, probs), color in zip(probs_dict.items(), colors):
        if probs is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'V7 ROC Curves — {target_name}', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe = target_name.replace(' ', '_').lower()
    save_fig(fig, f'fig_v7_roc_{safe}.png')


def plot_multi_target_comparison(all_summaries, task='regression'):
    metric = 'R2' if task == 'regression' else 'F1'
    ylabel = 'R$^2$ Score' if task == 'regression' else 'F1 Score'

    targets = list(all_summaries.keys())
    best_names, best_means, best_stds = [], [], []

    for t in targets:
        s = all_summaries[t]
        best = max(s, key=lambda m: s[m][metric]['mean'])
        best_names.append(best)
        best_means.append(s[best][metric]['mean'])
        best_stds.append(s[best][metric]['std'])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(targets)))
    bars = ax.bar(targets, best_means, yerr=best_stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    for bar, mean, std, nm in zip(bars, best_means, best_stds, best_names):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.01,
                f'{mean:.3f}\n({nm})', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Target Variable', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'V7 Best Model {ylabel} Across All Targets', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    save_fig(fig, f'fig_v7_multi_target_{metric.lower()}.png')


def plot_multioutput_nn_comparison(y_true_multi, y_pred_multi, target_names):
    n_targets = len(target_names)
    ncols = min(3, n_targets)
    nrows = (n_targets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, target in enumerate(target_names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        y_t = y_true_multi[:, i]
        y_p = y_pred_multi[:, i]
        ax.scatter(y_t, y_p, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        margin = (lims[1] - lims[0]) * 0.05
        ax.plot([lims[0]-margin, lims[1]+margin], [lims[0]-margin, lims[1]+margin],
                'r--', linewidth=1.5)
        r2 = r2_score(y_t, y_p)
        ax.set_title(f'{target}\nR$^2$ = {r2:.4f}', fontsize=11)
        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.grid(True, alpha=0.3)

    for i in range(n_targets, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle('V7 Multi-Output Neural Network — All Targets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig_v7_multioutput_nn.png')


# results export

def export_results_txt(all_reg_summaries, all_clf_summaries, detected, n_samples,
                       gpr_results=None, conformal_results=None, optuna_result=None,
                       multioutput_r2=None):
    ensure_fig_dir()
    path = os.path.join(FIG_DIR, 'v7_ml_results_summary.txt')

    with open(path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RP3 V7 COMPOSITE ML PIPELINE — RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {n_samples} samples\n")
        f.write(f"Features: {len(detected['features'])} inputs (incl. engineered)\n")
        f.write(f"Cross-validation: {N_REPEATS}x{K_FOLDS}-fold ({N_REPEATS * K_FOLDS} total folds)\n")
        f.write(f"Models: Linear, Ridge, RF, GB, XGBoost, CatBoost, NN, PINN\n\n")

        f.write("-" * 70 + "\n")
        f.write("REGRESSION RESULTS\n")
        f.write("-" * 70 + "\n\n")
        for target, summary in all_reg_summaries.items():
            f.write(f"\nTarget: {target}\n")
            f.write(f"{'Model':25s} {'R2':>14s} {'RMSE':>14s} {'MAE':>14s}\n")
            f.write("-" * 70 + "\n")
            for name in summary:
                r2 = summary[name]['R2']
                rmse = summary[name]['RMSE']
                mae = summary[name]['MAE']
                f.write(f"{name:25s} {r2['mean']:.4f}+/-{r2['std']:.4f}"
                        f"  {rmse['mean']:.4f}+/-{rmse['std']:.4f}"
                        f"  {mae['mean']:.4f}+/-{mae['std']:.4f}\n")

        if gpr_results:
            f.write("\n" + "-" * 70 + "\n")
            f.write("GAUSSIAN PROCESS REGRESSION\n")
            f.write("-" * 70 + "\n\n")
            for target, metrics in gpr_results.items():
                f.write(f"  {target:25s}: R2={metrics['R2']:.4f}, "
                        f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}\n")

        if conformal_results:
            f.write("\n" + "-" * 70 + "\n")
            f.write("CONFORMAL PREDICTION (90% intervals)\n")
            f.write("-" * 70 + "\n\n")
            for target, metrics in conformal_results.items():
                if metrics:
                    f.write(f"  {target:25s}: Coverage={metrics['coverage']:.3f}, "
                            f"Avg Width={metrics['avg_width']:.4f}\n")

        if multioutput_r2:
            f.write("\n" + "-" * 70 + "\n")
            f.write("MULTI-OUTPUT NEURAL NETWORK\n")
            f.write("-" * 70 + "\n\n")
            for target, r2 in multioutput_r2.items():
                f.write(f"  {target:25s}: R2={r2:.4f}\n")

        if optuna_result:
            f.write("\n" + "-" * 70 + "\n")
            f.write("OPTUNA HYPERPARAMETER TUNING (XGBoost)\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"  Target: {optuna_result['target']}\n")
            f.write(f"  Best CV R2: {optuna_result['best_r2']:.4f}\n")
            f.write(f"  Best params: {optuna_result['best_params']}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("CLASSIFICATION RESULTS\n")
        f.write("-" * 70 + "\n\n")
        for target, summary in all_clf_summaries.items():
            f.write(f"\nTarget: {target}\n")
            f.write(f"{'Model':25s} {'Accuracy':>14s} {'F1':>14s} "
                    f"{'Precision':>14s} {'Recall':>14s}\n")
            f.write("-" * 80 + "\n")
            for name in summary:
                acc = summary[name]['Accuracy']
                f1v = summary[name]['F1']
                prec = summary[name]['Precision']
                rec = summary[name]['Recall']
                f.write(f"{name:25s} {acc['mean']:.4f}+/-{acc['std']:.4f}"
                        f"  {f1v['mean']:.4f}+/-{f1v['std']:.4f}"
                        f"  {prec['mean']:.4f}+/-{prec['std']:.4f}"
                        f"  {rec['mean']:.4f}+/-{rec['std']:.4f}\n")

    print(f"\nResults summary saved: {path}")


# main

def main():
    start_time = time.time()

    print("=" * 70)
    print("RP3 V7 ML PIPELINE — COMPOSITE CRACK-DAMAGE SURROGATE MODEL")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device:    {DEVICE}")
    print(f"Data:      {os.path.basename(DATA_PATH)}")
    print(f"Libraries: CatBoost={'YES' if HAS_CATBOOST else 'NO'}, "
          f"SHAP={'YES' if HAS_SHAP else 'NO'}, "
          f"Optuna={'YES' if HAS_OPTUNA else 'NO'}, "
          f"MAPIE={'YES' if HAS_MAPIE else 'NO'}")
    print("=" * 70)

    df = load_data(DATA_PATH)

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING")
    print(f"{'='*60}")
    df = engineer_features(df)

    detected = detect_columns(df)
    features = detected['features']
    n_features = len(features)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget statistics:")
    target_cols = detected['reg_targets'] + detected['clf_targets']
    if target_cols:
        print(df[target_cols].describe().to_string())

    print(f"\n{'='*60}")
    print("GENERATING DATA OVERVIEW PLOTS")
    print(f"{'='*60}")
    plot_data_overview(df, features, detected)

    all_reg_summaries = {}
    all_reg_models = {}

    for target in detected['reg_targets']:
        X, y, scaler = preprocess(df, features, target, task='regression')
        summary, last_models = repeated_kfold_cv(
            X, y, features, target, task='regression')
        all_reg_summaries[target] = summary
        all_reg_models[target] = last_models

        plot_cv_comparison(summary, target, task='regression')

        # feature importance from the best available tree model
        for mname in ['XGBoost', 'CatBoost', 'Gradient Boosting', 'Random Forest']:
            if mname in last_models and hasattr(last_models[mname], 'feature_importances_'):
                plot_feature_importance(last_models[mname], features, target, mname)
                break

        for mname in ['XGBoost', 'CatBoost', 'Gradient Boosting']:
            if mname in last_models:
                plot_shap_analysis(last_models[mname], X, features, target, mname)
                break

        # hold-out scatter for the CV winner
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                                    random_state=RANDOM_SEED)
        best_name = max(summary, key=lambda m: summary[m]['R2']['mean'])
        if 'Neural' in best_name or 'PINN' in best_name:
            nn_m, _ = train_nn_regression(X_tr, y_tr, X_te, y_te, n_features,
                                          use_physics_loss=('PINN' in best_name),
                                          verbose=False)
            y_pred = predict_nn(nn_m, X_te, 'regression')
        else:
            best_m = copy.deepcopy(last_models[best_name])
            best_m.fit(X_tr, y_tr)
            y_pred = best_m.predict(X_te)
        plot_regression_scatter(y_te, y_pred, target, best_name)

    if len(all_reg_summaries) > 1:
        plot_multi_target_comparison(all_reg_summaries, task='regression')

    optuna_result = None
    if HAS_OPTUNA and detected['reg_targets']:
        print(f"\n{'='*60}")
        print("OPTUNA HYPERPARAMETER TUNING (XGBoost)")
        print(f"{'='*60}")
        primary_target = detected['reg_targets'][0]
        X, y, scaler = preprocess(df, features, primary_target, task='regression')
        tuned_xgb = optuna_tune_xgboost(X, y, n_trials=OPTUNA_N_TRIALS)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                                    random_state=RANDOM_SEED)
        tuned_xgb.fit(X_tr, y_tr)
        y_pred_tuned = tuned_xgb.predict(X_te)
        tuned_r2 = r2_score(y_te, y_pred_tuned)
        print(f"  Tuned XGBoost hold-out R2: {tuned_r2:.4f}")

        plot_regression_scatter(y_te, y_pred_tuned, primary_target, 'XGBoost (Optuna-tuned)')

        optuna_result = {
            'target': primary_target,
            'best_r2': tuned_r2,
            'best_params': tuned_xgb.get_params()
        }

    gpr_results = {}
    print(f"\n{'='*60}")
    print("GAUSSIAN PROCESS REGRESSION (with uncertainty)")
    print(f"{'='*60}")

    for target in detected['reg_targets'][:3]:  # GPR is slow; cap at 3 targets
        X, y, scaler = preprocess(df, features, target, task='regression')
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                                    random_state=RANDOM_SEED)
        y_pred_gpr, y_std_gpr, gpr_metrics, gpr_model = run_gpr(
            X_tr, y_tr, X_te, y_te, target)
        gpr_results[target] = gpr_metrics
        plot_gpr_uncertainty(y_te, y_pred_gpr, y_std_gpr, target)

    conformal_results = {}
    if HAS_MAPIE:
        print(f"\n{'='*60}")
        print("CONFORMAL PREDICTION (MAPIE)")
        print(f"{'='*60}")

        for target in detected['reg_targets'][:3]:
            X, y, scaler = preprocess(df, features, target, task='regression')
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                                        random_state=RANDOM_SEED)
            y_pred_cp, y_intervals_cp, cp_metrics = run_conformal_prediction(
                X_tr, y_tr, X_te, y_te, target)
            conformal_results[target] = cp_metrics
            if y_intervals_cp is not None:
                plot_conformal_intervals(y_te, y_pred_cp, y_intervals_cp, target)

    print(f"\n{'='*60}")
    print("MC DROPOUT UNCERTAINTY (Neural Network)")
    print(f"{'='*60}")

    for target in detected['reg_targets'][:2]:
        X, y, scaler = preprocess(df, features, target, task='regression')
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                                    random_state=RANDOM_SEED)
        nn_split = int(len(X_tr) * 0.8)
        nn_model, _ = train_nn_regression(
            X_tr[:nn_split], y_tr[:nn_split],
            X_tr[nn_split:], y_tr[nn_split:],
            n_features, use_physics_loss=True, verbose=False)

        y_mean_mc, y_std_mc = mc_dropout_predict(nn_model, X_te, n_forward=50)
        mc_r2 = r2_score(y_te, y_mean_mc)
        print(f"  {target}: MC Dropout R2={mc_r2:.4f}, Mean Std={y_std_mc.mean():.4f}")
        plot_mc_dropout_uncertainty(y_te, y_mean_mc, y_std_mc, target)

    multioutput_r2 = None
    if len(detected['reg_targets']) > 1:
        print(f"\n{'='*60}")
        print("MULTI-OUTPUT NEURAL NETWORK")
        print(f"{'='*60}")

        reg_targets = detected['reg_targets']
        X_all = df[features].values.astype(np.float64)
        y_all = df[reg_targets].values.astype(np.float64)

        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_all)
        scaler_Y = StandardScaler()
        y_scaled = scaler_Y.fit_transform(y_all)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y_scaled, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        nn_split = int(len(X_tr) * 0.8)
        mo_model, mo_history = train_multioutput_nn(
            X_tr[:nn_split], y_tr[:nn_split],
            X_tr[nn_split:], y_tr[nn_split:],
            n_features, n_outputs=len(reg_targets), verbose=True)

        y_pred_multi = predict_multioutput_nn(mo_model, X_te)

        # unscale before metrics
        y_te_orig = scaler_Y.inverse_transform(y_te)
        y_pred_orig = scaler_Y.inverse_transform(y_pred_multi)

        multioutput_r2 = {}
        print("\n  Multi-Output NN Results:")
        for i, target in enumerate(reg_targets):
            r2 = r2_score(y_te_orig[:, i], y_pred_orig[:, i])
            multioutput_r2[target] = r2
            print(f"    {target:25s}: R2 = {r2:.4f}")

        plot_multioutput_nn_comparison(y_te_orig, y_pred_orig, reg_targets)

    all_clf_summaries = {}
    all_clf_models = {}

    for target in detected['clf_targets']:
        X, y, scaler = preprocess(df, features, target, task='classification')

        unique, counts = np.unique(y, return_counts=True)
        print(f"\n  {target} class balance: {dict(zip(unique.astype(int), counts))}")

        if len(unique) < 2:
            print(f"  WARNING: {target} has only one class — skipping.")
            continue

        summary, last_models = repeated_kfold_cv(
            X, y, features, target, task='classification')
        all_clf_summaries[target] = summary
        all_clf_models[target] = last_models

        plot_cv_comparison(summary, target, task='classification')

        for mname in ['XGBoost', 'CatBoost', 'Gradient Boosting', 'Random Forest']:
            if mname in last_models and hasattr(last_models[mname], 'feature_importances_'):
                plot_feature_importance(last_models[mname], features, target, mname)
                break

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                                    random_state=RANDOM_SEED, stratify=y)
        probs_dict = {}
        for mname in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'CatBoost']:
            if mname in last_models:
                m = copy.deepcopy(last_models[mname])
                m.fit(X_tr, y_tr)
                probs_dict[mname] = m.predict_proba(X_te)[:, 1]
                plot_confusion_matrix_fig(y_te, m.predict(X_te), target, mname)

        nn_m, _ = train_nn_classification(X_tr, y_tr, X_te, y_te, n_features, verbose=False)
        nn_preds, nn_probs = predict_nn(nn_m, X_te, 'classification')
        probs_dict['Neural Network'] = nn_probs
        plot_confusion_matrix_fig(y_te, nn_preds, target, 'Neural Network')
        plot_roc_curves(y_te, probs_dict, target)

    if len(all_clf_summaries) > 1:
        plot_multi_target_comparison(all_clf_summaries, task='classification')

    export_results_txt(all_reg_summaries, all_clf_summaries, detected, len(df),
                       gpr_results=gpr_results,
                       conformal_results=conformal_results,
                       optuna_result=optuna_result,
                       multioutput_r2=multioutput_r2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("V7 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    print(f"All figures saved to: {FIG_DIR}/")
    print(f"Results summary: {FIG_DIR}/v7_ml_results_summary.txt")

    print("\nFigures generated:")
    if os.path.exists(FIG_DIR):
        figs = sorted(os.listdir(FIG_DIR))
        for f in figs:
            print(f"  {f}")
        print(f"\nTotal: {len(figs)} files")

    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)
    print(f"\nRegression ({N_REPEATS}x{K_FOLDS}-fold CV R2):")
    for target, summary in all_reg_summaries.items():
        best = max(summary, key=lambda m: summary[m]['R2']['mean'])
        r2 = summary[best]['R2']
        print(f"  {target:25s}: {best} R2={r2['mean']:.4f}+/-{r2['std']:.4f}")

    if gpr_results:
        print("\nGPR (with uncertainty):")
        for target, metrics in gpr_results.items():
            print(f"  {target:25s}: R2={metrics['R2']:.4f}")

    if multioutput_r2:
        print("\nMulti-Output NN:")
        for target, r2 in multioutput_r2.items():
            print(f"  {target:25s}: R2={r2:.4f}")

    print(f"\nClassification ({N_REPEATS}x{K_FOLDS}-fold CV F1):")
    for target, summary in all_clf_summaries.items():
        best = max(summary, key=lambda m: summary[m]['F1']['mean'])
        f1v = summary[best]['F1']
        print(f"  {target:25s}: {best} F1={f1v['mean']:.4f}+/-{f1v['std']:.4f}")

    print("\n" + "=" * 70)
    print("IMPROVEMENTS OVER V4:")
    print("  - 2x data (1000 vs 500 samples)")
    print("  - CatBoost model added")
    print("  - SHAP explainability analysis")
    print("  - Physics-based dimensionless features")
    print("  - Gaussian Process Regression with uncertainty")
    print("  - Optuna hyperparameter tuning")
    print("  - Multi-output neural network (shared layers)")
    print("  - Conformal prediction intervals (MAPIE)")
    print("  - Physics-informed neural network loss (PINN)")
    print("  - MC Dropout uncertainty estimation")
    print("  - Repeated k-fold CV (3x5 = 15 folds)")
    print("=" * 70)


if __name__ == '__main__':
    main()
