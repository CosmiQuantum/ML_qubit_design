#!/usr/bin/env python3
"""
ml_22: evaluate the combined inverse+surrogate model on the held-out test set.

HPC-friendly script version of ml_22_print_results_surrogate_defined_loss.ipynb.
Runs on CPU, writes prediction and runtime CSVs/JSON (no plotting).

Example:
  python ml_22_print_results_surrogate_defined_loss.py
  python ml_22_print_results_surrogate_defined_loss.py --n-warmup 5 --n-repeat 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

# Quiet TF before import.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices')

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from parameters_surrogate_defined_loss import (
    DATA_DIR,
    METADATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    SCALERS_DIR,
)


class ScalerConversionLayer(tf.keras.layers.Layer):
    """Must be registered before loading the combined .keras model."""

    def __init__(self, scale_a, scale_b, **kwargs):
        kwargs.setdefault('trainable', False)
        super().__init__(**kwargs)
        self._scale_a = tf.constant(scale_a, dtype=tf.float32)
        self._scale_b = tf.constant(scale_b, dtype=tf.float32)
        self._cfg = dict(
            scale_a=list(scale_a) if hasattr(scale_a, '__iter__') else scale_a,
            scale_b=list(scale_b) if hasattr(scale_b, '__iter__') else scale_b,
        )

    def call(self, inputs):
        a = tf.cast(self._scale_a, inputs.dtype)
        b = tf.cast(self._scale_b, inputs.dtype)
        return inputs * a + b

    def get_config(self):
        config = super().get_config()
        config.update(self._cfg)
        return config


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {label}: {path}\n'
            'Copy the gitignored data/model/scalers artifacts into this experiment folder first.'
        )
    return path


def _bench(predict_fn, X, n_warmup=10, n_repeat=50, single_sample=False):
    """Time a keras .predict() call. Returns (mean_ms, std_ms) per call."""
    for _ in range(n_warmup):
        _ = predict_fn(X[:1] if single_sample else X, verbose=0)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = predict_fn(X[:1] if single_sample else X, verbose=0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--model',
        default=None,
        help='Combined model path (default: MODEL_DIR/surrogate_loss_2in_3out_best_model.keras)',
    )
    p.add_argument('--n-warmup', type=int, default=10, help='Warmup predict calls before timing')
    p.add_argument('--n-repeat', type=int, default=50, help='Timed predict repeats')
    p.add_argument(
        '--ansys-min-per-sample',
        type=float,
        default=2.0,
        help='EM simulator reference time in minutes per sample (for speedup numbers)',
    )
    p.add_argument('--skip-runtime-bench', action='store_true', help='Skip CPU timing section')
    return p.parse_args()


def main():
    args = parse_args()
    encoding = 'surrogate_defined_loss'

    x_path = _require_file(
        Path(DATA_DIR) / 'npy' / 'x_test_one_hot_encoding_augmented.npy',
        'test Hamiltonian features',
    )
    y_path = _require_file(
        Path(DATA_DIR) / 'npy' / 'y_test_one_hot_encoding_augmented.npy',
        'test Quantum Metal targets',
    )
    x_names_path = _require_file(Path(METADATA_DIR) / 'X_names', 'Hamiltonian column names')
    y_cols_path = _require_file(Path(METADATA_DIR) / 'y_columns.npy', 'Quantum Metal column names')

    chosen_path = Path(args.model) if args.model else Path(MODEL_DIR) / 'surrogate_loss_2in_3out_best_model.keras'
    _require_file(chosen_path, 'combined inverse+surrogate model')

    X_test = np.load(x_path, allow_pickle=True)
    y_test = np.load(y_path, allow_pickle=True)

    with open(x_names_path, 'r') as f:
        Hamiltonian_column_names = f.read().splitlines()
    qiskit_param_names = np.load(y_cols_path, allow_pickle=True).astype(str).tolist()

    print(f'Inputs (Hamiltonian):     {X_test.shape[1]} columns')
    print(f'Outputs (Quantum Metal):  {y_test.shape[1]} columns')
    print(f'Test samples: {len(X_test)}')
    print(f'Hamiltonian columns: {Hamiltonian_column_names}')
    print(f'Quantum Metal columns: {qiskit_param_names}')
    print(f'Model path: {chosen_path}')

    X_test_cur = np.asarray(X_test)
    y_test_cur = np.asarray(y_test)

    tf.keras.backend.clear_session()
    gc.collect()

    with tf.device('/CPU:0'):
        combined_model = load_model(
            str(chosen_path),
            compile=False,
            custom_objects={'ScalerConversionLayer': ScalerConversionLayer},
        )
        inverse_model = combined_model.get_layer('inverse_model')
        predictions = combined_model.predict(X_test_cur, verbose=0)

    if isinstance(predictions, list):
        Hamiltonian_pred = predictions[0]
        qiskit_pred = predictions[1]
    else:
        Hamiltonian_pred = predictions
        with tf.device('/CPU:0'):
            qiskit_pred = inverse_model.predict(X_test_cur, verbose=0)

    print(f'Samples: {len(X_test_cur)}')
    print(f'Hamiltonian target dim: {X_test_cur.shape[1]}')
    print(f'Quantum Metal output dim: {y_test_cur.shape[1]}')

    if not args.skip_runtime_bench:
        X_bench = np.asarray(X_test_cur, dtype=np.float32)
        N_bench = len(X_bench)

        with tf.device('/CPU:0'):
            inv_batch_mean_ms, inv_batch_std_ms = _bench(
                inverse_model.predict, X_bench,
                n_warmup=args.n_warmup, n_repeat=args.n_repeat, single_sample=False,
            )
            inv_single_mean_ms, inv_single_std_ms = _bench(
                inverse_model.predict, X_bench,
                n_warmup=args.n_warmup, n_repeat=args.n_repeat, single_sample=True,
            )
            comb_batch_mean_ms, comb_batch_std_ms = _bench(
                combined_model.predict, X_bench,
                n_warmup=args.n_warmup, n_repeat=args.n_repeat, single_sample=False,
            )
            comb_single_mean_ms, comb_single_std_ms = _bench(
                combined_model.predict, X_bench,
                n_warmup=args.n_warmup, n_repeat=args.n_repeat, single_sample=True,
            )

        inv_per_sample_ms = inv_batch_mean_ms / N_bench
        comb_per_sample_ms = comb_batch_mean_ms / N_bench
        ansys_min_per_sample = args.ansys_min_per_sample
        ansys_ms_per_sample = ansys_min_per_sample * 60.0 * 1000.0
        speedup_inv_vs_ansys = ansys_ms_per_sample / inv_per_sample_ms
        speedup_comb_vs_ansys = ansys_ms_per_sample / comb_per_sample_ms
        speedup_single_vs_ansys = ansys_ms_per_sample / comb_single_mean_ms

        print('=' * 78)
        print(' Runtime benchmark on CPU')
        print('=' * 78)
        print(f' Hardware : {platform.processor() or platform.machine()}')
        print(f' System   : {platform.system()} {platform.release()}')
        print(f' Python   : {platform.python_version()}')
        print(f' TF       : {tf.__version__}')
        print(f' N_test   : {N_bench} samples')
        print(f' Warmup   : {args.n_warmup} calls | Repeat: {args.n_repeat} calls')
        print('-' * 78)
        print(' Inverse MLP (Hamiltonian -> Quantum Metal params)')
        print(
            f'   batch  : {inv_batch_mean_ms:8.3f} +/- {inv_batch_std_ms:.3f} ms total  '
            f'({inv_per_sample_ms * 1000.0:7.2f} us / sample)'
        )
        print(f'   single : {inv_single_mean_ms:8.3f} +/- {inv_single_std_ms:.3f} ms / call')
        print('-' * 78)
        print(' Combined pipeline (Hamiltonian -> Quantum Metal -> reconstructed Hamiltonian)')
        print(
            f'   batch  : {comb_batch_mean_ms:8.3f} +/- {comb_batch_std_ms:.3f} ms total  '
            f'({comb_per_sample_ms * 1000.0:7.2f} us / sample)'
        )
        print(f'   single : {comb_single_mean_ms:8.3f} +/- {comb_single_std_ms:.3f} ms / call')
        print('-' * 78)
        print(
            f' EM simulator reference  : ~{ansys_min_per_sample:.0f} min / sample '
            f'(~{ansys_ms_per_sample:.2e} ms / sample)'
        )
        print(
            f' Speedup (inverse MLP batch, per-sample) : {speedup_inv_vs_ansys:.2e}  '
            f'({int(np.log10(speedup_inv_vs_ansys))} orders of magnitude)'
        )
        print(
            f' Speedup (combined batch, per-sample)    : {speedup_comb_vs_ansys:.2e}  '
            f'({int(np.log10(speedup_comb_vs_ansys))} orders of magnitude)'
        )
        print(
            f' Speedup (combined single-sample call)   : {speedup_single_vs_ansys:.2e}  '
            f'({int(np.log10(speedup_single_vs_ansys))} orders of magnitude)'
        )
        print('=' * 78)

        runtime_stats = {
            'created_by': 'ml_22_print_results_surrogate_defined_loss.py',
            'hardware': platform.processor() or platform.machine(),
            'system': f'{platform.system()} {platform.release()}',
            'python': platform.python_version(),
            'tf': tf.__version__,
            'n_test': N_bench,
            'n_warmup': args.n_warmup,
            'n_repeat': args.n_repeat,
            'inverse_batch_total_ms': inv_batch_mean_ms,
            'inverse_per_sample_ms': inv_per_sample_ms,
            'inverse_single_call_ms': inv_single_mean_ms,
            'combined_batch_total_ms': comb_batch_mean_ms,
            'combined_per_sample_ms': comb_per_sample_ms,
            'combined_single_call_ms': comb_single_mean_ms,
            'ansys_min_per_sample': ansys_min_per_sample,
            'speedup_combined_per_sample': speedup_comb_vs_ansys,
            'speedup_combined_single_call': speedup_single_vs_ansys,
        }
        runtime_path = Path(RESULTS_DIR) / 'runtime' / 'runtime_benchmark.json'
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        with runtime_path.open('w') as f:
            json.dump(runtime_stats, f, indent=2)
        print(f'\nSaved runtime benchmark to {runtime_path}')

    # Unscale predictions and write reconstruction / percent-error tables.
    with open(x_names_path, 'r') as f:
        Hamiltonian_names = f.read().splitlines()
    qiskit_names = np.load(y_cols_path, allow_pickle=True).astype(str).tolist()

    X_test_cur = np.asarray(X_test_cur, dtype=np.float32)
    y_test_cur = np.asarray(y_test_cur, dtype=np.float32)
    Hamiltonian_pred = np.asarray(Hamiltonian_pred, dtype=np.float32)
    qiskit_pred = np.asarray(qiskit_pred, dtype=np.float32)

    X_test_unscaled = X_test_cur.copy()
    Hamiltonian_pred_unscaled = Hamiltonian_pred.copy()
    for j, Hamiltonian_name in enumerate(Hamiltonian_names[:X_test_cur.shape[1]]):
        scaler_path = _require_file(
            Path(SCALERS_DIR) / f'scaler_X_{Hamiltonian_name}.save',
            f'X scaler for {Hamiltonian_name}',
        )
        scaler = joblib.load(scaler_path)
        X_test_unscaled[:, j] = scaler.inverse_transform(X_test_cur[:, [j]]).ravel()
        Hamiltonian_pred_unscaled[:, j] = scaler.inverse_transform(Hamiltonian_pred[:, [j]]).ravel()

    qiskit_pred_unscaled = qiskit_pred.copy()
    y_test_unscaled = y_test_cur.copy()
    for j, col_name in enumerate(qiskit_names[:qiskit_pred.shape[1]]):
        scaler_path = _require_file(
            Path(SCALERS_DIR) / f'scaler_y_{col_name}_one_hot_encoding.save',
            f'y scaler for {col_name}',
        )
        scaler = joblib.load(scaler_path)
        qiskit_pred_unscaled[:, j] = scaler.inverse_transform(qiskit_pred[:, [j]]).ravel()
        y_test_unscaled[:, j] = scaler.inverse_transform(y_test_cur[:, [j]]).ravel()

    prediction_rows = []
    for i in range(X_test_unscaled.shape[0]):
        row = {'sample_idx': i}
        for j, name in enumerate(Hamiltonian_names[:X_test_unscaled.shape[1]]):
            row[f'target_{name}'] = X_test_unscaled[i, j]
            row[f'pred_{name}'] = Hamiltonian_pred_unscaled[i, j]
            row[f'abs_error_{name}'] = abs(Hamiltonian_pred_unscaled[i, j] - X_test_unscaled[i, j])
        for j, name in enumerate(qiskit_names[:qiskit_pred_unscaled.shape[1]]):
            short = name.replace('design_options.', '')
            row[f'ref_{short}'] = y_test_unscaled[i, j]
            row[f'pred_{short}'] = qiskit_pred_unscaled[i, j]
        prediction_rows.append(row)

    pred_df = pd.DataFrame(prediction_rows)
    predictions_path = (
        Path(RESULTS_DIR) / 'predictions' / f'surrogate_loss_Hamiltonian_reconstruction_unscaled_{encoding}.csv'
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_path, index=False, float_format='%.8g')
    print(f'Saved unscaled predictions to {predictions_path}')

    eps = 1e-15
    pct_errors_unscaled = (
        100.0 * np.abs(Hamiltonian_pred_unscaled - X_test_unscaled) / (np.abs(X_test_unscaled) + eps)
    )
    percent_error_df = pd.DataFrame({
        'frequency': pct_errors_unscaled[:, 0],
        'anharmonicity': pct_errors_unscaled[:, 1],
    })
    percent_error_path = Path(RESULTS_DIR) / 'validation' / 'inverse+surrogate_percentErrors.csv'
    percent_error_path.parent.mkdir(parents=True, exist_ok=True)
    percent_error_df.to_csv(percent_error_path, index=False)
    print(f'Saved manuscript percent errors to {percent_error_path}')

    summary = percent_error_df.agg(['mean', 'median', 'std', 'min', 'max']).T
    print(summary.round(4).to_string())
    print('Done.')


if __name__ == '__main__':
    main()
