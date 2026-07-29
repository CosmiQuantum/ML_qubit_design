#!/usr/bin/env python3
"""
ml_30: batch-size runtime benchmark for inverse+surrogate and surrogate-only models.

HPC-friendly script version of ml_30_runtime_batch_benchmark.ipynb.
Writes timing CSVs/JSON only (plotting lives in ml_31).

Example:
  python ml_30_runtime_batch_benchmark.py --cpu-only
  python ml_30_runtime_batch_benchmark.py --n-warmup 5 --n-repeat 10
  python ml_30_runtime_batch_benchmark.py --batch-numbers 1,8,64,512
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as importlib_metadata
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

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from parameters_surrogate_defined_loss import DATA_DIR, METADATA_DIR, MODEL_DIR, RESULTS_DIR


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
            'Copy the gitignored data/model artifacts into this experiment folder first.'
        )
    return path


def _load_npy(path: Path) -> np.ndarray:
    _require_file(path, 'data file')
    return np.load(path, allow_pickle=True).astype('float32')


def make_batch(X, batch_number):
    """Tile the test set when a requested batch is larger than the held-out set."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) >= batch_number:
        return X[:batch_number]
    reps = int(np.ceil(batch_number / len(X)))
    return np.tile(X, (reps, 1))[:batch_number]


def sync_tensorflow_outputs(outputs):
    """Force GPU work to finish before stopping the timer."""
    if isinstance(outputs, (list, tuple)):
        for output in outputs:
            sync_tensorflow_outputs(output)
    elif isinstance(outputs, dict):
        for output in outputs.values():
            sync_tensorflow_outputs(output)
    elif tf.is_tensor(outputs):
        _ = outputs.numpy()
    else:
        _ = np.asarray(outputs)


def call_model_once(model, x_tensor):
    outputs = model(x_tensor, training=False)
    sync_tensorflow_outputs(outputs)


def package_version(package_name):
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--combined-model',
        default=None,
        help='Combined inverse+surrogate model path '
             '(default: MODEL_DIR/best_keras_model_surrogate_defined_loss.keras)',
    )
    p.add_argument(
        '--surrogate-model',
        default=None,
        help='Surrogate-only model path (default: MODEL_DIR/best_keras_model_model2_surrogate.keras)',
    )
    p.add_argument(
        '--batch-numbers',
        default='1,2,4,8,16,32,64,128,256,512,1024,2048',
        help='Comma-separated batch sizes to time',
    )
    p.add_argument('--n-warmup', type=int, default=20, help='Warmup forward passes per batch size')
    p.add_argument('--n-repeat', type=int, default=50, help='Timed repeats per batch size')
    p.add_argument(
        '--ansys-seconds-per-sample',
        type=float,
        default=2.0 * 60.0,
        help='EM simulator reference seconds per sample (stored in metadata)',
    )
    device = p.add_mutually_exclusive_group()
    device.add_argument('--cpu-only', action='store_true', help='Run CPU benchmarks only')
    device.add_argument('--gpu-only', action='store_true', help='Run GPU benchmarks only')
    return p.parse_args()


def benchmark_model_path(
    model_path,
    X,
    benchmark_name,
    device_label,
    device_name,
    batch_numbers,
    n_warmup,
    n_repeat,
    gpu_available,
    custom_objects=None,
):
    rows = []
    custom_objects = custom_objects or {}

    if device_label == 'GPU' and not gpu_available:
        print(f'Skipping {benchmark_name} on GPU because no TensorFlow GPU is visible.')
        return rows

    print(f'Loading {benchmark_name} on {device_label}: {model_path}')
    tf.keras.backend.clear_session()
    gc.collect()

    try:
        with tf.device(device_name):
            model = load_model(str(model_path), compile=False, custom_objects=custom_objects)
    except Exception as e:
        rows.append({
            'benchmark': benchmark_name,
            'device': device_label,
            'batch_number': np.nan,
            'repeat': np.nan,
            'status': 'load_error',
            'error': repr(e),
        })
        print(f'Could not load {benchmark_name} on {device_label}: {repr(e)}')
        return rows

    for batch_number in batch_numbers:
        X_batch = make_batch(X, batch_number)
        try:
            with tf.device(device_name):
                x_tensor = tf.convert_to_tensor(X_batch, dtype=tf.float32)
                actual_tensor_device = str(x_tensor.device)

                for _ in range(n_warmup):
                    call_model_once(model, x_tensor)

                for repeat in range(n_repeat):
                    t0 = time.perf_counter()
                    call_model_once(model, x_tensor)
                    total_seconds = time.perf_counter() - t0

                    rows.append({
                        'benchmark': benchmark_name,
                        'device': device_label,
                        'device_name': device_name,
                        'actual_tensor_device': actual_tensor_device,
                        'batch_number': int(batch_number),
                        'repeat': int(repeat),
                        'total_seconds': float(total_seconds),
                        'time_per_sample_seconds': float(total_seconds / batch_number),
                        'status': 'ok',
                        'error': '',
                    })
            print(
                f'  finished {benchmark_name} {device_label} batch={batch_number} '
                f'({n_repeat} timed repeats)'
            )
        except Exception as e:
            rows.append({
                'benchmark': benchmark_name,
                'device': device_label,
                'device_name': device_name,
                'batch_number': int(batch_number),
                'repeat': np.nan,
                'status': 'benchmark_error',
                'error': repr(e),
            })
            print(
                f'Benchmark error for {benchmark_name} on {device_label}, '
                f'batch {batch_number}: {repr(e)}'
            )

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return rows


def main():
    args = parse_args()
    batch_numbers = [int(x.strip()) for x in args.batch_numbers.split(',') if x.strip()]
    if not batch_numbers:
        raise ValueError('No batch numbers provided.')

    run_cpu = not args.gpu_only
    run_gpu = not args.cpu_only

    combined_model_path = (
        Path(args.combined_model)
        if args.combined_model
        else Path(MODEL_DIR) / 'best_keras_model_surrogate_defined_loss.keras'
    )
    surrogate_model_path = (
        Path(args.surrogate_model)
        if args.surrogate_model
        else Path(MODEL_DIR) / 'best_keras_model_model2_surrogate.keras'
    )
    _require_file(combined_model_path, 'combined inverse+surrogate model')
    _require_file(surrogate_model_path, 'surrogate-only model')

    runtime_dir = Path(RESULTS_DIR) / 'runtime'
    runtime_dir.mkdir(parents=True, exist_ok=True)

    print(f'Combined inverse + surrogate model: {combined_model_path}')
    print(f'Surrogate-only model:              {surrogate_model_path}')
    print(f'Runtime results dir:               {runtime_dir}')

    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print('Could not set GPU memory growth:', repr(e))

    gpu_devices = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpu_devices) > 0
    print('CPU benchmark:', run_cpu)
    print('GPU benchmark:', run_gpu, '| visible GPUs:', gpu_devices)

    X_inverse_test = _load_npy(Path(DATA_DIR) / 'npy' / 'x_test_one_hot_encoding_augmented.npy')
    X_surrogate_test = _load_npy(Path(DATA_DIR) / 'npy' / 'y_test_linear_encoding_scaled.npy')

    with open(_require_file(Path(METADATA_DIR) / 'X_names', 'Hamiltonian column names'), 'r') as f:
        Hamiltonian_column_names = f.read().splitlines()
    qiskit_param_names = np.load(
        _require_file(Path(METADATA_DIR) / 'y_columns.npy', 'Quantum Metal column names'),
        allow_pickle=True,
    ).astype(str).tolist()

    print(f'Combined model benchmark input: {X_inverse_test.shape}')
    print(f'Surrogate-only benchmark input: {X_surrogate_test.shape}')
    print(f'Hamiltonian columns: {Hamiltonian_column_names}')
    print(f'Quantum Metal columns: {qiskit_param_names}')
    print(f'Batch numbers: {batch_numbers}')
    print(f'Warmup={args.n_warmup} | Repeat={args.n_repeat}')

    all_timing_rows = []
    custom_objects = {'ScalerConversionLayer': ScalerConversionLayer}
    common_kwargs = dict(
        batch_numbers=batch_numbers,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
        gpu_available=gpu_available,
    )

    if run_cpu:
        all_timing_rows.extend(benchmark_model_path(
            combined_model_path,
            X_inverse_test,
            benchmark_name='inverse+surrogate',
            device_label='CPU',
            device_name='/CPU:0',
            custom_objects=custom_objects,
            **common_kwargs,
        ))
        all_timing_rows.extend(benchmark_model_path(
            surrogate_model_path,
            X_surrogate_test,
            benchmark_name='surrogate-only',
            device_label='CPU',
            device_name='/CPU:0',
            **common_kwargs,
        ))

    if run_gpu:
        all_timing_rows.extend(benchmark_model_path(
            combined_model_path,
            X_inverse_test,
            benchmark_name='inverse+surrogate',
            device_label='GPU',
            device_name='/GPU:0',
            custom_objects=custom_objects,
            **common_kwargs,
        ))
        all_timing_rows.extend(benchmark_model_path(
            surrogate_model_path,
            X_surrogate_test,
            benchmark_name='surrogate-only',
            device_label='GPU',
            device_name='/GPU:0',
            **common_kwargs,
        ))

    timing_raw_df = pd.DataFrame(all_timing_rows)
    if timing_raw_df.empty:
        raise RuntimeError('No timing rows were produced. Check benchmark settings and visible devices.')

    raw_path = runtime_dir / 'ml_30_runtime_batch_benchmark_raw.csv'
    summary_path = runtime_dir / 'ml_30_runtime_batch_benchmark_summary.csv'
    metadata_path = runtime_dir / 'ml_30_runtime_batch_benchmark_metadata.json'

    timing_raw_df.to_csv(raw_path, index=False)

    ok_df = timing_raw_df[timing_raw_df['status'] == 'ok'].copy()
    if ok_df.empty:
        raise RuntimeError(
            f'No successful timing rows were produced. See errors in {raw_path}'
        )

    summary_df = (
        ok_df
        .groupby(['benchmark', 'device', 'batch_number'], as_index=False)
        .agg(
            mean_total_seconds=('total_seconds', 'mean'),
            std_total_seconds=('total_seconds', 'std'),
            mean_time_per_sample_seconds=('time_per_sample_seconds', 'mean'),
            std_time_per_sample_seconds=('time_per_sample_seconds', 'std'),
            repeats=('time_per_sample_seconds', 'count'),
        )
    )
    summary_df['mean_time_per_sample_ms'] = summary_df['mean_time_per_sample_seconds'] * 1000.0
    summary_df['std_time_per_sample_ms'] = summary_df['std_time_per_sample_seconds'] * 1000.0
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        'created_by': 'ml_30_runtime_batch_benchmark.py',
        'system': f'{platform.system()} {platform.release()}',
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'tensorflow': tf.__version__,
        'keras': package_version('keras'),
        'gpu_available': gpu_available,
        'gpu_devices': [str(g) for g in gpu_devices],
        'batch_numbers': batch_numbers,
        'n_warmup': args.n_warmup,
        'n_repeat': args.n_repeat,
        'run_cpu': run_cpu,
        'run_gpu': run_gpu,
        'combined_model_path': str(combined_model_path),
        'surrogate_model_path': str(surrogate_model_path),
        'ansys_seconds_per_sample': args.ansys_seconds_per_sample,
    }
    with metadata_path.open('w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Saved raw timing data -> {raw_path}')
    print(f'Saved timing summary -> {summary_path}')
    print(f'Saved timing metadata -> {metadata_path}')
    print(summary_df.to_string(index=False))

    for benchmark_name in ['inverse+surrogate', 'surrogate-only']:
        safe_name = benchmark_name.replace('+', '_plus_').replace('-', '_')
        out_path = runtime_dir / f'ml_30_{safe_name}_batch_timing_summary.csv'
        summary_df[summary_df['benchmark'] == benchmark_name].to_csv(out_path, index=False)
        print(f'Saved {benchmark_name} summary -> {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
