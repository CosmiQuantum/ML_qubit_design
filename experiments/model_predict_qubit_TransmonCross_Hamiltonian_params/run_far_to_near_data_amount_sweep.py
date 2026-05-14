#!/usr/bin/env python3
"""Run a far-to-near training-fraction sweep for the transmon inverse model.

The subset selection is intentionally ordered by geometry-space support:
training samples farthest from the fixed validation/test set are added first,
and progressively closer samples are added as the fraction increases.

This script uses a lightweight NumPy implementation of the tandem
inverse-plus-surrogate model so the sweep can be regenerated in environments
without TensorFlow. The architecture mirrors the manuscript model dimensions:
a 3 -> 736 -> 2 frozen forward surrogate and a 2 -> 64 -> 3 inverse MLP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


EXPERIMENT_DIR = Path(__file__).resolve().parent
METADATA_PATH = EXPERIMENT_DIR / "metadata" / "qubit-TransmonCross-Hamiltonian_params.json"
OUT_PATH = EXPERIMENT_DIR / "data_amount_sweep_far_to_near.csv"

FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)
SEEDS = (0, 1, 2)
LEAKY_SLOPE = 0.01
EPS = 1e-12


@dataclass
class Scaler:
    min_: np.ndarray
    max_: np.ndarray

    @property
    def range_(self) -> np.ndarray:
        return np.maximum(self.max_ - self.min_, EPS)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min_) / self.range_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.range_ + self.min_


def parse_um(value: object) -> float:
    text = str(value).strip()
    if text.endswith("um"):
        return float(text[:-2])
    return float(text)


def load_arrays() -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(METADATA_PATH.read_text())
    hamiltonian = []
    geometry = []
    for row in data:
        h = row["Hamiltonian_params"]
        opts = row["design"]["design_options"]
        readout = opts["connection_pads"]["readout"]
        hamiltonian.append(
            [
                float(h["qubit_frequency_GHz"]),
                float(h["anharmonicity_MHz"]),
            ]
        )
        geometry.append(
            [
                parse_um(readout["claw_length"]),
                parse_um(readout["ground_spacing"]),
                parse_um(opts["cross_length"]),
            ]
        )
    return np.asarray(hamiltonian, dtype=np.float64), np.asarray(geometry, dtype=np.float64)


def fixed_split(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n_rows)
    train_idx, val_test_idx = train_test_split(
        idx,
        test_size=0.3,
        random_state=42,
        shuffle=True,
    )
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=0.5,
        random_state=42,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def init_mlp(in_dim: int, hidden_dim: int, out_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, out_dim)),
        "b2": np.zeros(out_dim),
    }


def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, z, LEAKY_SLOPE * z)


def leaky_grad(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0, LEAKY_SLOPE)


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = x @ params["w1"] + params["b1"]
    a1 = leaky_relu(z1)
    y = a1 @ params["w2"] + params["b2"]
    return y, {"x": x, "z1": z1, "a1": a1}


def backward_from_output(
    params: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    grad_y: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    grad_w2 = cache["a1"].T @ grad_y
    grad_b2 = grad_y.sum(axis=0)
    grad_a1 = grad_y @ params["w2"].T
    grad_z1 = grad_a1 * leaky_grad(cache["z1"])
    grad_w1 = cache["x"].T @ grad_z1
    grad_b1 = grad_z1.sum(axis=0)
    grad_x = grad_z1 @ params["w1"].T
    grads = {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}
    return grads, grad_x


def adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, dict[str, np.ndarray] | int],
    lr: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    state["t"] = int(state["t"]) + 1
    t = int(state["t"])
    for key, param in params.items():
        m = state["m"][key]  # type: ignore[index]
        v = state["v"][key]  # type: ignore[index]
        grad = grads[key]
        m[:] = beta1 * m + (1.0 - beta1) * grad
        v[:] = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def make_adam_state(params: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray] | int]:
    return {
        "t": 0,
        "m": {key: np.zeros_like(value) for key, value in params.items()},
        "v": {key: np.zeros_like(value) for key, value in params.items()},
    }


def train_surrogate(
    geom_train: np.ndarray,
    h_train: np.ndarray,
    seed: int = 123,
    epochs: int = 1200,
    batch_size: int = 128,
    lr: float = 0.001,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = init_mlp(3, 736, 2, rng)
    state = make_adam_state(params)
    n = len(geom_train)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            xb = geom_train[batch_idx]
            yb = h_train[batch_idx]
            pred, cache = forward(params, xb)
            grad_y = 2.0 * (pred - yb) / pred.size
            grads, _ = backward_from_output(params, cache, grad_y)
            adam_update(params, grads, state, lr)
    return params


def surrogate_input_grad(
    surrogate: dict[str, np.ndarray],
    geom_scaled: np.ndarray,
    h_target_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h_pred, cache = forward(surrogate, geom_scaled)
    grad_h = 2.0 * (h_pred - h_target_scaled) / h_pred.size
    _, grad_geom = backward_from_output(surrogate, cache, grad_h)
    return h_pred, grad_geom


def train_inverse(
    h_subset: np.ndarray,
    surrogate: dict[str, np.ndarray],
    seed: int,
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 0.002,
    boundary_penalty: float = 1.0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = init_mlp(2, 64, 3, rng)
    state = make_adam_state(params)
    n = len(h_subset)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            xb = h_subset[batch_idx]
            geom_pred, inverse_cache = forward(params, xb)
            _, grad_geom = surrogate_input_grad(surrogate, geom_pred, xb)

            below = geom_pred < 0.0
            above = geom_pred > 1.0
            penalty_grad = np.zeros_like(geom_pred)
            penalty_grad[below] = 2.0 * geom_pred[below] / geom_pred.size
            penalty_grad[above] = 2.0 * (geom_pred[above] - 1.0) / geom_pred.size
            grad_geom = grad_geom + boundary_penalty * penalty_grad

            grads, _ = backward_from_output(params, inverse_cache, grad_geom)
            adam_update(params, grads, state, lr)
    return params


def evaluate_percent_error(
    inverse: dict[str, np.ndarray],
    surrogate: dict[str, np.ndarray],
    h_scaled: np.ndarray,
    h_unscaled: np.ndarray,
    h_scaler: Scaler,
) -> dict[str, float]:
    geom_pred, _ = forward(inverse, h_scaled)
    h_pred_scaled, _ = forward(surrogate, geom_pred)
    h_pred = h_scaler.inverse_transform(h_pred_scaled)
    pct = 100.0 * np.abs(h_pred - h_unscaled) / np.maximum(np.abs(h_unscaled), EPS)
    return {
        "omega_q_mean_pct": float(np.mean(pct[:, 0])),
        "alpha_mean_pct": float(np.mean(pct[:, 1])),
        "mean_hamiltonian_pct": float(np.mean(pct)),
    }


def main() -> None:
    h_raw, geom_raw = load_arrays()
    train_idx, val_idx, test_idx = fixed_split(len(h_raw))

    h_scaler = Scaler(h_raw[train_idx].min(axis=0), h_raw[train_idx].max(axis=0))
    geom_scaler = Scaler(geom_raw[train_idx].min(axis=0), geom_raw[train_idx].max(axis=0))
    h_scaled = h_scaler.transform(h_raw)
    geom_scaled = geom_scaler.transform(geom_raw)

    train_geom = geom_scaled[train_idx]
    heldout_geom = geom_scaled[np.concatenate([val_idx, test_idx])]
    nn = NearestNeighbors(n_neighbors=1).fit(heldout_geom)
    train_nn_distance = nn.kneighbors(train_geom, return_distance=True)[0][:, 0]
    far_to_near_order = np.argsort(-train_nn_distance)

    print("Training frozen forward surrogate on the full training split...")
    surrogate = train_surrogate(geom_scaled[train_idx], h_scaled[train_idx])
    surrogate_test = evaluate_surrogate(surrogate, geom_scaled[test_idx], h_raw[test_idx], h_scaler)
    print(f"Frozen surrogate test mean Hamiltonian error: {surrogate_test:.3f}%")

    rows = []
    n_train = len(train_idx)
    for fraction in FRACTIONS:
        n_subset = max(1, int(round(fraction * n_train)))
        subset_local = far_to_near_order[:n_subset]
        subset_idx = train_idx[subset_local]
        subset_dist = train_nn_distance[subset_local]
        print(f"Training inverse models with {fraction:.0%} of training set ({n_subset} samples)...")
        for seed in SEEDS:
            inverse = train_inverse(h_scaled[subset_idx], surrogate, seed=seed)
            train_metrics = evaluate_percent_error(
                inverse,
                surrogate,
                h_scaled[subset_idx],
                h_raw[subset_idx],
                h_scaler,
            )
            val_metrics = evaluate_percent_error(
                inverse,
                surrogate,
                h_scaled[val_idx],
                h_raw[val_idx],
                h_scaler,
            )
            test_metrics = evaluate_percent_error(
                inverse,
                surrogate,
                h_scaled[test_idx],
                h_raw[test_idx],
                h_scaler,
            )
            rows.append(
                {
                    "selection_method": "far_to_near_by_heldout_geometry_nn_distance",
                    "fraction": fraction,
                    "training_percent": fraction * 100.0,
                    "n_samples": n_subset,
                    "seed": seed,
                    "subset_nn_distance_min": float(np.min(subset_dist)),
                    "subset_nn_distance_median": float(np.median(subset_dist)),
                    "subset_nn_distance_max": float(np.max(subset_dist)),
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH.relative_to(EXPERIMENT_DIR.parent.parent.parent)}")


def evaluate_surrogate(
    surrogate: dict[str, np.ndarray],
    geom_scaled: np.ndarray,
    h_unscaled: np.ndarray,
    h_scaler: Scaler,
) -> float:
    h_pred_scaled, _ = forward(surrogate, geom_scaled)
    h_pred = h_scaler.inverse_transform(h_pred_scaled)
    pct = 100.0 * np.abs(h_pred - h_unscaled) / np.maximum(np.abs(h_unscaled), EPS)
    return float(np.mean(pct))


if __name__ == "__main__":
    main()
