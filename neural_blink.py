from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from hp_acoustic_wave.dsp import ChunkFeature


MODEL_VERSION = 1
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "neural_blink_model.npz"


FEATURE_NAMES = (
    "i",
    "q",
    "amplitude",
    "amplitude_delta",
    "phase_delta",
    "phase_pair_delta",
    "phase_pair_vote_ratio",
    "phase_pair_consistency",
    "phase_pair_candidate_count",
    "motion_energy",
    "rms",
    "peak_abs",
    "range_spread_ratio",
    "range_dominance_ratio",
    "phase_sin",
    "phase_cos",
)


@dataclass(frozen=True)
class NeuralBlinkModel:
    window_size: int
    feature_names: Sequence[str]
    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    threshold: float
    refractory_s: float
    local_peak_radius: int


def default_model_path() -> Path:
    return DEFAULT_MODEL_PATH


def feature_vector_from_chunk(feature: ChunkFeature) -> np.ndarray:
    phase = _finite(float(feature.phase))
    values = (
        _finite(float(feature.i_value)),
        _finite(float(feature.q_value)),
        _finite(float(feature.amplitude)),
        _finite(float(feature.amplitude_delta)),
        _finite(float(feature.phase_delta)),
        _finite(float(feature.phase_pair_delta or 0.0)),
        _finite(float(feature.phase_pair_vote_ratio or 0.0)),
        _finite(float(feature.phase_pair_consistency or 0.0)),
        _finite(float(feature.phase_pair_candidate_count or 0.0)) / 10.0,
        _finite(float(feature.motion_energy)),
        _finite(float(feature.rms)),
        _finite(float(feature.peak_abs)),
        _finite(float(feature.range_spread_ratio or 0.0)),
        _finite(float(feature.range_dominance_ratio or 0.0)),
        float(np.sin(phase)),
        float(np.cos(phase)),
    )
    return np.asarray(values, dtype=np.float32)


def feature_vector_from_row(row: dict) -> np.ndarray:
    phase = _row_float(row, "phase")
    values = (
        _row_float(row, "i"),
        _row_float(row, "q"),
        _row_float(row, "amplitude"),
        _row_float(row, "amplitude_delta"),
        _row_float(row, "phase_delta"),
        _row_float(row, "phase_pair_delta"),
        _row_float(row, "phase_pair_vote_ratio"),
        _row_float(row, "phase_pair_consistency"),
        _row_float(row, "phase_pair_candidate_count") / 10.0,
        _row_float(row, "motion_energy"),
        _row_float(row, "rms"),
        _row_float(row, "peak_abs"),
        _row_float(row, "range_spread_ratio"),
        _row_float(row, "range_dominance_ratio"),
        float(np.sin(phase)),
        float(np.cos(phase)),
    )
    return np.asarray(values, dtype=np.float32)


def make_window_vector(vectors: Sequence[np.ndarray], center_index: int, window_size: int) -> np.ndarray:
    if not vectors:
        return np.zeros((int(window_size) * len(FEATURE_NAMES),), dtype=np.float32)
    radius = int(window_size) // 2
    pieces: List[np.ndarray] = []
    last_index = len(vectors) - 1
    for offset in range(-radius, radius + 1):
        index = min(max(center_index + offset, 0), last_index)
        pieces.append(np.asarray(vectors[index], dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32, copy=False)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def predict_proba(model: NeuralBlinkModel, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x_norm = (x - model.mean) / model.scale
    hidden = np.maximum(0.0, x_norm @ model.w1 + model.b1)
    logits = hidden @ model.w2 + model.b2
    return sigmoid(logits).reshape(-1)


def save_model(path: Path, model: NeuralBlinkModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray([MODEL_VERSION], dtype=np.int32),
        window_size=np.asarray([int(model.window_size)], dtype=np.int32),
        feature_names=np.asarray(list(model.feature_names)),
        mean=model.mean.astype(np.float32),
        scale=model.scale.astype(np.float32),
        w1=model.w1.astype(np.float32),
        b1=model.b1.astype(np.float32),
        w2=model.w2.astype(np.float32),
        b2=model.b2.astype(np.float32),
        threshold=np.asarray([float(model.threshold)], dtype=np.float32),
        refractory_s=np.asarray([float(model.refractory_s)], dtype=np.float32),
        local_peak_radius=np.asarray([int(model.local_peak_radius)], dtype=np.int32),
    )


def load_model(path: Optional[Union[Path, str]] = None) -> NeuralBlinkModel:
    model_path = Path(path) if path else DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"neural blink model not found: {model_path}. "
            "Run train_neural_blink.py to create it."
        )
    with np.load(model_path, allow_pickle=False) as data:
        version = int(data["version"][0]) if "version" in data else 0
        if version != MODEL_VERSION:
            raise ValueError(f"unsupported neural blink model version: {version}")
        feature_names = tuple(str(item) for item in data["feature_names"])
        if feature_names != FEATURE_NAMES:
            raise ValueError("neural blink model feature set does not match this code")
        return NeuralBlinkModel(
            window_size=int(data["window_size"][0]),
            feature_names=feature_names,
            mean=np.asarray(data["mean"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
            w1=np.asarray(data["w1"], dtype=np.float32),
            b1=np.asarray(data["b1"], dtype=np.float32),
            w2=np.asarray(data["w2"], dtype=np.float32),
            b2=np.asarray(data["b2"], dtype=np.float32),
            threshold=float(data["threshold"][0]),
            refractory_s=float(data["refractory_s"][0]),
            local_peak_radius=int(data["local_peak_radius"][0]),
        )


def _row_float(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return 0.0
    try:
        return _finite(float(value))
    except (TypeError, ValueError):
        return 0.0


def _finite(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)
