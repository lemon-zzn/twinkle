import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hp_acoustic_wave.benchmark import (
    _load_audio_config,
    load_visual_blink_markers,
    reprocess_audio_feature_rows,
)
from hp_acoustic_wave.neural_blink import (
    FEATURE_NAMES,
    NeuralBlinkModel,
    default_model_path,
    feature_vector_from_row,
    make_window_vector,
    predict_proba,
    save_model,
)


@dataclass(frozen=True)
class SessionDataset:
    session: Path
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_times: np.ndarray
    full_x: np.ndarray
    full_y: np.ndarray
    full_times: np.ndarray
    markers: Tuple[float, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class TrainResult:
    model: NeuralBlinkModel
    train_loss: float
    val_f1: float
    val_precision: float
    val_recall: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a CPU neural blink detector from saved sessions.")
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--output", default=str(default_model_path()))
    parser.add_argument("--source", choices=["features", "audio"], default="features")
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--positive-window-s", type=float, default=0.25)
    parser.add_argument("--near-negative-exclude-s", type=float, default=0.0)
    parser.add_argument("--negative-ratio", type=float, default=4.0)
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=0.0,
        help="Positive class loss weight; <=0 uses the sampled negative/positive ratio.",
    )
    parser.add_argument(
        "--hard-negative-rounds",
        type=int,
        default=0,
        help="Mining rounds that add high-scoring non-blink frames back into training.",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=8.0,
        help="Hard negatives mined per positive training sample in each session.",
    )
    parser.add_argument(
        "--hard-negative-threshold",
        type=float,
        default=0.50,
        help="Only mine non-blink frames with model probability at or above this value.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=4.0,
        help="Loss weight multiplier for mined hard negatives.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Per-session fraction of positive and negative samples held out for validation.",
    )
    parser.add_argument(
        "--validation-negative-ratio",
        type=float,
        default=8.0,
        help="Per-session validation negatives sampled per validation positive.",
    )
    parser.add_argument(
        "--recent-session-count",
        type=int,
        default=0,
        help="Give the newest N sessions extra training weight; 0 disables.",
    )
    parser.add_argument("--recent-session-weight", type=float, default=1.0)
    parser.add_argument("--threshold-grid", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99,0.995")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refractory-s", type=float, default=0.75)
    parser.add_argument("--local-peak-radius", type=int, default=2)
    parser.add_argument("--metadata-output", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    datasets = load_datasets(
        Path(args.session_root),
        source=args.source,
        window_size=args.window_size,
        positive_window_s=args.positive_window_s,
        near_negative_exclude_s=args.near_negative_exclude_s,
        negative_ratio=args.negative_ratio,
        validation_fraction=args.validation_fraction,
        validation_negative_ratio=args.validation_negative_ratio,
        recent_session_count=args.recent_session_count,
        recent_session_weight=args.recent_session_weight,
        rng=rng,
    )
    if not datasets:
        raise SystemExit("No sessions with visual blink labels were found.")

    result, validation_rows = train_with_session_validation(
        datasets,
        window_size=args.window_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        positive_class_weight=args.positive_class_weight,
        hard_negative_rounds=args.hard_negative_rounds,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_negative_threshold=args.hard_negative_threshold,
        hard_negative_weight=args.hard_negative_weight,
        threshold_grid=_parse_float_list(args.threshold_grid),
        refractory_s=args.refractory_s,
        local_peak_radius=args.local_peak_radius,
        seed=args.seed,
    )
    save_model(Path(args.output), result.model)
    metadata = {
        "sessions": [dataset.session.name for dataset in datasets],
        "samples": int(sum(dataset.x.shape[0] for dataset in datasets)),
        "weighted_samples": float(sum(dataset.x.shape[0] * dataset.weight for dataset in datasets)),
        "positive_samples": int(sum(np.sum(dataset.y) for dataset in datasets)),
        "validation_samples": int(sum(dataset.val_x.shape[0] for dataset in datasets)),
        "validation_positive_samples": int(sum(np.sum(dataset.val_y) for dataset in datasets)),
        "positive_window_s": float(args.positive_window_s),
        "near_negative_exclude_s": float(args.near_negative_exclude_s),
        "negative_ratio": float(args.negative_ratio),
        "positive_class_weight": float(args.positive_class_weight),
        "hard_negative_rounds": int(args.hard_negative_rounds),
        "hard_negative_ratio": float(args.hard_negative_ratio),
        "hard_negative_threshold": float(args.hard_negative_threshold),
        "hard_negative_weight": float(args.hard_negative_weight),
        "validation_fraction": float(args.validation_fraction),
        "validation_negative_ratio": float(args.validation_negative_ratio),
        "window_size": int(args.window_size),
        "hidden_size": int(args.hidden_size),
        "threshold": float(result.model.threshold),
        "refractory_s": float(result.model.refractory_s),
        "local_peak_radius": int(result.model.local_peak_radius),
        "validation": {
            "f1": float(result.val_f1),
            "precision": float(result.val_precision),
            "recall": float(result.val_recall),
            "rows": validation_rows,
        },
        "feature_names": list(FEATURE_NAMES),
    }
    metadata_path = Path(args.metadata_output) if args.metadata_output else Path(args.output).with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"trained_sessions: {len(datasets)}")
    print(f"samples: {metadata['samples']} positives: {metadata['positive_samples']}")
    print(
        "validation: "
        f"precision={result.val_precision:.3f} recall={result.val_recall:.3f} f1={result.val_f1:.3f}"
    )
    print(f"threshold: {result.model.threshold:.3f}")
    print(f"wrote_model: {Path(args.output)}")
    print(f"wrote_metadata: {metadata_path}")


def load_datasets(
    session_root: Path,
    source: str,
    window_size: int,
    positive_window_s: float,
    near_negative_exclude_s: float,
    negative_ratio: float,
    validation_fraction: float,
    validation_negative_ratio: float,
    recent_session_count: int,
    recent_session_weight: float,
    rng: np.random.Generator,
) -> List[SessionDataset]:
    datasets = []
    session_paths = sorted(Path(session_root).glob("hp_blink_*"))
    recent_names = {
        path.name
        for path in session_paths[-max(0, int(recent_session_count)):]
    }
    for session in session_paths:
        if not (session / "metadata.json").exists():
            continue
        markers = tuple(marker.time_s for marker in load_visual_blink_markers(session))
        if not markers:
            continue
        rows = _feature_rows(session, source)
        if not rows:
            continue
        vectors = [feature_vector_from_row(row) for row in rows]
        times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float32)
        labels = _labels_from_markers(times, markers, positive_window_s, near_negative_exclude_s)
        full_x = np.vstack(
            [make_window_vector(vectors, int(index), int(window_size)) for index in range(len(vectors))]
        )
        train_keep, val_keep = _sample_train_validation_indices(
            labels,
            negative_ratio=negative_ratio,
            validation_fraction=validation_fraction,
            validation_negative_ratio=validation_negative_ratio,
            rng=rng,
        )
        x = full_x[train_keep]
        y = labels[train_keep].astype(np.float32)
        val_x = full_x[val_keep]
        val_y = labels[val_keep].astype(np.float32)
        if np.any(y > 0.5) and np.any(y < 0.5):
            datasets.append(
                SessionDataset(
                    session=session,
                    x=x,
                    y=y,
                    times=times[train_keep],
                    val_x=val_x,
                    val_y=val_y,
                    val_times=times[val_keep],
                    full_x=full_x,
                    full_y=labels.astype(np.float32),
                    full_times=times,
                    markers=markers,
                    weight=(
                        float(recent_session_weight)
                        if session.name in recent_names
                        else 1.0
                    ),
                )
            )
    return datasets


def train_with_session_validation(
    datasets: Sequence[SessionDataset],
    window_size: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    positive_class_weight: float,
    hard_negative_rounds: int,
    hard_negative_ratio: float,
    hard_negative_threshold: float,
    hard_negative_weight: float,
    threshold_grid: Sequence[float],
    refractory_s: float,
    local_peak_radius: int,
    seed: int,
) -> Tuple[TrainResult, List[Dict[str, object]]]:
    all_x = np.vstack([dataset.x for dataset in datasets]).astype(np.float32)
    all_y = np.concatenate([dataset.y for dataset in datasets]).astype(np.float32)
    all_weights = np.concatenate(
        [
            np.full((dataset.y.size,), float(dataset.weight), dtype=np.float32)
            for dataset in datasets
        ]
    )
    train_x = all_x
    train_y = all_y
    train_weights = all_weights
    model = None
    train_loss = 0.0
    for mining_round in range(max(0, int(hard_negative_rounds)) + 1):
        mean, scale = _fit_scaler(train_x)
        model, train_loss = _train_mlp(
            train_x,
            train_y,
            sample_weights=train_weights,
            mean=mean,
            scale=scale,
            hidden_size=hidden_size,
            window_size=window_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            positive_class_weight=positive_class_weight,
            seed=seed + mining_round,
            threshold=0.5,
            refractory_s=refractory_s,
            local_peak_radius=local_peak_radius,
        )
        if mining_round >= max(0, int(hard_negative_rounds)):
            break
        hard_x, hard_y, hard_weights = _mine_hard_negatives(
            datasets,
            model,
            hard_negative_ratio=hard_negative_ratio,
            hard_negative_threshold=hard_negative_threshold,
            hard_negative_weight=hard_negative_weight,
        )
        if hard_x.size == 0:
            break
        train_x = np.vstack([all_x, hard_x]).astype(np.float32)
        train_y = np.concatenate([all_y, hard_y]).astype(np.float32)
        train_weights = np.concatenate([all_weights, hard_weights]).astype(np.float32)
    calibrated_threshold = _choose_model_threshold(
        datasets,
        model,
        threshold_grid=threshold_grid,
        refractory_s=refractory_s,
        local_peak_radius=local_peak_radius,
    )
    model = _copy_model_with_threshold(model, calibrated_threshold)
    val_rows, totals = _validate_model(
        datasets,
        model,
        model.threshold,
        refractory_s=refractory_s,
        local_peak_radius=local_peak_radius,
    )
    return (
        TrainResult(
            model=model,
            train_loss=train_loss,
            val_f1=totals["f1"],
            val_precision=totals["precision"],
            val_recall=totals["recall"],
        ),
        val_rows,
    )


def _choose_threshold(
    datasets: Sequence[SessionDataset],
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    positive_class_weight: float,
    threshold_grid: Sequence[float],
    refractory_s: float,
    local_peak_radius: int,
    seed: int,
) -> float:
    if len(datasets) < 2:
        return 0.5
    prob_sets = []
    for holdout_index, holdout in enumerate(datasets):
        train_sets = [dataset for index, dataset in enumerate(datasets) if index != holdout_index]
        train_x = np.vstack([dataset.x for dataset in train_sets]).astype(np.float32)
        train_y = np.concatenate([dataset.y for dataset in train_sets]).astype(np.float32)
        train_weights = np.concatenate(
            [
                np.full((dataset.y.size,), float(dataset.weight), dtype=np.float32)
                for dataset in train_sets
            ]
        )
        mean, scale = _fit_scaler(train_x)
        model, _ = _train_mlp(
            train_x,
            train_y,
            sample_weights=train_weights,
            mean=mean,
            scale=scale,
            hidden_size=hidden_size,
            window_size=train_x.shape[1] // len(FEATURE_NAMES),
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            positive_class_weight=positive_class_weight,
            seed=seed + holdout_index + 1,
            threshold=0.5,
            refractory_s=refractory_s,
            local_peak_radius=local_peak_radius,
        )
        prob_sets.append((holdout, predict_proba(model, holdout.full_x)))

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in threshold_grid:
        tp = fp = fn = 0
        for dataset, probabilities in prob_sets:
            events = _events_from_probabilities(
                dataset.full_times,
                probabilities,
                threshold=threshold,
                refractory_s=refractory_s,
                local_peak_radius=local_peak_radius,
            )
            summary = _score_event_times(events, dataset.markers)
            tp += summary["tp"]
            fp += summary["fp"]
            fn += summary["fn"]
        precision, recall, f1 = _prf(tp, fp, fn)
        if f1 > best_f1 or (f1 == best_f1 and recall > 0.0 and threshold > best_threshold):
            best_threshold = float(threshold)
            best_f1 = float(f1)
    return best_threshold


def _choose_model_threshold(
    datasets: Sequence[SessionDataset],
    model: NeuralBlinkModel,
    threshold_grid: Sequence[float],
    refractory_s: float,
    local_peak_radius: int,
) -> float:
    prob_sets = [(dataset, predict_proba(model, dataset.val_x)) for dataset in datasets]
    best_threshold = float(model.threshold)
    best_f1 = -1.0
    best_precision = -1.0
    min_recall = 0.65
    for threshold in threshold_grid:
        tp = fp = fn = 0
        for dataset, probabilities in prob_sets:
            summary = _score_sample_predictions(probabilities, dataset.val_y, threshold)
            tp += summary["tp"]
            fp += summary["fp"]
            fn += summary["fn"]
        precision, recall, f1 = _prf(tp, fp, fn)
        objective = (0.35 * recall + 0.65 * precision) if recall >= min_recall else f1 * 0.75
        if objective > best_f1 or (objective == best_f1 and precision > best_precision):
            best_threshold = float(threshold)
            best_f1 = float(objective)
            best_precision = float(precision)
    return best_threshold


def _copy_model_with_threshold(model: NeuralBlinkModel, threshold: float) -> NeuralBlinkModel:
    return NeuralBlinkModel(
        window_size=model.window_size,
        feature_names=model.feature_names,
        mean=model.mean,
        scale=model.scale,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        threshold=float(threshold),
        refractory_s=model.refractory_s,
        local_peak_radius=model.local_peak_radius,
    )


def _validate_model(
    datasets: Sequence[SessionDataset],
    model: NeuralBlinkModel,
    threshold: float,
    refractory_s: float,
    local_peak_radius: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rows = []
    tp = fp = fn = 0
    for dataset in datasets:
        probabilities = predict_proba(model, dataset.val_x)
        summary = _score_sample_predictions(probabilities, dataset.val_y, threshold)
        tp += summary["tp"]
        fp += summary["fp"]
        fn += summary["fn"]
        rows.append(
            {
                "session": dataset.session.name,
                "samples": int(dataset.val_y.size),
                "positives": int(np.sum(dataset.val_y > 0.5)),
                "negatives": int(np.sum(dataset.val_y < 0.5)),
                "predicted_positive": int(summary["tp"] + summary["fp"]),
                **summary,
            }
        )
    precision, recall, f1 = _prf(tp, fp, fn)
    return rows, {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    hidden_size: int,
    window_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    positive_class_weight: float,
    seed: int,
    threshold: float,
    refractory_s: float,
    local_peak_radius: int,
) -> Tuple[NeuralBlinkModel, float]:
    rng = np.random.default_rng(seed)
    x_norm = ((x - mean) / scale).astype(np.float32)
    y_col = y.reshape(-1, 1).astype(np.float32)
    n_samples, input_size = x_norm.shape
    pos = max(float(np.sum(y)), 1.0)
    neg = max(float(y.size - np.sum(y)), 1.0)
    if positive_class_weight > 0.0:
        positive_weight = float(positive_class_weight)
    else:
        positive_weight = min(neg / pos, 20.0)
    base_weight = np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1)
    sample_weight = (base_weight * np.where(y_col > 0.5, positive_weight, 1.0)).astype(np.float32)

    w1 = (rng.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, hidden_size))).astype(np.float32)
    b1 = np.zeros((hidden_size,), dtype=np.float32)
    w2 = (rng.normal(0.0, np.sqrt(2.0 / hidden_size), size=(hidden_size, 1))).astype(np.float32)
    initial_bias = np.log(pos / neg)
    b2 = np.asarray([initial_bias], dtype=np.float32)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(param) for param in params]
    v = [np.zeros_like(param) for param in params]
    loss = 0.0
    for epoch in range(1, int(epochs) + 1):
        hidden_pre = x_norm @ w1 + b1
        hidden = np.maximum(0.0, hidden_pre)
        logits = hidden @ w2 + b2
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        loss_terms = -(
            y_col * np.log(probabilities + 1e-7)
            + (1.0 - y_col) * np.log(1.0 - probabilities + 1e-7)
        )
        loss = float(np.sum(loss_terms * sample_weight) / np.sum(sample_weight))
        loss += 0.5 * weight_decay * (float(np.sum(w1 * w1)) + float(np.sum(w2 * w2)))

        dlogits = (probabilities - y_col) * sample_weight / np.sum(sample_weight)
        dw2 = hidden.T @ dlogits + weight_decay * w2
        db2 = np.sum(dlogits, axis=0)
        dhidden = dlogits @ w2.T
        dhidden_pre = dhidden * (hidden_pre > 0.0)
        dw1 = x_norm.T @ dhidden_pre + weight_decay * w1
        db1 = np.sum(dhidden_pre, axis=0)
        grads = [dw1.astype(np.float32), db1.astype(np.float32), dw2.astype(np.float32), db2.astype(np.float32)]
        for i, (param, grad) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad
            v[i] = beta2 * v[i] + (1.0 - beta2) * (grad * grad)
            m_hat = m[i] / (1.0 - beta1 ** epoch)
            v_hat = v[i] / (1.0 - beta2 ** epoch)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    model = NeuralBlinkModel(
        window_size=int(window_size),
        feature_names=FEATURE_NAMES,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        threshold=float(threshold),
        refractory_s=float(refractory_s),
        local_peak_radius=int(local_peak_radius),
    )
    return model, loss


def _feature_rows(session: Path, source: str) -> List[dict]:
    if source == "features":
        path = session / "features.csv"
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    audio_config = _load_audio_config(session)
    signal_mode = str(audio_config.get("signal_mode", "tone"))
    emission = str(audio_config.get("fmcw_emission", "linear"))
    if signal_mode != "fmcw":
        emission = "cw_single"
    return reprocess_audio_feature_rows(
        session / "audio.wav",
        sample_rate=int(audio_config.get("sample_rate", 48_000)),
        tone_hz=float(audio_config.get("tone_hz", 18_500.0)),
        chunk_size=int(audio_config.get("chunk_size", 1024)),
        signal_mode=signal_mode,
        fmcw_freq_low=float(audio_config.get("fmcw_freq_low", 17_000.0)),
        fmcw_freq_high=float(audio_config.get("fmcw_freq_high", 23_000.0)),
        fmcw_chirp_duration=float(audio_config.get("fmcw_chirp_duration", 0.05)),
        fmcw_range_bin=int(audio_config.get("fmcw_range_bin", 15)),
        fmcw_lowpass_cutoff=float(audio_config.get("fmcw_lowpass_cutoff", 5_000.0)),
        fmcw_motion_amplitude_floor=float(audio_config.get("fmcw_motion_amplitude_floor", 0.02)),
        fmcw_output_amplitude=float(audio_config.get("output_amplitude", 0.2)),
        fmcw_emission=emission,
    )


def _labels_from_markers(
    times: np.ndarray,
    markers: Sequence[float],
    positive_window_s: float,
    near_negative_exclude_s: float,
) -> np.ndarray:
    labels = np.full(times.shape, -1.0, dtype=np.float32)
    for index, time_s in enumerate(times):
        nearest = min(abs(float(time_s) - marker) for marker in markers)
        if nearest <= positive_window_s:
            labels[index] = 1.0
        elif near_negative_exclude_s <= 0.0 or nearest >= near_negative_exclude_s:
            labels[index] = 0.0
    return labels


def _sample_train_validation_indices(
    labels: np.ndarray,
    negative_ratio: float,
    validation_fraction: float,
    validation_negative_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(labels > 0.5)
    negative = np.flatnonzero(labels == 0.0)
    if positive.size == 0 or negative.size == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty

    pos_order = rng.permutation(positive)
    val_pos_count = min(
        positive.size - 1 if positive.size > 1 else 1,
        max(1, int(round(positive.size * max(0.0, min(0.8, validation_fraction))))),
    )
    val_positive = pos_order[:val_pos_count]
    train_positive = pos_order[val_pos_count:]
    if train_positive.size == 0:
        train_positive = val_positive[:1]

    neg_order = rng.permutation(negative)
    val_neg_count = min(
        negative.size,
        max(1, int(round(val_positive.size * max(1.0, validation_negative_ratio)))),
    )
    val_negative = neg_order[:val_neg_count]
    remaining_negative = neg_order[val_neg_count:]
    if remaining_negative.size == 0:
        remaining_negative = neg_order
    train_neg_count = min(
        remaining_negative.size,
        max(1, int(round(train_positive.size * max(1.0, negative_ratio)))),
    )
    train_negative = remaining_negative[:train_neg_count]

    train_keep = np.concatenate([train_positive, train_negative]).astype(np.int64, copy=False)
    val_keep = np.concatenate([val_positive, val_negative]).astype(np.int64, copy=False)
    rng.shuffle(train_keep)
    rng.shuffle(val_keep)
    return train_keep, val_keep


def _sample_indices(labels: np.ndarray, negative_ratio: float, rng: np.random.Generator) -> np.ndarray:
    positive = np.flatnonzero(labels > 0.5)
    negative = np.flatnonzero(labels == 0.0)
    if positive.size == 0 or negative.size == 0:
        return np.asarray([], dtype=np.int64)
    neg_count = min(negative.size, max(1, int(round(positive.size * negative_ratio))))
    sampled_negative = rng.choice(negative, size=neg_count, replace=False)
    keep = np.concatenate([positive, sampled_negative])
    rng.shuffle(keep)
    return keep


def _mine_hard_negatives(
    datasets: Sequence[SessionDataset],
    model: NeuralBlinkModel,
    hard_negative_ratio: float,
    hard_negative_threshold: float,
    hard_negative_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mined_x = []
    mined_weights = []
    threshold = float(hard_negative_threshold)
    for dataset in datasets:
        negative_indices = np.flatnonzero(dataset.full_y == 0.0)
        if negative_indices.size == 0:
            continue
        probabilities = predict_proba(model, dataset.full_x[negative_indices])
        hard_mask = probabilities >= threshold
        hard_indices = negative_indices[hard_mask]
        if hard_indices.size == 0:
            continue
        order = np.argsort(probabilities[hard_mask])[::-1]
        max_count = max(1, int(round(np.sum(dataset.y > 0.5) * max(1.0, hard_negative_ratio))))
        selected = hard_indices[order[: min(hard_indices.size, max_count)]]
        mined_x.append(dataset.full_x[selected])
        mined_weights.append(
            np.full(
                (selected.size,),
                float(dataset.weight) * max(1.0, float(hard_negative_weight)),
                dtype=np.float32,
            )
        )
    if not mined_x:
        input_size = int(model.window_size) * len(FEATURE_NAMES)
        return (
            np.empty((0, input_size), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    x = np.vstack(mined_x).astype(np.float32)
    y = np.zeros((x.shape[0],), dtype=np.float32)
    weights = np.concatenate(mined_weights).astype(np.float32)
    return x, y, weights


def _score_sample_predictions(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, int]:
    predicted = np.asarray(probabilities, dtype=np.float32) >= float(threshold)
    truth = np.asarray(labels, dtype=np.float32) > 0.5
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    return {"tp": tp, "fp": fp, "fn": fn}


def _fit_scaler(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0).astype(np.float32)
    scale = np.std(x, axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    return mean, scale


def _events_from_probabilities(
    times: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    refractory_s: float,
    local_peak_radius: int,
) -> List[float]:
    events: List[float] = []
    radius = max(1, int(local_peak_radius))
    last_event = -1e9
    for index, probability in enumerate(probabilities):
        if probability < threshold:
            continue
        left = max(0, index - radius)
        right = min(probabilities.size, index + radius + 1)
        if probability < float(np.max(probabilities[left:right])):
            continue
        time_s = float(times[index])
        if time_s - last_event >= refractory_s:
            events.append(time_s)
            last_event = time_s
    return events


def _score_event_times(events: Sequence[float], markers: Sequence[float]) -> Dict[str, int]:
    candidates = []
    for marker_index, marker in enumerate(markers):
        for event_index, event_time in enumerate(events):
            if marker - 0.5 <= event_time <= marker + 0.5:
                candidates.append((abs(event_time - marker), event_index, marker_index))
    candidates.sort()
    used_events = set()
    used_markers = set()
    for _, event_index, marker_index in candidates:
        if event_index in used_events or marker_index in used_markers:
            continue
        used_events.add(event_index)
        used_markers.add(marker_index)
    tp = len(used_events)
    fp = max(0, len(events) - tp)
    fn = max(0, len(markers) - tp)
    return {"tp": tp, "fp": fp, "fn": fn}


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _parse_float_list(value: str) -> Tuple[float, ...]:
    return tuple(float(token.strip()) for token in str(value).split(",") if token.strip())


if __name__ == "__main__":
    main()
