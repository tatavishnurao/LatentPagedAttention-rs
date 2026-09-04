"""Validate the P0 preprojected GPU result against the Python model_small oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from latent_paged_attention.attention_ref import (
    direct_paged_latent_gqa_fp16_storage_runtime_intermediates_ref,
)


def _metrics(
    actual_parts: tuple[np.ndarray, ...], reference_parts: tuple[np.ndarray, ...]
) -> dict[str, float]:
    actual = np.concatenate([part.reshape(-1).astype(np.float64) for part in actual_parts])
    reference = np.concatenate([part.reshape(-1).astype(np.float64) for part in reference_parts])
    absolute = np.abs(actual - reference)
    relative = absolute / np.maximum(np.abs(reference), 1.0e-12)
    probabilities = actual_parts[1]
    return {
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(absolute * absolute))),
        "max_relative_error": float(relative.max()),
        "max_probability_row_sum_error": float(
            np.max(np.abs(probabilities.sum(axis=-1) - 1.0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    inputs = case["inputs"]

    q = np.asarray(inputs["q"], dtype=np.float32).reshape(
        1, case["q_heads"], case["head_dim"]
    )
    latent = np.asarray(inputs["latent_storage_f16_as_f32"], dtype=np.float32).reshape(
        case["num_physical_blocks"], case["block_size"], case["latent_dim"]
    )
    table = np.asarray(inputs["block_table"], dtype=np.int32)
    k_projection = np.asarray(inputs["k_projection"], dtype=np.float32).reshape(
        case["latent_dim"], case["kv_heads"] * case["head_dim"]
    )
    v_projection = np.asarray(inputs["v_projection"], dtype=np.float32).reshape(
        case["latent_dim"], case["kv_heads"] * case["head_dim"]
    )

    reference = direct_paged_latent_gqa_fp16_storage_runtime_intermediates_ref(
        q,
        latent,
        table,
        case["max_seq_len"],
        case["active_seq_len"],
        case["block_size"],
        k_projection,
        v_projection,
        q_heads=case["q_heads"],
        kv_heads=case["kv_heads"],
        head_dim=case["head_dim"],
        group_size=case["group_size"],
    )
    actual = (
        np.asarray(case["scores"], dtype=np.float32).reshape(reference[0].shape),
        np.asarray(case["probabilities"], dtype=np.float32).reshape(reference[1].shape),
        np.asarray(case["context"], dtype=np.float32).reshape(reference[2].shape),
    )
    result = {
        "case": str(args.case),
        "comparison": "latent_preprojected_gpu_vs_python_oracle",
        "metrics": _metrics(actual, reference),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
