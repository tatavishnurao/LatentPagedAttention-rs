"""Mechanically specialize the unchanged P0 kernels/harness for P1 lengths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072)


def specialize_kernel(source: str, length: int, blocks: int, kind: str) -> str:
    suffix = str(length)
    if kind == "full":
        source = source.replace("pub mod full_kv_baseline_kernel", f"pub mod full_kv_baseline_kernel_{suffix}")
        source = source.replace("model_small_full_kv_scores_fp16_storage", f"model_small_full_kv_scores_fp16_storage_{suffix}")
        source = source.replace("model_small_full_kv_context_fp16_storage", f"model_small_full_kv_context_fp16_storage_{suffix}")
        source = source.replace("1024", str(length))
    elif kind == "latent":
        source = source.replace("pub mod model_profile_kernel", f"pub mod model_profile_kernel_{suffix}")
        source = source.replace("model_small_scores_fp16_storage", f"model_small_scores_fp16_storage_{suffix}")
        source = source.replace("model_small_softmax_1024_runtime", f"model_small_softmax_{suffix}_runtime")
        source = source.replace("model_small_context_fp16_storage", f"model_small_context_fp16_storage_{suffix}")
        source = source.replace("1024", str(length))
    else:
        source = source.replace("pub mod model_profile_preprojected_kernel", f"pub mod model_profile_preprojected_kernel_{suffix}")
        source = source.replace("model_small_project_query_once", f"model_small_project_query_once_{suffix}")
        source = source.replace("model_small_scores_fp16_storage_preprojected", f"model_small_scores_fp16_storage_preprojected_{suffix}")
    source = source.replace("Tensor<i32, { [64] }>", f"Tensor<i32, {{ [{blocks}] }}>")
    source = source.replace("Tile<i32, { [64] }>", f"Tile<i32, {{ [{blocks}] }}>")
    source = source.replace("const_shape![64], [0]", f"const_shape![{blocks}], [0]")
    source = source.replace("0i32..64i32", f"0i32..{blocks}i32")
    return source


def specialize_rtable(source: str, length: int, blocks: int, kind: str) -> str:
    suffix = str(length)
    if kind == "full":
        source = source.replace(
            "pub mod p15b_full_kv_baseline_kernel",
            f"pub mod p15b_full_kv_baseline_kernel_{suffix}",
        )
        source = source.replace(
            "model_small_full_kv_scores_fp16_storage",
            f"model_small_full_kv_scores_fp16_storage_rtable_{suffix}",
        )
        source = source.replace(
            "model_small_full_kv_context_fp16_storage",
            f"model_small_full_kv_context_fp16_storage_rtable_{suffix}",
        )
    else:
        source = source.replace(
            "pub mod p15b_model_profile_kernel",
            f"pub mod p15b_model_profile_kernel_{suffix}",
        )
        source = source.replace(
            "model_small_scores_fp16_storage",
            f"model_small_scores_fp16_storage_rtable_{suffix}",
        )
        source = source.replace(
            "model_small_softmax_1024_runtime",
            f"model_small_softmax_{suffix}_runtime",  # shared unchanged softmax
        )
        source = source.replace(
            "model_small_context_fp16_storage",
            f"model_small_context_fp16_storage_rtable_{suffix}",
        )
    source = source.replace("1024", str(length))
    source = source.replace("Tensor<i32, { [64] }>", f"Tensor<i32, {{ [{blocks}] }}>")
    source = source.replace("0i32..64i32", f"0i32..{blocks}i32")
    return source


def main() -> None:
    full = (ROOT / "crates/plkv-kernels/src/cutile/full_kv_baseline.rs").read_text()
    latent = (ROOT / "crates/plkv-kernels/src/cutile/model_profile.rs").read_text()
    pre = (ROOT / "crates/plkv-kernels/src/cutile/model_profile_preprojected.rs").read_text()
    chunks = ["// GENERATED mechanically from the P0 kernels; logic and tile shapes are unchanged.\n"]
    for length in LENGTHS:
        blocks = length // 16
        chunks.extend(
            [
                specialize_kernel(full, length, blocks, "full"),
                specialize_kernel(latent, length, blocks, "latent"),
                specialize_kernel(pre, length, blocks, "pre"),
            ]
        )
    out = ROOT / "crates/plkv-kernels/src/cutile/p1_sequence_kernels.rs"
    out.write_text("\n".join(chunks), encoding="utf-8")

    rtable_full = (ROOT / "crates/plkv-kernels/src/cutile/p15b_full_kv_baseline.rs").read_text()
    rtable_latent = (ROOT / "crates/plkv-kernels/src/cutile/p15b_model_profile.rs").read_text()
    rtable_chunks = ["// GENERATED R-TABLE kernels; A0/B0 sources remain unchanged.\n"]
    for length in LENGTHS:
        blocks = length // 16
        rtable_chunks.extend(
            [
                specialize_rtable(rtable_full, length, blocks, "full"),
                specialize_rtable(rtable_latent, length, blocks, "latent"),
            ]
        )
    (ROOT / "crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs").write_text(
        "\n".join(rtable_chunks), encoding="utf-8"
    )

    template = (ROOT / "crates/plkv-kernels/examples/p0_gpu_baseline.rs").read_text()
    for length in LENGTHS:
        blocks = length // 16
        text = template
        text = text.replace("use plkv_kernels::cutile::full_kv_baseline::full_kv_baseline_kernel;", f"use plkv_kernels::cutile::p1_sequence_kernels::full_kv_baseline_kernel_{length};")
        text = text.replace("use plkv_kernels::cutile::model_profile::model_profile_kernel;", f"use plkv_kernels::cutile::p1_sequence_kernels::model_profile_kernel_{length};")
        text = text.replace("use plkv_kernels::cutile::model_profile_preprojected::model_profile_preprojected_kernel;", f"use plkv_kernels::cutile::p1_sequence_kernels::model_profile_preprojected_kernel_{length};")
        text = text.replace("full_kv_baseline_kernel::", f"full_kv_baseline_kernel_{length}::")
        text = text.replace("model_profile_kernel::", f"model_profile_kernel_{length}::")
        text = text.replace("model_profile_preprojected_kernel::", f"model_profile_preprojected_kernel_{length}::")
        text = text.replace("model_small_full_kv_scores_fp16_storage(", f"model_small_full_kv_scores_fp16_storage_{length}(")
        text = text.replace("model_small_full_kv_context_fp16_storage(", f"model_small_full_kv_context_fp16_storage_{length}(")
        text = text.replace("model_small_scores_fp16_storage(", f"model_small_scores_fp16_storage_{length}(")
        text = text.replace("model_small_softmax_1024_runtime(", f"model_small_softmax_{length}_runtime(")
        text = text.replace("model_small_context_fp16_storage(", f"model_small_context_fp16_storage_{length}(")
        text = text.replace("model_small_project_query_once(", f"model_small_project_query_once_{length}(")
        text = text.replace("model_small_scores_fp16_storage_preprojected(", f"model_small_scores_fp16_storage_preprojected_{length}(")
        text = text.replace("const MAX_SEQ_LEN: usize = 1024;", f"const MAX_SEQ_LEN: usize = {length};")
        text = text.replace("const ACTIVE_SEQ_LEN: usize = 1024;", f"const ACTIVE_SEQ_LEN: usize = {length};")
        text = text.replace("const NUM_PHYSICAL_BLOCKS: usize = 64;", f"const NUM_PHYSICAL_BLOCKS: usize = {blocks};")
        (ROOT / f"crates/plkv-kernels/examples/p1_seq_{length}.rs").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
