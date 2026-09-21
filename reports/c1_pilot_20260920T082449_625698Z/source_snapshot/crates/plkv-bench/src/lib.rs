//! Placeholder benchmark entrypoints.

pub fn benchmark_matrix() -> &'static [&'static str] {
    &[
        "memory-model",
        "block-table",
        "decode-attention",
        "paged-decode",
        "latent-kv",
    ]
}

#[cfg(test)]
mod tests {
    use super::benchmark_matrix;

    #[test]
    fn benchmark_matrix_has_expected_entries() {
        assert!(benchmark_matrix().contains(&"memory-model"));
    }
}
