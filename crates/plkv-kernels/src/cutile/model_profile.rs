#[cutile::module]
pub mod model_profile_kernel {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [64] }>, logical: i32) -> i32 {
        let table_tile: Tile<i32, { [64] }> = table.load_tile(const_shape![64], [0]);
        let selected: Tile<i32, { [1] }> = extract(table_tile, [scalar_to_tile(logical)]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 1024] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [64] }>,
        head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [32] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let latent: Tile<f32, { [16, 32] }> = convert_tile(latent_fp16.load_tile(
            const_shape![16, 32],
            [physical_block(table, logical_block), 0],
        ));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 32])
                * latent,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_scores_fp16_storage(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [64] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
        k_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let kp: Tile<f32, { [32, 64] }> =
            k_projection.load_tile(const_shape![32, 64], [kv_head, 0]);
        let projected: Tile<f32, { [32] }> =
            reduce_sum(kp * q_row.broadcast(const_shape![32, 64]), 1i32);
        let latent_f16: Tile<f16, { [16, 32] }> =
            latent_fp16.load_tile(const_shape![16, 32], [physical, 0]);
        let latent_f32: Tile<f32, { [16, 32] }> = convert_tile(latent_f16);
        let dots: Tile<f32, { [16] }> = reduce_sum(
            latent_f32
                * projected
                    .reshape(const_shape![1, 32])
                    .broadcast(const_shape![16, 32]),
            1i32,
        );
        let scale: Tile<f32, { [16] }> = broadcast_scalar(0.125f32, const_shape![16]);
        let scores = dots * scale;
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        let masked = select(
            active_mask,
            scores,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
        );
        out.store(masked.reshape(const_shape![1, 16]));
    }

    #[cutile::entry()]
    pub fn model_small_softmax_1024_runtime(
        out: &mut Tensor<f32, { [1, 1024] }>,
        scores: &Tensor<f32, { [-1, 1024] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 1024] }> =
            scores.load_tile(const_shape![1, 1024], [q_head, 0]);
        let token_indices: Tile<i32, { [1024] }> = iota(const_shape![1024]);
        let active_mask_1d: Tile<bool, { [1024] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![1024]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 1024]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 1024]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 1024]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 1024]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 1024]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 1024] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [64] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..64i32 {
            latent_context = latent_context
                + latent_block_contribution(probabilities, latent_fp16, table, head, logical_block);
        }
        let vp: Tile<f32, { [32, 64] }> =
            v_projection.load_tile(const_shape![32, 64], [kv_head, 0]);
        let context: Tile<f32, { [64] }> = reduce_sum(
            latent_context
                .reshape(const_shape![32, 1])
                .broadcast(const_shape![32, 64])
                * vp,
            0i32,
        );
        out.store(context.reshape(const_shape![1, 64]));
    }
}
