// GENERATED R-TABLE kernels; A0/B0 sources remain unchanged.

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_1024 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [64] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 1024] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [64] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_1024(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [64] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_1024(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 1024] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..64i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_1024 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [64] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
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
    pub fn model_small_scores_fp16_storage_rtable_1024(
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
    pub fn model_small_context_fp16_storage_rtable_1024(
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_2048 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [128] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 2048] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [128] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_2048(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [128] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_2048(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 2048] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [128] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..128i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_2048 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [128] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 2048] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [128] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_2048(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [128] }>,
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
    pub fn model_small_softmax_2048_runtime(
        out: &mut Tensor<f32, { [1, 2048] }>,
        scores: &Tensor<f32, { [-1, 2048] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 2048] }> =
            scores.load_tile(const_shape![1, 2048], [q_head, 0]);
        let token_indices: Tile<i32, { [2048] }> = iota(const_shape![2048]);
        let active_mask_1d: Tile<bool, { [2048] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![2048]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 2048]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 2048]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 2048]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 2048]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 2048]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_2048(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 2048] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [128] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..128i32 {
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_4096 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [256] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 4096] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [256] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_4096(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [256] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_4096(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 4096] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [256] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..256i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_4096 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [256] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 4096] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [256] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_4096(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [256] }>,
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
    pub fn model_small_softmax_4096_runtime(
        out: &mut Tensor<f32, { [1, 4096] }>,
        scores: &Tensor<f32, { [-1, 4096] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 4096] }> =
            scores.load_tile(const_shape![1, 4096], [q_head, 0]);
        let token_indices: Tile<i32, { [4096] }> = iota(const_shape![4096]);
        let active_mask_1d: Tile<bool, { [4096] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![4096]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 4096]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 4096]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 4096]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 4096]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 4096]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_4096(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 4096] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [256] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..256i32 {
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_8192 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [512] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 8192] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [512] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_8192(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [512] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_8192(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 8192] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [512] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..512i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_8192 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [512] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 8192] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [512] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_8192(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [512] }>,
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
    pub fn model_small_softmax_8192_runtime(
        out: &mut Tensor<f32, { [1, 8192] }>,
        scores: &Tensor<f32, { [-1, 8192] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 8192] }> =
            scores.load_tile(const_shape![1, 8192], [q_head, 0]);
        let token_indices: Tile<i32, { [8192] }> = iota(const_shape![8192]);
        let active_mask_1d: Tile<bool, { [8192] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![8192]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 8192]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 8192]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 8192]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 8192]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 8192]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_8192(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 8192] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [512] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..512i32 {
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_16384 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [1024] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 16384] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [1024] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_16384(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [1024] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_16384(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 16384] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [1024] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..1024i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_16384 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [1024] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 16384] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [1024] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_16384(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [1024] }>,
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
    pub fn model_small_softmax_16384_runtime(
        out: &mut Tensor<f32, { [1, 16384] }>,
        scores: &Tensor<f32, { [-1, 16384] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 16384] }> =
            scores.load_tile(const_shape![1, 16384], [q_head, 0]);
        let token_indices: Tile<i32, { [16384] }> = iota(const_shape![16384]);
        let active_mask_1d: Tile<bool, { [16384] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16384]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 16384]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 16384]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 16384]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 16384]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 16384]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_16384(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 16384] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [1024] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..1024i32 {
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_32768 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [2048] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 32768] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [2048] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_32768(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [2048] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_32768(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 32768] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [2048] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..2048i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_32768 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [2048] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 32768] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [2048] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_32768(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [2048] }>,
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
    pub fn model_small_softmax_32768_runtime(
        out: &mut Tensor<f32, { [1, 32768] }>,
        scores: &Tensor<f32, { [-1, 32768] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 32768] }> =
            scores.load_tile(const_shape![1, 32768], [q_head, 0]);
        let token_indices: Tile<i32, { [32768] }> = iota(const_shape![32768]);
        let active_mask_1d: Tile<bool, { [32768] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![32768]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 32768]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 32768]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 32768]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 32768]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 32768]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_32768(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 32768] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [2048] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..2048i32 {
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_49152 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [3072] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 49152] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [3072] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_49152(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [3072] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_49152(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 49152] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [3072] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..3072i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_49152 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [3072] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 49152] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [3072] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_49152(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [3072] }>,
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
    pub fn model_small_softmax_49152_runtime(
        out: &mut Tensor<f32, { [1, 49152] }>,
        scores: &Tensor<f32, { [-1, 49152] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 49152] }> =
            scores.load_tile(const_shape![1, 49152], [q_head, 0]);
        let token_indices: Tile<i32, { [49152] }> = iota(const_shape![49152]);
        let active_mask_1d: Tile<bool, { [49152] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![49152]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 49152]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 49152]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 49152]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 49152]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 49152]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_49152(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 49152] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [3072] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..3072i32 {
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_65536 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [4096] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 65536] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [4096] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_65536(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [4096] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_65536(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 65536] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [4096] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..4096i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_65536 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [4096] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 65536] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [4096] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_65536(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [4096] }>,
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
    pub fn model_small_softmax_65536_runtime(
        out: &mut Tensor<f32, { [1, 65536] }>,
        scores: &Tensor<f32, { [-1, 65536] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 65536] }> =
            scores.load_tile(const_shape![1, 65536], [q_head, 0]);
        let token_indices: Tile<i32, { [65536] }> = iota(const_shape![65536]);
        let active_mask_1d: Tile<bool, { [65536] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![65536]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 65536]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 65536]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 65536]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 65536]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 65536]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_65536(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 65536] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [4096] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..4096i32 {
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_98304 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [6144] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 98304] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [6144] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_98304(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [6144] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_98304(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 98304] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [6144] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..6144i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_98304 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [6144] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 98304] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [6144] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_98304(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [6144] }>,
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
    pub fn model_small_softmax_98304_runtime(
        out: &mut Tensor<f32, { [1, 98304] }>,
        scores: &Tensor<f32, { [-1, 98304] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 98304] }> =
            scores.load_tile(const_shape![1, 98304], [q_head, 0]);
        let token_indices: Tile<i32, { [98304] }> = iota(const_shape![98304]);
        let active_mask_1d: Tile<bool, { [98304] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![98304]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 98304]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 98304]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 98304]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 98304]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 98304]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_98304(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 98304] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [6144] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..6144i32 {
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

#[cutile::module]
pub mod p15b_full_kv_baseline_kernel_131072 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [8192] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn value_block_contribution(
        probabilities: &Tensor<f32, { [-1, 131072] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [8192] }>,
        head: i32,
        kv_head: i32,
        logical_block: i32,
    ) -> Tile<f32, { [64] }> {
        let probs: Tile<f32, { [1, 16] }> =
            probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let values: Tile<f32, { [16, 64] }> =
            convert_tile(v_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        reduce_sum(
            probs
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 64])
                * values,
            0i32,
        )
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_scores_fp16_storage_rtable_131072(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [8192] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 4i32;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let physical_tile = physical * 4i32 + kv_head;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let k_tile: Tile<f32, { [16, 64] }> =
            convert_tile(k_fp16.load_tile(const_shape![16, 64], [physical_tile, 0]));
        let dots: Tile<f32, { [16] }> =
            reduce_sum(k_tile * q_row.broadcast(const_shape![16, 64]), 1i32);
        let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
        let token_indices: Tile<i32, { [16] }> =
            iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
        let active_mask: Tile<bool, { [16] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![16]),
            predicate::LessThan,
        );
        out.store(
            select(
                active_mask,
                scores,
                broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_full_kv_context_fp16_storage_rtable_131072(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 131072] }>,
        v_fp16: &Tensor<f16, { [-1, 64] }>,
        table: &Tensor<i32, { [8192] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut context: Tile<f32, { [64] }> = broadcast_scalar(0.0f32, const_shape![64]);
        for logical_block in 0i32..8192i32 {
            context = context
                + value_block_contribution(
                    probabilities,
                    v_fp16,
                    table,
                    head,
                    kv_head,
                    logical_block,
                );
        }
        out.store(context.reshape(const_shape![1, 64]));
    }
}

#[cutile::module]
pub mod p15b_model_profile_kernel_131072 {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [8192] }>, logical: i32) -> i32 {
        let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    fn latent_block_contribution(
        probabilities: &Tensor<f32, { [-1, 131072] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [8192] }>,
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
    pub fn model_small_scores_fp16_storage_rtable_131072(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [8192] }>,
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
    pub fn model_small_softmax_131072_runtime(
        out: &mut Tensor<f32, { [1, 131072] }>,
        scores: &Tensor<f32, { [-1, 131072] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let score_row: Tile<f32, { [1, 131072] }> =
            scores.load_tile(const_shape![1, 131072], [q_head, 0]);
        let token_indices: Tile<i32, { [131072] }> = iota(const_shape![131072]);
        let active_mask_1d: Tile<bool, { [131072] }> = cmpi(
            token_indices,
            active_tile.broadcast(const_shape![131072]),
            predicate::LessThan,
        );
        let active_mask = active_mask_1d.reshape(const_shape![1, 131072]);
        let masked_scores = select(
            active_mask,
            score_row,
            broadcast_scalar(-3.4028234663852886e38f32, const_shape![1, 131072]),
        );
        let row_max: Tile<f32, { [1] }> = reduce_max(masked_scores, 1i32);
        let shifted = masked_scores
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 131072]);
        let numerators = select(
            active_mask,
            exp(shifted),
            broadcast_scalar(0.0f32, const_shape![1, 131072]),
        );
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        out.store(
            numerators
                / denominator
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 131072]),
        );
    }

    #[cutile::entry()]
    pub fn model_small_context_fp16_storage_rtable_131072(
        out: &mut Tensor<f32, { [1, 64] }>,
        probabilities: &Tensor<f32, { [-1, 131072] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [8192] }>,
        v_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 4i32;
        let mut latent_context: Tile<f32, { [32] }> = broadcast_scalar(0.0f32, const_shape![32]);
        for logical_block in 0i32..8192i32 {
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
