#[cutile::module]
pub mod full_kv_baseline_kernel {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [64] }>, logical: i32) -> i32 {
        let table_tile: Tile<i32, { [64] }> = table.load_tile(const_shape![64], [0]);
        let selected: Tile<i32, { [1] }> = extract(table_tile, [scalar_to_tile(logical)]);
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
    pub fn model_small_full_kv_scores_fp16_storage(
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
    pub fn model_small_full_kv_context_fp16_storage(
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
