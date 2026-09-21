#[cutile::module]
pub mod paged_gqa_decode_kernel {
    use cutile::core::*;

    fn physical_block_for_logical_block(
        block_table: &Tensor<i32, { [4] }>,
        logical_block: i32,
    ) -> i32 {
        let table_tile: Tile<i32, { [4] }> = block_table.load_tile(const_shape![4], [0]);
        let logical_tile: Tile<i32, { [] }> = scalar_to_tile(logical_block);
        let selected: Tile<i32, { [1] }> = extract(table_tile, [logical_tile]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    #[cutile::entry()]
    pub fn paged_gqa_scores(
        scores_out: &mut Tensor<f32, { [1, 2] }>,
        q: &Tensor<f32, { [-1, 8] }>,
        k_physical: &Tensor<f32, { [-1, 8] }>,
        block_table: &Tensor<i32, { [4] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 2i32;
        let physical_block = physical_block_for_logical_block(block_table, logical_block);
        let physical_tile = physical_block * 2i32 + kv_head;

        let q_row: Tile<f32, { [1, 8] }> = q.load_tile(const_shape![1, 8], [q_head, 0]);
        let k_tile: Tile<f32, { [2, 8] }> =
            k_physical.load_tile(const_shape![2, 8], [physical_tile, 0]);
        let q_broadcast: Tile<f32, { [2, 8] }> = q_row.broadcast(const_shape![2, 8]);
        let dots: Tile<f32, { [2] }> = reduce_sum(k_tile * q_broadcast, 1i32);
        let scale: Tile<f32, { [2] }> = broadcast_scalar(0.3535533905932738f32, const_shape![2]);
        let scores: Tile<f32, { [2] }> = dots * scale;
        scores_out.store(scores.reshape(const_shape![1, 2]));
    }

    #[cutile::entry()]
    pub fn stable_softmax_8(
        probabilities_out: &mut Tensor<f32, { [1, 8] }>,
        scores: &Tensor<f32, { [-1, 8] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let score_row: Tile<f32, { [1, 8] }> = scores.load_tile(const_shape![1, 8], [q_head, 0]);
        let row_max: Tile<f32, { [1] }> = reduce_max(score_row, 1i32);
        let shifted = score_row
            - row_max
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 8]);
        let numerators: Tile<f32, { [1, 8] }> = exp(shifted);
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerators, 1i32);
        let probabilities = numerators
            / denominator
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 8]);
        probabilities_out.store(probabilities);
    }

    #[cutile::entry()]
    pub fn paged_gqa_context(
        context_out: &mut Tensor<f32, { [1, 8] }>,
        probabilities: &Tensor<f32, { [-1, 8] }>,
        v_physical: &Tensor<f32, { [-1, 8] }>,
        block_table: &Tensor<i32, { [4] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let kv_head = q_head / 2i32;

        let physical_block_0 = physical_block_for_logical_block(block_table, 0i32);
        let physical_block_1 = physical_block_for_logical_block(block_table, 1i32);
        let physical_block_2 = physical_block_for_logical_block(block_table, 2i32);
        let physical_block_3 = physical_block_for_logical_block(block_table, 3i32);

        let v_tile_0: Tile<f32, { [2, 8] }> =
            v_physical.load_tile(const_shape![2, 8], [physical_block_0 * 2i32 + kv_head, 0]);
        let v_tile_1: Tile<f32, { [2, 8] }> =
            v_physical.load_tile(const_shape![2, 8], [physical_block_1 * 2i32 + kv_head, 0]);
        let v_tile_2: Tile<f32, { [2, 8] }> =
            v_physical.load_tile(const_shape![2, 8], [physical_block_2 * 2i32 + kv_head, 0]);
        let v_tile_3: Tile<f32, { [2, 8] }> =
            v_physical.load_tile(const_shape![2, 8], [physical_block_3 * 2i32 + kv_head, 0]);

        let p_tile_0: Tile<f32, { [1, 2] }> =
            probabilities.load_tile(const_shape![1, 2], [q_head, 0]);
        let p_tile_1: Tile<f32, { [1, 2] }> =
            probabilities.load_tile(const_shape![1, 2], [q_head, 1]);
        let p_tile_2: Tile<f32, { [1, 2] }> =
            probabilities.load_tile(const_shape![1, 2], [q_head, 2]);
        let p_tile_3: Tile<f32, { [1, 2] }> =
            probabilities.load_tile(const_shape![1, 2], [q_head, 3]);

        let p_broadcast_0: Tile<f32, { [2, 8] }> = p_tile_0
            .reshape(const_shape![2, 1])
            .broadcast(const_shape![2, 8]);
        let p_broadcast_1: Tile<f32, { [2, 8] }> = p_tile_1
            .reshape(const_shape![2, 1])
            .broadcast(const_shape![2, 8]);
        let p_broadcast_2: Tile<f32, { [2, 8] }> = p_tile_2
            .reshape(const_shape![2, 1])
            .broadcast(const_shape![2, 8]);
        let p_broadcast_3: Tile<f32, { [2, 8] }> = p_tile_3
            .reshape(const_shape![2, 1])
            .broadcast(const_shape![2, 8]);

        let block_context_0: Tile<f32, { [8] }> = reduce_sum(p_broadcast_0 * v_tile_0, 0i32);
        let block_context_1: Tile<f32, { [8] }> = reduce_sum(p_broadcast_1 * v_tile_1, 0i32);
        let block_context_2: Tile<f32, { [8] }> = reduce_sum(p_broadcast_2 * v_tile_2, 0i32);
        let block_context_3: Tile<f32, { [8] }> = reduce_sum(p_broadcast_3 * v_tile_3, 0i32);

        let context: Tile<f32, { [8] }> =
            block_context_0 + block_context_1 + block_context_2 + block_context_3;
        context_out.store(context.reshape(const_shape![1, 8]));
    }
}
