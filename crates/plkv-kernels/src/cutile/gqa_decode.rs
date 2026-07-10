#[cutile::module]
pub mod gqa_decode_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn contiguous_gqa_decode(
        scores_out: &mut Tensor<f32, { [1, 8] }>,
        probabilities_out: &mut Tensor<f32, { [1, 8] }>,
        context_out: &mut Tensor<f32, { [1, 8] }>,
        q: &Tensor<f32, { [-1, 8] }>,
        k_head_major: &Tensor<f32, { [-1, 8] }>,
        v_head_major: &Tensor<f32, { [-1, 8] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let kv_head = q_head / 2i32;

        let q_row: Tile<f32, { [1, 8] }> = q.load_tile(const_shape![1, 8], [q_head, 0]);
        let k_start = kv_head * 8i32;
        let k_tile: Tile<f32, { [8, 8] }> =
            k_head_major.load_tile(const_shape![8, 8], [k_start, 0]);
        let q_broadcast: Tile<f32, { [8, 8] }> = q_row.broadcast(const_shape![8, 8]);
        let dot_products: Tile<f32, { [8] }> = reduce_sum(k_tile * q_broadcast, 1i32);
        let scores: Tile<f32, { [8] }> = dot_products * 0.3535533905932738f32;
        scores_out.store(scores.reshape(const_shape![1, 8]));

        let row_max: Tile<f32, { [] }> = reduce_max(scores, 0i32);
        let shifted = scores - row_max.reshape(const_shape![1]).broadcast(const_shape![8]);
        let numerators: Tile<f32, { [8] }> = exp(shifted);
        let denominator: Tile<f32, { [] }> = reduce_sum(numerators, 0i32);
        let probabilities =
            numerators / denominator.reshape(const_shape![1]).broadcast(const_shape![8]);
        probabilities_out.store(probabilities.reshape(const_shape![1, 8]));

        let v_start = kv_head * 8i32;
        let v_tile: Tile<f32, { [8, 8] }> =
            v_head_major.load_tile(const_shape![8, 8], [v_start, 0]);
        let probability_weights: Tile<f32, { [8, 8] }> =
            probabilities.reshape(const_shape![8, 1]).broadcast(const_shape![8, 8]);
        let context: Tile<f32, { [8] }> = reduce_sum(probability_weights * v_tile, 0i32);
        context_out.store(context.reshape(const_shape![1, 8]));
    }
}
