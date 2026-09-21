#[cutile::module]
pub mod direct_latent_gqa_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn direct_latent_gqa_decode(
        scores_out: &mut Tensor<f32, { [1, 8] }>,
        probabilities_out: &mut Tensor<f32, { [1, 8] }>,
        context_out: &mut Tensor<f32, { [1, 8] }>,
        q: &Tensor<f32, { [-1, 8] }>,
        latent_cache: &Tensor<f32, { [8, 8] }>,
        k_projection_head_major: &Tensor<f32, { [-1, 8] }>,
        v_projection_head_major: &Tensor<f32, { [-1, 8] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let kv_head = q_head / 2i32;

        let q_row: Tile<f32, { [1, 8] }> = q.load_tile(const_shape![1, 8], [q_head, 0]);
        let k_projection_tile: Tile<f32, { [8, 8] }> =
            k_projection_head_major.load_tile(const_shape![8, 8], [kv_head, 0]);
        let v_projection_tile: Tile<f32, { [8, 8] }> =
            v_projection_head_major.load_tile(const_shape![8, 8], [kv_head, 0]);

        let q_broadcast: Tile<f32, { [8, 8] }> = q_row.broadcast(const_shape![8, 8]);
        let projected_query_latent: Tile<f32, { [8] }> =
            reduce_sum(k_projection_tile * q_broadcast, 1i32);

        let projected_query_broadcast: Tile<f32, { [8, 8] }> = projected_query_latent
            .reshape(const_shape![1, 8])
            .broadcast(const_shape![8, 8]);
        let latent_cache_tile: Tile<f32, { [8, 8] }> =
            latent_cache.load_tile(const_shape![8, 8], [0, 0]);
        let raw_scores: Tile<f32, { [8] }> =
            reduce_sum(latent_cache_tile * projected_query_broadcast, 1i32);
        let scale: Tile<f32, { [8] }> = broadcast_scalar(0.3535533905932738f32, const_shape![8]);
        let scores: Tile<f32, { [8] }> = raw_scores * scale;
        scores_out.store(scores.reshape(const_shape![1, 8]));

        let row_max: Tile<f32, { [] }> = reduce_max(scores, 0i32);
        let shifted = scores - row_max.reshape(const_shape![1]).broadcast(const_shape![8]);
        let numerators: Tile<f32, { [8] }> = exp(shifted);
        let denominator: Tile<f32, { [] }> = reduce_sum(numerators, 0i32);
        let probabilities = numerators
            / denominator
                .reshape(const_shape![1])
                .broadcast(const_shape![8]);
        probabilities_out.store(probabilities.reshape(const_shape![1, 8]));

        let latent_context_weights: Tile<f32, { [8, 8] }> = probabilities
            .reshape(const_shape![8, 1])
            .broadcast(const_shape![8, 8]);
        let latent_context: Tile<f32, { [8] }> =
            reduce_sum(latent_context_weights * latent_cache_tile, 0i32);
        let latent_context_broadcast: Tile<f32, { [8, 8] }> = latent_context
            .reshape(const_shape![8, 1])
            .broadcast(const_shape![8, 8]);
        let context: Tile<f32, { [8] }> =
            reduce_sum(latent_context_broadcast * v_projection_tile, 0i32);
        context_out.store(context.reshape(const_shape![1, 8]));
    }
}
