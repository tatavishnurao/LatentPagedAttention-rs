#[cutile::module]
pub mod direct_paged_latent_gqa_kernel {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [4] }>, logical: i32) -> i32 {
        let t: Tile<i32, { [4] }> = table.load_tile(const_shape![4], [0]);
        let selected: Tile<i32, { [1] }> = extract(t, [scalar_to_tile(logical)]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    #[cutile::entry()]
    pub fn direct_paged_latent_scores(
        out: &mut Tensor<f32, { [1, 2] }>,
        q: &Tensor<f32, { [-1, 8] }>,
        latent: &Tensor<f32, { [-1, 8] }>,
        table: &Tensor<i32, { [4] }>,
        k_projection: &Tensor<f32, { [-1, 8] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical = pid.1;
        let kv = q_head / 2i32;
        let physical = physical_block(table, logical);
        let q_row: Tile<f32, { [1, 8] }> = q.load_tile(const_shape![1, 8], [q_head, 0]);
        let kp: Tile<f32, { [8, 8] }> = k_projection.load_tile(const_shape![8, 8], [kv, 0]);
        let q_b: Tile<f32, { [8, 8] }> = q_row.broadcast(const_shape![8, 8]);
        let projected: Tile<f32, { [8] }> = reduce_sum(kp * q_b, 1i32);
        let block: Tile<f32, { [2, 8] }> = latent.load_tile(const_shape![2, 8], [physical, 0]);
        let p_b: Tile<f32, { [2, 8] }> = projected
            .reshape(const_shape![1, 8])
            .broadcast(const_shape![2, 8]);
        let scale: Tile<f32, { [2] }> = broadcast_scalar(0.3535533905932738f32, const_shape![2]);
        let dots: Tile<f32, { [2] }> = reduce_sum(block * p_b, 1i32);
        let scores: Tile<f32, { [2] }> = dots * scale;
        out.store(scores.reshape(const_shape![1, 2]));
    }

    #[cutile::entry()]
    pub fn direct_paged_latent_context(
        out: &mut Tensor<f32, { [1, 8] }>,
        probabilities: &Tensor<f32, { [-1, 8] }>,
        latent: &Tensor<f32, { [-1, 8] }>,
        table: &Tensor<i32, { [4] }>,
        v_projection: &Tensor<f32, { [-1, 8] }>,
    ) {
        let h = get_tile_block_id().0;
        let kv = h / 2i32;
        let p0: Tile<f32, { [1, 2] }> = probabilities.load_tile(const_shape![1, 2], [h, 0]);
        let p1: Tile<f32, { [1, 2] }> = probabilities.load_tile(const_shape![1, 2], [h, 1]);
        let p2: Tile<f32, { [1, 2] }> = probabilities.load_tile(const_shape![1, 2], [h, 2]);
        let p3: Tile<f32, { [1, 2] }> = probabilities.load_tile(const_shape![1, 2], [h, 3]);
        let b0 = physical_block(table, 0i32);
        let b1 = physical_block(table, 1i32);
        let b2 = physical_block(table, 2i32);
        let b3 = physical_block(table, 3i32);
        let l0: Tile<f32, { [2, 8] }> = latent.load_tile(const_shape![2, 8], [b0, 0]);
        let l1: Tile<f32, { [2, 8] }> = latent.load_tile(const_shape![2, 8], [b1, 0]);
        let l2: Tile<f32, { [2, 8] }> = latent.load_tile(const_shape![2, 8], [b2, 0]);
        let l3: Tile<f32, { [2, 8] }> = latent.load_tile(const_shape![2, 8], [b3, 0]);
        let c0: Tile<f32, { [8] }> = reduce_sum(
            p0.reshape(const_shape![2, 1]).broadcast(const_shape![2, 8]) * l0,
            0i32,
        );
        let c1: Tile<f32, { [8] }> = reduce_sum(
            p1.reshape(const_shape![2, 1]).broadcast(const_shape![2, 8]) * l1,
            0i32,
        );
        let c2: Tile<f32, { [8] }> = reduce_sum(
            p2.reshape(const_shape![2, 1]).broadcast(const_shape![2, 8]) * l2,
            0i32,
        );
        let c3: Tile<f32, { [8] }> = reduce_sum(
            p3.reshape(const_shape![2, 1]).broadcast(const_shape![2, 8]) * l3,
            0i32,
        );
        let kp: Tile<f32, { [8, 8] }> = v_projection.load_tile(const_shape![8, 8], [kv, 0]);
        let latent_context: Tile<f32, { [8] }> = c0 + c1 + c2 + c3;
        let context: Tile<f32, { [8] }> = reduce_sum(
            latent_context
                .reshape(const_shape![8, 1])
                .broadcast(const_shape![8, 8])
                * kp,
            0i32,
        );
        out.store(context.reshape(const_shape![1, 8]));
    }
}
