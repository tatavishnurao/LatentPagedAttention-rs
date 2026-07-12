#[cutile::module]
pub mod direct_paged_latent_gqa_fp16_kernel {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [4] }>, logical: i32) -> i32 {
        let table_tile: Tile<i32, { [4] }> = table.load_tile(const_shape![4], [0]);
        let selected: Tile<i32, { [1] }> = extract(table_tile, [scalar_to_tile(logical)]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    #[cutile::entry()]
    pub fn direct_paged_latent_scores_fp16_storage(
        out: &mut Tensor<f32, { [1, 2] }>,
        q: &Tensor<f32, { [-1, 8] }>,
        latent_fp16: &Tensor<f16, { [-1, 8] }>,
        table: &Tensor<i32, { [4] }>,
        k_projection: &Tensor<f32, { [-1, 8] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let kv_head = q_head / 2i32;
        let physical = physical_block(table, logical_block);
        let q_row: Tile<f32, { [1, 8] }> = q.load_tile(const_shape![1, 8], [q_head, 0]);
        let kp: Tile<f32, { [8, 8] }> = k_projection.load_tile(const_shape![8, 8], [kv_head, 0]);
        let projected: Tile<f32, { [8] }> =
            reduce_sum(kp * q_row.broadcast(const_shape![8, 8]), 1i32);
        let latent_f16: Tile<f16, { [2, 8] }> =
            latent_fp16.load_tile(const_shape![2, 8], [physical, 0]);
        let latent_f32: Tile<f32, { [2, 8] }> = convert_tile(latent_f16);
        let projected_broadcast = projected
            .reshape(const_shape![1, 8])
            .broadcast(const_shape![2, 8]);
        let dots: Tile<f32, { [2] }> = reduce_sum(latent_f32 * projected_broadcast, 1i32);
        let scale: Tile<f32, { [2] }> = broadcast_scalar(0.3535533905932738f32, const_shape![2]);
        out.store((dots * scale).reshape(const_shape![1, 2]));
    }

    #[cutile::entry()]
    pub fn direct_paged_latent_context_fp16_storage(
        out: &mut Tensor<f32, { [1, 8] }>,
        probabilities: &Tensor<f32, { [-1, 8] }>,
        latent_fp16: &Tensor<f16, { [-1, 8] }>,
        table: &Tensor<i32, { [4] }>,
        v_projection: &Tensor<f32, { [-1, 8] }>,
    ) {
        let head = get_tile_block_id().0;
        let kv_head = head / 2i32;
        let p0: Tile<f32, { [1, 2] }> = probabilities.load_tile(const_shape![1, 2], [head, 0]);
        let p1: Tile<f32, { [1, 2] }> = probabilities.load_tile(const_shape![1, 2], [head, 1]);
        let p2: Tile<f32, { [1, 2] }> = probabilities.load_tile(const_shape![1, 2], [head, 2]);
        let p3: Tile<f32, { [1, 2] }> = probabilities.load_tile(const_shape![1, 2], [head, 3]);
        let b0 = physical_block(table, 0i32);
        let b1 = physical_block(table, 1i32);
        let b2 = physical_block(table, 2i32);
        let b3 = physical_block(table, 3i32);
        let l0: Tile<f32, { [2, 8] }> =
            convert_tile(latent_fp16.load_tile(const_shape![2, 8], [b0, 0]));
        let l1: Tile<f32, { [2, 8] }> =
            convert_tile(latent_fp16.load_tile(const_shape![2, 8], [b1, 0]));
        let l2: Tile<f32, { [2, 8] }> =
            convert_tile(latent_fp16.load_tile(const_shape![2, 8], [b2, 0]));
        let l3: Tile<f32, { [2, 8] }> =
            convert_tile(latent_fp16.load_tile(const_shape![2, 8], [b3, 0]));
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
        let latent_context: Tile<f32, { [8] }> = c0 + c1 + c2 + c3;
        let vp: Tile<f32, { [8, 8] }> = v_projection.load_tile(const_shape![8, 8], [kv_head, 0]);
        let context: Tile<f32, { [8] }> = reduce_sum(
            latent_context
                .reshape(const_shape![8, 1])
                .broadcast(const_shape![8, 8])
                * vp,
            0i32,
        );
        out.store(context.reshape(const_shape![1, 8]));
    }
}
