#[cutile::module]
pub mod model_profile_preprojected_kernel {
    use cutile::core::*;

    fn physical_block(table: &Tensor<i32, { [64] }>, logical: i32) -> i32 {
        let table_tile: Tile<i32, { [64] }> = table.load_tile(const_shape![64], [0]);
        let selected: Tile<i32, { [1] }> = extract(table_tile, [scalar_to_tile(logical)]);
        tile_to_scalar(selected.reshape(const_shape![]))
    }

    #[cutile::entry()]
    pub fn model_small_project_query_once(
        out: &mut Tensor<f32, { [1, 32] }>,
        q: &Tensor<f32, { [-1, 64] }>,
        k_projection: &Tensor<f32, { [-1, 64] }>,
    ) {
        let q_head = get_tile_block_id().0;
        let kv_head = q_head / 4i32;
        let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
        let kp: Tile<f32, { [32, 64] }> =
            k_projection.load_tile(const_shape![32, 64], [kv_head, 0]);
        let projected: Tile<f32, { [32] }> =
            reduce_sum(kp * q_row.broadcast(const_shape![32, 64]), 1i32);
        out.store(projected.reshape(const_shape![1, 32]));
    }

    #[cutile::entry()]
    pub fn model_small_scores_fp16_storage_preprojected(
        out: &mut Tensor<f32, { [1, 16] }>,
        projected_query: &Tensor<f32, { [-1, 32] }>,
        latent_fp16: &Tensor<f16, { [-1, 32] }>,
        table: &Tensor<i32, { [64] }>,
        active_seq_len: &Tensor<i32, { [1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_head = pid.0;
        let logical_block = pid.1;
        let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
        let physical = physical_block(table, logical_block);
        let projected: Tile<f32, { [1, 32] }> =
            projected_query.load_tile(const_shape![1, 32], [q_head, 0]);
        let latent_f16: Tile<f16, { [16, 32] }> =
            latent_fp16.load_tile(const_shape![16, 32], [physical, 0]);
        let latent_f32: Tile<f32, { [16, 32] }> = convert_tile(latent_f16);
        let dots: Tile<f32, { [16] }> =
            reduce_sum(latent_f32 * projected.broadcast(const_shape![16, 32]), 1i32);
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
                broadcast_scalar(-3.402_823_5e38_f32, const_shape![16]),
            )
            .reshape(const_shape![1, 16]),
        );
    }
}
