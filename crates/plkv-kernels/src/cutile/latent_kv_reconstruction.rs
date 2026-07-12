#[cutile::module]
pub mod latent_kv_reconstruction_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn reconstruct_latent_kv(
        k_out: &mut Tensor<f32, { [1, 16] }>,
        v_out: &mut Tensor<f32, { [1, 16] }>,
        latent_cache: &Tensor<f32, { [-1, 8] }>,
        k_projection: &Tensor<f32, { [8, 16] }>,
        v_projection: &Tensor<f32, { [8, 16] }>,
    ) {
        let token = get_tile_block_id().0;
        let latent_row: Tile<f32, { [1, 8] }> =
            latent_cache.load_tile(const_shape![1, 8], [token, 0]);
        let k_projection_tile: Tile<f32, { [8, 16] }> =
            k_projection.load_tile(const_shape![8, 16], [0, 0]);
        let v_projection_tile: Tile<f32, { [8, 16] }> =
            v_projection.load_tile(const_shape![8, 16], [0, 0]);

        let latent_weights: Tile<f32, { [8, 16] }> = latent_row
            .reshape(const_shape![8, 1])
            .broadcast(const_shape![8, 16]);
        let k_flat: Tile<f32, { [16] }> = reduce_sum(latent_weights * k_projection_tile, 0i32);
        let v_flat: Tile<f32, { [16] }> = reduce_sum(latent_weights * v_projection_tile, 0i32);

        k_out.store(k_flat.reshape(const_shape![1, 16]));
        v_out.store(v_flat.reshape(const_shape![1, 16]));
    }
}
