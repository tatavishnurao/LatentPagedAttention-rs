#[cutile::module]
pub mod paged_latent_write_fp16_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn paged_latent_write_fp16(
        latent_cache: &mut Tensor<f16, { [2, 8] }>,
        block_table: &Tensor<i32, { [4] }>,
        token_position: &Tensor<i32, { [1] }>,
        new_latent_f32: &Tensor<f32, { [8] }>,
    ) {
        let physical_block_id = get_tile_block_id().0;
        let token_tile: Tile<i32, { [1] }> = token_position.load_tile(const_shape![1], [0]);
        let token_scalar: i32 = tile_to_scalar(token_tile.reshape(const_shape![]));
        let logical_block = token_scalar / 2i32;
        let block_offset = token_scalar % 2i32;
        let table_tile: Tile<i32, { [4] }> = block_table.load_tile(const_shape![4], [0]);
        let target_tile: Tile<i32, { [1] }> = extract(table_tile, [scalar_to_tile(logical_block)]);
        let target_physical_block: i32 = tile_to_scalar(target_tile.reshape(const_shape![]));

        let mut latent_tile: Tile<f16, { [2, 8] }> = load_tile_mut(latent_cache);
        if physical_block_id == target_physical_block {
            let row_indices: Tile<i32, { [2] }> = iota(const_shape![2]);
            let offset_tile: Tile<i32, { [2] }> = broadcast_scalar(block_offset, const_shape![2]);
            let row_mask: Tile<bool, { [2] }> = cmpi(row_indices, offset_tile, predicate::Equal);
            let row_mask_2d = row_mask
                .reshape(const_shape![2, 1])
                .broadcast(const_shape![2, 8]);
            let replacement_f32: Tile<f32, { [8] }> =
                new_latent_f32.load_tile(const_shape![8], [0]);
            let replacement_f16: Tile<f16, { [8] }> = convert_tile(replacement_f32);
            let replacement = replacement_f16
                .reshape(const_shape![1, 8])
                .broadcast(const_shape![2, 8]);
            latent_tile = select(row_mask_2d, replacement, latent_tile);
        }
        latent_cache.store(latent_tile);
    }
}
