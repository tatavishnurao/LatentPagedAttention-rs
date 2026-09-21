#[cutile::module]
pub mod paged_kv_write_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn paged_kv_write(
        k_cache: &mut Tensor<f32, { [2, 8] }>,
        v_cache: &mut Tensor<f32, { [2, 8] }>,
        block_table: &Tensor<i32, { [-1] }>,
        token_position: &Tensor<i32, { [1] }>,
        new_k: &Tensor<f32, { [-1] }>,
        new_v: &Tensor<f32, { [-1] }>,
    ) {
        let physical_block_id = get_tile_block_id().0;
        let token_tile: Tile<i32, { [1] }> = token_position.load_tile(const_shape![1], [0]);
        let token_scalar: i32 = tile_to_scalar(token_tile.reshape(const_shape![]));
        let logical_block = token_scalar / 2i32;
        let block_offset = token_scalar % 2i32;
        let table_tile: Tile<i32, { [4] }> = block_table.load_tile(const_shape![4], [0]);
        let logical_tile: Tile<i32, { [] }> = scalar_to_tile(logical_block);
        let target_tile: Tile<i32, { [1] }> = extract(table_tile, [logical_tile]);
        let target_physical_block: i32 = tile_to_scalar(target_tile.reshape(const_shape![]));

        let mut k_tile: Tile<f32, { [2, 8] }> = load_tile_mut(k_cache);
        let mut v_tile: Tile<f32, { [2, 8] }> = load_tile_mut(v_cache);
        if physical_block_id == target_physical_block {
            let row_indices: Tile<i32, { [2] }> = iota(const_shape![2]);
            let offset_tile: Tile<i32, { [2] }> = broadcast_scalar(block_offset, const_shape![2]);
            let row_mask: Tile<bool, { [2] }> = cmpi(row_indices, offset_tile, predicate::Equal);
            let row_mask_2d: Tile<bool, { [2, 8] }> = row_mask
                .reshape(const_shape![2, 1])
                .broadcast(const_shape![2, 8]);
            let k_value: Tile<f32, { [8] }> = new_k.load_tile(const_shape![8], [0]);
            let v_value: Tile<f32, { [8] }> = new_v.load_tile(const_shape![8], [0]);
            let k_replacement = k_value
                .reshape(const_shape![1, 8])
                .broadcast(const_shape![2, 8]);
            let v_replacement = v_value
                .reshape(const_shape![1, 8])
                .broadcast(const_shape![2, 8]);
            k_tile = select(row_mask_2d, k_replacement, k_tile);
            v_tile = select(row_mask_2d, v_replacement, v_tile);
        }
        k_cache.store(k_tile);
        v_cache.store(v_tile);
    }

    #[cutile::entry()]
    pub fn paged_lookup_width8(
        output: &mut Tensor<f32, { [2, 8] }>,
        physical_cache: &Tensor<f32, { [-1, 8] }>,
        block_table: &Tensor<i32, { [-1] }>,
    ) {
        let logical_block_id = get_tile_block_id().0;
        let table_tile: Tile<i32, { [4] }> = block_table.load_tile(const_shape![4], [0]);
        let logical_tile: Tile<i32, { [] }> = scalar_to_tile(logical_block_id);
        let target_tile: Tile<i32, { [1] }> = extract(table_tile, [logical_tile]);
        let physical_block_id: i32 = tile_to_scalar(target_tile.reshape(const_shape![]));
        let cache_tile = physical_cache.load_tile(const_shape![2, 8], [physical_block_id, 0]);
        output.store(cache_tile);
    }
}
