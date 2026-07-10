#[::cutile::module]
pub mod paged_lookup_kernel {
    use ::cutile::core::*;

    #[cutile::entry()]
    pub fn paged_lookup(
        output: &mut Tensor<f32, { [2, 4] }>,
        physical_blocks: &Tensor<f32, { [-1, 4] }>,
        block_table: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let table_tile: Tile<i32, { [4] }> = block_table.load_tile(const_shape![4], [0]);
        let pid_tile: Tile<i32, { [] }> = scalar_to_tile(pid);
        let selected: Tile<i32, { [1] }> = extract(table_tile, [pid_tile]);
        let selected_scalar: Tile<i32, { [] }> = selected.reshape(const_shape![]);
        let physical_block: i32 = tile_to_scalar(selected_scalar);
        let input_tile = physical_blocks.load_tile(const_shape![2, 4], [physical_block, 0]);
        output.store(input_tile);
    }
}
