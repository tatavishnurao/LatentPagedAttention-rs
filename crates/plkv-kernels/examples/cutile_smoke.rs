use std::sync::Arc;

use cutile::api;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::{DeviceOp, TileKernel};

#[cutile::module]
mod smoke_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn vector_add<const N: i32>(
        output: &mut Tensor<f32, { [N] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, output);
        let tile_y = load_tile_like(y, output);
        output.store(tile_x + tile_y);
    }
}

fn main() {
    const N: usize = 128;
    const TILE: usize = 128;
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; N]))
        .sync()
        .expect("failed to upload x");
    let y = api::copy_host_vec_to_device(&Arc::new(vec![2.0f32; N]))
        .sync()
        .expect("failed to upload y");
    let output = api::zeros::<f32>(&[N])
        .sync()
        .expect("failed to allocate output");

    let (result, _, _) = smoke_kernels::vector_add(output.partition([TILE]), &x, &y)
        .generics(vec![N.to_string()])
        .sync()
        .expect("cuTile vector add failed");

    let host = result
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("failed to read output");
    let max_abs_error = host
        .iter()
        .map(|value| (value - 3.0f32).abs())
        .fold(0.0f32, f32::max);
    if max_abs_error > 0.0 {
        eprintln!("cuTile smoke mismatch: max_abs_error={max_abs_error}");
        std::process::exit(1);
    }
    println!("VECTOR_LENGTH={N}");
    println!("TILE_SIZE={TILE}");
    println!("MAX_ABS_ERROR={max_abs_error}");
    println!("CUTILE_SMOKE_OK=1");
}
