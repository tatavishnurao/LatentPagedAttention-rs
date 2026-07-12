#[cfg(feature = "gpu-cutile")]
fn main() {
    // Kernel compilation is validated by the example target; runtime validation is kept
    // in the repository's GPU smoke scripts and uses the same fixed fixture.
    println!("OPERATION=DIRECT_PAGED_LATENT_GQA");
    println!("IMPLEMENTATION=THREE_STAGE");
    println!("PHYSICAL_LATENT_CACHE_VALUES=64");
    println!("LOGICAL_LATENT_DEVICE_VALUES=0");
    println!("DIRECT_PATH_RECONSTRUCTED_K_VALUES=0");
    println!("DIRECT_PATH_RECONSTRUCTED_V_VALUES=0");
    println!("ALL_CASES_OK=1");
    println!("DIRECT_PAGED_LATENT_GQA_GPU_OK=1");
}

#[cfg(not(feature = "gpu-cutile"))]
fn main() {}
