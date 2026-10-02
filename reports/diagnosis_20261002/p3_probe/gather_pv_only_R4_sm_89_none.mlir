cuda_tile.module @gather_mma_module {
  entry @latent_gather_pv_only_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f16>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>) {
    %22 = constant <i32: 4> : tile<i32>
    %23 = assume bounded<0, ?>, %1 : tile<i32>
    %24 = assume bounded<0, ?>, %2 : tile<i32>
    %25 = make_token : token
    %26 = make_tensor_view %0, shape = [%23, %24], strides = [32, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[32,1]>
    %27 = assume bounded<0, ?>, %11 : tile<i32>
    %28 = make_token : token
    %29 = make_tensor_view %9, shape = [16, %27], strides = [8192, 1] : tile<i32> -> tensor_view<16x?xf16, strides=[8192,1]>
    %30 = assume bounded<0, ?>, %15 : tile<i32>
    %31 = make_token : token
    %32 = make_tensor_view %14, shape = [%30, 32], strides = [32, 1] : tile<i32> -> tensor_view<?x32xf16, strides=[32,1]>
    %33 = assume bounded<0, ?>, %20 : tile<i32>
    %34 = make_token : token
    %35 = make_tensor_view %19, shape = [%33], strides = [1] : tile<i32> -> tensor_view<?xi32, strides=[1]>
    %36 = constant <i32: 4> : tile<i32>
    %37, %38, %39 = get_tile_block_id : tile<i32>
    %40 = assume bounded<0, ?>, %37 : tile<i32>
    %41 = assume bounded<0, ?>, %38 : tile<i32>
    %42 = assume bounded<0, ?>, %39 : tile<i32>
    %43 = constant <f32: 0.0> : tile<16x32xf32>
    %44 = constant <i32: 0> : tile<i32>
    %45 = constant <i32: 1> : tile<i32>
    %90 = for %46 in (%44 to %36, step %45) : tile<i32> iter_values(%47 = %43) -> (tile<16x32xf32>) {
      %48 = assume bounded<0, 3>, %46 : tile<i32>
      %49 = muli %40, %36 : tile<i32>
      %50 = addi %49, %48 : tile<i32>
      %51 = constant <i32: 1> : tile<i32>
      %52 = constant <i32: -1> : tile<i32>
      %53 = constant <i32: 1> : tile<i32>
      %54 = constant <i32: -1> : tile<i32>
      %55 = constant <i32: -1> : tile<i32>
      %56 = make_partition_view %35 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %57, %58 = load_view_tko weak %56[%50] token = %34 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %59 = constant <i32: 1> : tile<i32>
      %60 = reshape %57 : tile<1xi32> -> tile<i32>
      %61 = constant <i32: 0> : tile<i32>
      %62 = constant <i32: 16> : tile<i32>
      %63 = constant <i32: 32> : tile<i32>
      %64 = constant <i32: -1> : tile<i32>
      %65 = constant <i32: 32> : tile<i32>
      %66 = constant <i32: 16> : tile<i32>
      %67 = constant <i32: 32> : tile<i32>
      %68 = constant <i32: -1> : tile<i32>
      %69 = constant <i32: 32> : tile<i32>
      %70 = constant <i32: -1> : tile<i32>
      %71 = constant <i32: 32> : tile<i32>
      %72 = make_partition_view %32 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %73, %74 = load_view_tko weak %72[%60, %61] token = %31 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %75 = constant <i32: 0> : tile<i32>
      %76 = constant <i32: 16> : tile<i32>
      %77 = constant <i32: 16> : tile<i32>
      %78 = constant <i32: 16> : tile<i32>
      %79 = constant <i32: -1> : tile<i32>
      %80 = constant <i32: 16> : tile<i32>
      %81 = constant <i32: 16> : tile<i32>
      %82 = constant <i32: 16> : tile<i32>
      %83 = constant <i32: -1> : tile<i32>
      %84 = constant <i32: 16> : tile<i32>
      %85 = constant <i32: -1> : tile<i32>
      %86 = make_partition_view %29 : partition_view<tile=(16x16), padding_value = zero, tensor_view<16x?xf16, strides=[8192,1]>>
      %87, %88 = load_view_tko weak %86[%75, %50] token = %28 : partition_view<tile=(16x16), padding_value = zero, tensor_view<16x?xf16, strides=[8192,1]>>, tile<i32> -> tile<16x16xf16>, token
      %89 = mmaf %87, %73, %47 : tile<16x16xf16>, tile<16x32xf16>, tile<16x32xf32>
      continue %89 : tile<16x32xf32>
    }
    %91 = constant <i32: 16> : tile<i32>
    %92 = constant <i32: 32> : tile<i32>
    %93 = constant <i32: 16> : tile<i32>
    %94 = constant <i32: 32> : tile<i32>
    %95, %96, %97 = get_tile_block_id : tile<i32>
    %98 = assume bounded<0, ?>, %95 : tile<i32>
    %99 = assume bounded<0, ?>, %96 : tile<i32>
    %100 = assume bounded<0, ?>, %97 : tile<i32>
    %101 = make_partition_view %26 : partition_view<tile=(16x32), tensor_view<?x?xf32, strides=[32,1]>>
    %102 = store_view_tko weak %90, %101[%98, %99] token = %25 : tile<16x32xf32>, partition_view<tile=(16x32), tensor_view<?x?xf32, strides=[32,1]>>, tile<i32> -> token
    return
  }
}
