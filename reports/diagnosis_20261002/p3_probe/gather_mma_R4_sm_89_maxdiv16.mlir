cuda_tile.module @gather_mma_module {
  entry @latent_gather_mma_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f16>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>) {
    %22 = constant <i32: 4> : tile<i32>
    %23 = assume bounded<0, ?>, %1 : tile<i32>
    %24 = assume div_by<16>, %23 : tile<i32>
    %25 = assume bounded<0, ?>, %2 : tile<i32>
    %26 = assume div_by<16>, %25 : tile<i32>
    %27 = make_token : token
    %28 = assume div_by<16>, %0 : tile<ptr<f32>>
    %29 = make_tensor_view %28, shape = [%24, %26], strides = [32, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[32,1]>
    %30 = make_token : token
    %31 = assume div_by<16>, %9 : tile<ptr<f16>>
    %32 = make_tensor_view %31, shape = [16, 32], strides = [32, 1] : tensor_view<16x32xf16, strides=[32,1]>
    %33 = assume bounded<0, ?>, %15 : tile<i32>
    %34 = assume div_by<16>, %33 : tile<i32>
    %35 = make_token : token
    %36 = assume div_by<16>, %14 : tile<ptr<f16>>
    %37 = make_tensor_view %36, shape = [%34, 32], strides = [32, 1] : tile<i32> -> tensor_view<?x32xf16, strides=[32,1]>
    %38 = assume bounded<0, ?>, %20 : tile<i32>
    %39 = assume div_by<16>, %38 : tile<i32>
    %40 = make_token : token
    %41 = assume div_by<16>, %19 : tile<ptr<i32>>
    %42 = make_tensor_view %41, shape = [%39], strides = [1] : tile<i32> -> tensor_view<?xi32, strides=[1]>
    %43 = constant <i32: 4> : tile<i32>
    %44, %45, %46 = get_tile_block_id : tile<i32>
    %47 = assume bounded<0, ?>, %44 : tile<i32>
    %48 = assume bounded<0, ?>, %45 : tile<i32>
    %49 = assume bounded<0, ?>, %46 : tile<i32>
    %50 = constant <i32: 0> : tile<i32>
    %51 = constant <i32: 0> : tile<i32>
    %52 = constant <i32: 16> : tile<i32>
    %53 = constant <i32: 32> : tile<i32>
    %54 = constant <i32: 16> : tile<i32>
    %55 = constant <i32: 32> : tile<i32>
    %56 = constant <i32: 16> : tile<i32>
    %57 = constant <i32: 32> : tile<i32>
    %58 = constant <i32: 16> : tile<i32>
    %59 = constant <i32: 32> : tile<i32>
    %60 = constant <i32: 16> : tile<i32>
    %61 = constant <i32: 32> : tile<i32>
    %62 = make_partition_view %32 : partition_view<tile=(16x32), padding_value = zero, tensor_view<16x32xf16, strides=[32,1]>>
    %63, %64 = load_view_tko weak %62[%50, %51] token = %30 : partition_view<tile=(16x32), padding_value = zero, tensor_view<16x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
    %65 = constant <f32: 0.0> : tile<16x32xf32>
    %66 = constant <i32: 0> : tile<i32>
    %67 = constant <i32: 1> : tile<i32>
    %105 = for %68 in (%66 to %43, step %67) : tile<i32> iter_values(%69 = %65) -> (tile<16x32xf32>) {
      %70 = assume bounded<0, 3>, %68 : tile<i32>
      %71 = muli %47, %43 : tile<i32>
      %72 = addi %71, %70 : tile<i32>
      %73 = constant <i32: 1> : tile<i32>
      %74 = constant <i32: -1> : tile<i32>
      %75 = constant <i32: 1> : tile<i32>
      %76 = constant <i32: -1> : tile<i32>
      %77 = constant <i32: -1> : tile<i32>
      %78 = make_partition_view %42 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %79, %80 = load_view_tko weak %78[%72] token = %40 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %81 = constant <i32: 1> : tile<i32>
      %82 = reshape %79 : tile<1xi32> -> tile<i32>
      %83 = constant <i32: 0> : tile<i32>
      %84 = constant <i32: 16> : tile<i32>
      %85 = constant <i32: 32> : tile<i32>
      %86 = constant <i32: -1> : tile<i32>
      %87 = constant <i32: 32> : tile<i32>
      %88 = constant <i32: 16> : tile<i32>
      %89 = constant <i32: 32> : tile<i32>
      %90 = constant <i32: -1> : tile<i32>
      %91 = constant <i32: 32> : tile<i32>
      %92 = constant <i32: -1> : tile<i32>
      %93 = constant <i32: 32> : tile<i32>
      %94 = make_partition_view %37 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %95, %96 = load_view_tko weak %94[%82, %83] token = %35 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %97 = constant <i32: 16> : tile<i32>
      %98 = constant <i32: 32> : tile<i32>
      %99 = permute %95 [1, 0] : tile<16x32xf16> -> tile<32x16xf16>
      %100 = constant <f32: 0.0> : tile<16x16xf32>
      %101 = mmaf %63, %99, %100 : tile<16x32xf16>, tile<32x16xf16>, tile<16x16xf32>
      %102 = exp %101 : tile<16x16xf32>
      %103 = ftof %102 : tile<16x16xf32> -> tile<16x16xf16>
      %104 = mmaf %103, %95, %69 : tile<16x16xf16>, tile<16x32xf16>, tile<16x32xf32>
      continue %104 : tile<16x32xf32>
    }
    %106 = constant <i32: 16> : tile<i32>
    %107 = constant <i32: 32> : tile<i32>
    %108 = constant <i32: 16> : tile<i32>
    %109 = constant <i32: 32> : tile<i32>
    %110, %111, %112 = get_tile_block_id : tile<i32>
    %113 = assume bounded<0, ?>, %110 : tile<i32>
    %114 = assume bounded<0, ?>, %111 : tile<i32>
    %115 = assume bounded<0, ?>, %112 : tile<i32>
    %116 = make_partition_view %29 : partition_view<tile=(16x32), tensor_view<?x?xf32, strides=[32,1]>>
    %117 = store_view_tko weak %105, %116[%113, %114] token = %27 : tile<16x32xf32>, partition_view<tile=(16x32), tensor_view<?x?xf32, strides=[32,1]>>, tile<i32> -> token
    return
  }
}
