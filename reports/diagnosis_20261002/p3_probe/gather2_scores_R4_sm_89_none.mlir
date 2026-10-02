cuda_tile.module @gather_mma_module {
  entry @latent_gather2_scores_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f16>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>) {
    %22 = constant <i32: 4> : tile<i32>
    %23 = assume bounded<0, ?>, %1 : tile<i32>
    %24 = assume bounded<0, ?>, %2 : tile<i32>
    %25 = make_token : token
    %26 = make_tensor_view %0, shape = [%23, %24], strides = [16, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[16,1]>
    %27 = make_token : token
    %28 = make_tensor_view %9, shape = [32, 16], strides = [16, 1] : tensor_view<32x16xf16, strides=[16,1]>
    %29 = assume bounded<0, ?>, %15 : tile<i32>
    %30 = make_token : token
    %31 = make_tensor_view %14, shape = [%29, 32], strides = [32, 1] : tile<i32> -> tensor_view<?x32xf16, strides=[32,1]>
    %32 = assume bounded<0, ?>, %20 : tile<i32>
    %33 = make_token : token
    %34 = make_tensor_view %19, shape = [%32], strides = [1] : tile<i32> -> tensor_view<?xi32, strides=[1]>
    %35 = constant <i32: 4> : tile<i32>
    %36, %37, %38 = get_tile_block_id : tile<i32>
    %39 = assume bounded<0, ?>, %36 : tile<i32>
    %40 = assume bounded<0, ?>, %37 : tile<i32>
    %41 = assume bounded<0, ?>, %38 : tile<i32>
    %42 = constant <i32: 0> : tile<i32>
    %43 = constant <i32: 0> : tile<i32>
    %44 = constant <i32: 32> : tile<i32>
    %45 = constant <i32: 16> : tile<i32>
    %46 = constant <i32: 32> : tile<i32>
    %47 = constant <i32: 16> : tile<i32>
    %48 = constant <i32: 32> : tile<i32>
    %49 = constant <i32: 16> : tile<i32>
    %50 = constant <i32: 32> : tile<i32>
    %51 = constant <i32: 16> : tile<i32>
    %52 = constant <i32: 32> : tile<i32>
    %53 = constant <i32: 16> : tile<i32>
    %54 = make_partition_view %28 : partition_view<tile=(32x16), padding_value = zero, tensor_view<32x16xf16, strides=[16,1]>>
    %55, %56 = load_view_tko weak %54[%42, %43] token = %27 : partition_view<tile=(32x16), padding_value = zero, tensor_view<32x16xf16, strides=[16,1]>>, tile<i32> -> tile<32x16xf16>, token
    %57 = constant <f32: 0.0> : tile<32x16xf32>
    %58 = constant <i32: 0> : tile<i32>
    %59 = constant <i32: 1> : tile<i32>
    %120 = for %60 in (%58 to %35, step %59) : tile<i32> iter_values(%61 = %57) -> (tile<32x16xf32>) {
      %62 = assume bounded<0, 3>, %60 : tile<i32>
      %63 = muli %39, %35 : tile<i32>
      %64 = addi %63, %62 : tile<i32>
      %65 = constant <i32: 2> : tile<i32>
      %66 = muli %64, %65 : tile<i32>
      %67 = constant <i32: 1> : tile<i32>
      %68 = constant <i32: -1> : tile<i32>
      %69 = constant <i32: 1> : tile<i32>
      %70 = constant <i32: -1> : tile<i32>
      %71 = constant <i32: -1> : tile<i32>
      %72 = make_partition_view %34 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %73, %74 = load_view_tko weak %72[%66] token = %33 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %75 = constant <i32: 1> : tile<i32>
      %76 = addi %66, %75 : tile<i32>
      %77 = constant <i32: 1> : tile<i32>
      %78 = constant <i32: -1> : tile<i32>
      %79 = constant <i32: 1> : tile<i32>
      %80 = constant <i32: -1> : tile<i32>
      %81 = constant <i32: -1> : tile<i32>
      %82 = make_partition_view %34 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %83, %84 = load_view_tko weak %82[%76] token = %33 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %85 = constant <i32: 1> : tile<i32>
      %86 = reshape %73 : tile<1xi32> -> tile<i32>
      %87 = constant <i32: 1> : tile<i32>
      %88 = reshape %83 : tile<1xi32> -> tile<i32>
      %89 = constant <i32: 0> : tile<i32>
      %90 = constant <i32: 16> : tile<i32>
      %91 = constant <i32: 32> : tile<i32>
      %92 = constant <i32: -1> : tile<i32>
      %93 = constant <i32: 32> : tile<i32>
      %94 = constant <i32: 16> : tile<i32>
      %95 = constant <i32: 32> : tile<i32>
      %96 = constant <i32: -1> : tile<i32>
      %97 = constant <i32: 32> : tile<i32>
      %98 = constant <i32: -1> : tile<i32>
      %99 = constant <i32: 32> : tile<i32>
      %100 = make_partition_view %31 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %101, %102 = load_view_tko weak %100[%86, %89] token = %30 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %103 = constant <i32: 0> : tile<i32>
      %104 = constant <i32: 16> : tile<i32>
      %105 = constant <i32: 32> : tile<i32>
      %106 = constant <i32: -1> : tile<i32>
      %107 = constant <i32: 32> : tile<i32>
      %108 = constant <i32: 16> : tile<i32>
      %109 = constant <i32: 32> : tile<i32>
      %110 = constant <i32: -1> : tile<i32>
      %111 = constant <i32: 32> : tile<i32>
      %112 = constant <i32: -1> : tile<i32>
      %113 = constant <i32: 32> : tile<i32>
      %114 = make_partition_view %31 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %115, %116 = load_view_tko weak %114[%88, %103] token = %30 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %117 = constant <i32: 0> : tile<i32>
      %118 = cat %101, %115 dim = 0 : tile<16x32xf16>, tile<16x32xf16> -> tile<32x32xf16>
      %119 = mmaf %118, %55, %61 : tile<32x32xf16>, tile<32x16xf16>, tile<32x16xf32>
      continue %119 : tile<32x16xf32>
    }
    %121 = constant <i32: 32> : tile<i32>
    %122 = constant <i32: 16> : tile<i32>
    %123 = constant <i32: 32> : tile<i32>
    %124 = constant <i32: 16> : tile<i32>
    %125, %126, %127 = get_tile_block_id : tile<i32>
    %128 = assume bounded<0, ?>, %125 : tile<i32>
    %129 = assume bounded<0, ?>, %126 : tile<i32>
    %130 = assume bounded<0, ?>, %127 : tile<i32>
    %131 = make_partition_view %26 : partition_view<tile=(32x16), tensor_view<?x?xf32, strides=[16,1]>>
    %132 = store_view_tko weak %120, %131[%128, %129] token = %25 : tile<32x16xf32>, partition_view<tile=(32x16), tensor_view<?x?xf32, strides=[16,1]>>, tile<i32> -> token
    return
  }
}
