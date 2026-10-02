cuda_tile.module @gather_mma_module {
  entry @latent_gather2_scores_transposed_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f16>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>) {
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
    %131 = for %68 in (%66 to %43, step %67) : tile<i32> iter_values(%69 = %65) -> (tile<16x32xf32>) {
      %70 = assume bounded<0, 3>, %68 : tile<i32>
      %71 = muli %47, %43 : tile<i32>
      %72 = addi %71, %70 : tile<i32>
      %73 = constant <i32: 2> : tile<i32>
      %74 = muli %72, %73 : tile<i32>
      %75 = constant <i32: 1> : tile<i32>
      %76 = constant <i32: -1> : tile<i32>
      %77 = constant <i32: 1> : tile<i32>
      %78 = constant <i32: -1> : tile<i32>
      %79 = constant <i32: -1> : tile<i32>
      %80 = make_partition_view %42 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %81, %82 = load_view_tko weak %80[%74] token = %40 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %83 = constant <i32: 1> : tile<i32>
      %84 = addi %74, %83 : tile<i32>
      %85 = constant <i32: 1> : tile<i32>
      %86 = constant <i32: -1> : tile<i32>
      %87 = constant <i32: 1> : tile<i32>
      %88 = constant <i32: -1> : tile<i32>
      %89 = constant <i32: -1> : tile<i32>
      %90 = make_partition_view %42 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %91, %92 = load_view_tko weak %90[%84] token = %40 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %93 = constant <i32: 1> : tile<i32>
      %94 = reshape %81 : tile<1xi32> -> tile<i32>
      %95 = constant <i32: 1> : tile<i32>
      %96 = reshape %91 : tile<1xi32> -> tile<i32>
      %97 = constant <i32: 0> : tile<i32>
      %98 = constant <i32: 16> : tile<i32>
      %99 = constant <i32: 32> : tile<i32>
      %100 = constant <i32: -1> : tile<i32>
      %101 = constant <i32: 32> : tile<i32>
      %102 = constant <i32: 16> : tile<i32>
      %103 = constant <i32: 32> : tile<i32>
      %104 = constant <i32: -1> : tile<i32>
      %105 = constant <i32: 32> : tile<i32>
      %106 = constant <i32: -1> : tile<i32>
      %107 = constant <i32: 32> : tile<i32>
      %108 = make_partition_view %37 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %109, %110 = load_view_tko weak %108[%94, %97] token = %35 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %111 = constant <i32: 0> : tile<i32>
      %112 = constant <i32: 16> : tile<i32>
      %113 = constant <i32: 32> : tile<i32>
      %114 = constant <i32: -1> : tile<i32>
      %115 = constant <i32: 32> : tile<i32>
      %116 = constant <i32: 16> : tile<i32>
      %117 = constant <i32: 32> : tile<i32>
      %118 = constant <i32: -1> : tile<i32>
      %119 = constant <i32: 32> : tile<i32>
      %120 = constant <i32: -1> : tile<i32>
      %121 = constant <i32: 32> : tile<i32>
      %122 = make_partition_view %37 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %123, %124 = load_view_tko weak %122[%96, %111] token = %35 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %125 = constant <i32: 0> : tile<i32>
      %126 = cat %109, %123 dim = 0 : tile<16x32xf16>, tile<16x32xf16> -> tile<32x32xf16>
      %127 = constant <i32: 32> : tile<i32>
      %128 = constant <i32: 32> : tile<i32>
      %129 = permute %126 [1, 0] : tile<32x32xf16> -> tile<32x32xf16>
      %130 = mmaf %63, %129, %69 : tile<16x32xf16>, tile<32x32xf16>, tile<16x32xf32>
      continue %130 : tile<16x32xf32>
    }
    %132 = constant <i32: 16> : tile<i32>
    %133 = constant <i32: 32> : tile<i32>
    %134 = constant <i32: 16> : tile<i32>
    %135 = constant <i32: 32> : tile<i32>
    %136, %137, %138 = get_tile_block_id : tile<i32>
    %139 = assume bounded<0, ?>, %136 : tile<i32>
    %140 = assume bounded<0, ?>, %137 : tile<i32>
    %141 = assume bounded<0, ?>, %138 : tile<i32>
    %142 = make_partition_view %29 : partition_view<tile=(16x32), tensor_view<?x?xf32, strides=[32,1]>>
    %143 = store_view_tko weak %131, %142[%139, %140] token = %27 : tile<16x32xf32>, partition_view<tile=(16x32), tensor_view<?x?xf32, strides=[32,1]>>, tile<i32> -> token
    return
  }
}
