cuda_tile.module @p15b_full_kv_baseline_kernel_1024 {
  entry @model_small_full_kv_context_fp16_storage_rtable_1024_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>) {
    %22 = assume bounded<0, ?>, %1 : tile<i32>
    %23 = assume div_by<16>, %22 : tile<i32>
    %24 = assume bounded<0, ?>, %2 : tile<i32>
    %25 = assume div_by<16>, %24 : tile<i32>
    %26 = make_token : token
    %27 = assume div_by<16>, %0 : tile<ptr<f32>>
    %28 = make_tensor_view %27, shape = [%23, %25], strides = [64, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[64,1]>
    %29 = assume bounded<0, ?>, %10 : tile<i32>
    %30 = assume div_by<16>, %29 : tile<i32>
    %31 = make_token : token
    %32 = assume div_by<16>, %9 : tile<ptr<f32>>
    %33 = make_tensor_view %32, shape = [%30, 1024], strides = [1024, 1] : tile<i32> -> tensor_view<?x1024xf32, strides=[1024,1]>
    %34 = assume bounded<0, ?>, %15 : tile<i32>
    %35 = assume div_by<16>, %34 : tile<i32>
    %36 = make_token : token
    %37 = assume div_by<16>, %14 : tile<ptr<f16>>
    %38 = make_tensor_view %37, shape = [%35, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf16, strides=[64,1]>
    %39 = make_token : token
    %40 = assume div_by<16>, %19 : tile<ptr<i32>>
    %41 = make_tensor_view %40, shape = [64], strides = [1] : tensor_view<64xi32, strides=[1]>
    %42, %43, %44 = get_tile_block_id : tile<i32>
    %45 = assume bounded<0, ?>, %42 : tile<i32>
    %46 = assume bounded<0, ?>, %43 : tile<i32>
    %47 = assume bounded<0, ?>, %44 : tile<i32>
    %48 = constant <i32: 4> : tile<i32>
    %49 = divi %45, %48 signed rounding negative_inf : tile<i32>
    %50 = constant <f32: 0.0> : tile<f32>
    %51 = constant <i32: 64> : tile<i32>
    %52 = constant <i32: 1> : tile<i32>
    %53 = constant <i32: 1> : tile<i32>
    %54 = reshape %50 : tile<f32> -> tile<1xf32>
    %55 = constant <i32: 1> : tile<i32>
    %56 = constant <i32: 64> : tile<i32>
    %57 = broadcast %54 : tile<1xf32> -> tile<64xf32>
    %58 = constant <i32: 0> : tile<i32>
    %59 = constant <i32: 64> : tile<i32>
    %60 = constant <i32: 1> : tile<i32>
    %121 = for %61 in (%58 to %59, step %60) : tile<i32> iter_values(%62 = %57) -> (tile<64xf32>) {
      %63 = assume bounded<0, 63>, %61 : tile<i32>
      %64 = constant <i32: 1> : tile<i32>
      %65 = constant <i32: 16> : tile<i32>
      %66 = constant <i32: -1> : tile<i32>
      %67 = constant <i32: 1024> : tile<i32>
      %68 = constant <i32: 1> : tile<i32>
      %69 = constant <i32: 16> : tile<i32>
      %70 = constant <i32: -1> : tile<i32>
      %71 = constant <i32: 1024> : tile<i32>
      %72 = constant <i32: -1> : tile<i32>
      %73 = constant <i32: 1024> : tile<i32>
      %74 = make_partition_view %33 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x1024xf32, strides=[1024,1]>>
      %75, %76 = load_view_tko weak %74[%45, %63] token = %31 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x1024xf32, strides=[1024,1]>>, tile<i32> -> tile<1x16xf32>, token
      %77 = constant <i32: 1> : tile<i32>
      %78 = constant <i32: 64> : tile<i32>
      %79 = constant <i32: 1> : tile<i32>
      %80 = constant <i32: 64> : tile<i32>
      %81 = constant <i32: 64> : tile<i32>
      %82 = make_partition_view %41 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>
      %83, %84 = load_view_tko weak %82[%63] token = %39 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %85 = constant <i32: 1> : tile<i32>
      %86 = reshape %83 : tile<1xi32> -> tile<i32>
      %87 = constant <i32: 4> : tile<i32>
      %88 = muli %86, %87 : tile<i32>
      %89 = addi %88, %49 : tile<i32>
      %90 = constant <i32: 0> : tile<i32>
      %91 = constant <i32: 16> : tile<i32>
      %92 = constant <i32: 64> : tile<i32>
      %93 = constant <i32: -1> : tile<i32>
      %94 = constant <i32: 64> : tile<i32>
      %95 = constant <i32: 16> : tile<i32>
      %96 = constant <i32: 64> : tile<i32>
      %97 = constant <i32: -1> : tile<i32>
      %98 = constant <i32: 64> : tile<i32>
      %99 = constant <i32: -1> : tile<i32>
      %100 = constant <i32: 64> : tile<i32>
      %101 = make_partition_view %38 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>
      %102, %103 = load_view_tko weak %101[%89, %90] token = %36 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>, tile<i32> -> tile<16x64xf16>, token
      %104 = ftof %102 : tile<16x64xf16> -> tile<16x64xf32>
      %105 = constant <i32: 1> : tile<i32>
      %106 = constant <i32: 16> : tile<i32>
      %107 = constant <i32: 16> : tile<i32>
      %108 = constant <i32: 1> : tile<i32>
      %109 = reshape %75 : tile<1x16xf32> -> tile<16x1xf32>
      %110 = constant <i32: 16> : tile<i32>
      %111 = constant <i32: 1> : tile<i32>
      %112 = constant <i32: 16> : tile<i32>
      %113 = constant <i32: 64> : tile<i32>
      %114 = broadcast %109 : tile<16x1xf32> -> tile<16x64xf32>
      %115 = mulf %114, %104 : tile<16x64xf32>
      %119 = reduce %115 dim=0 identities=[0] : tile<16x64xf32> -> tile<64xf32> {
      ^bb0(%116: tile<f32>, %117: tile<f32>):
        %118 = addf %116, %117 : tile<f32>
        yield %118 : tile<f32>
      }
      %120 = addf %62, %119 : tile<64xf32>
      continue %120 : tile<64xf32>
    }
    %122 = constant <i32: 64> : tile<i32>
    %123 = constant <i32: 1> : tile<i32>
    %124 = constant <i32: 64> : tile<i32>
    %125 = reshape %121 : tile<64xf32> -> tile<1x64xf32>
    %126 = constant <i32: 1> : tile<i32>
    %127 = constant <i32: 64> : tile<i32>
    %128 = constant <i32: 1> : tile<i32>
    %129 = constant <i32: 64> : tile<i32>
    %130, %131, %132 = get_tile_block_id : tile<i32>
    %133 = assume bounded<0, ?>, %130 : tile<i32>
    %134 = assume bounded<0, ?>, %131 : tile<i32>
    %135 = assume bounded<0, ?>, %132 : tile<i32>
    %136 = make_partition_view %28 : partition_view<tile=(1x64), tensor_view<?x?xf32, strides=[64,1]>>
    %137 = store_view_tko weak %125, %136[%133, %134] token = %26 : tile<1x64xf32>, partition_view<tile=(1x64), tensor_view<?x?xf32, strides=[64,1]>>, tile<i32> -> token
    return
  }
}
