cuda_tile.module @p15b_full_kv_baseline_kernel_1024 {
  entry @model_small_full_kv_context_fp16_storage_rtable_1024_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>) {
    %22 = assume bounded<0, ?>, %1 : tile<i32>
    %23 = assume bounded<0, ?>, %2 : tile<i32>
    %24 = make_token : token
    %25 = make_tensor_view %0, shape = [%22, %23], strides = [64, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[64,1]>
    %26 = assume bounded<0, ?>, %10 : tile<i32>
    %27 = make_token : token
    %28 = make_tensor_view %9, shape = [%26, 1024], strides = [1024, 1] : tile<i32> -> tensor_view<?x1024xf32, strides=[1024,1]>
    %29 = assume bounded<0, ?>, %15 : tile<i32>
    %30 = make_token : token
    %31 = make_tensor_view %14, shape = [%29, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf16, strides=[64,1]>
    %32 = make_token : token
    %33 = make_tensor_view %19, shape = [64], strides = [1] : tensor_view<64xi32, strides=[1]>
    %34, %35, %36 = get_tile_block_id : tile<i32>
    %37 = assume bounded<0, ?>, %34 : tile<i32>
    %38 = assume bounded<0, ?>, %35 : tile<i32>
    %39 = assume bounded<0, ?>, %36 : tile<i32>
    %40 = constant <i32: 4> : tile<i32>
    %41 = divi %37, %40 signed rounding negative_inf : tile<i32>
    %42 = constant <f32: 0.0> : tile<f32>
    %43 = constant <i32: 64> : tile<i32>
    %44 = constant <i32: 1> : tile<i32>
    %45 = constant <i32: 1> : tile<i32>
    %46 = reshape %42 : tile<f32> -> tile<1xf32>
    %47 = constant <i32: 1> : tile<i32>
    %48 = constant <i32: 64> : tile<i32>
    %49 = broadcast %46 : tile<1xf32> -> tile<64xf32>
    %50 = constant <i32: 0> : tile<i32>
    %51 = constant <i32: 64> : tile<i32>
    %52 = constant <i32: 1> : tile<i32>
    %113 = for %53 in (%50 to %51, step %52) : tile<i32> iter_values(%54 = %49) -> (tile<64xf32>) {
      %55 = assume bounded<0, 63>, %53 : tile<i32>
      %56 = constant <i32: 1> : tile<i32>
      %57 = constant <i32: 16> : tile<i32>
      %58 = constant <i32: -1> : tile<i32>
      %59 = constant <i32: 1024> : tile<i32>
      %60 = constant <i32: 1> : tile<i32>
      %61 = constant <i32: 16> : tile<i32>
      %62 = constant <i32: -1> : tile<i32>
      %63 = constant <i32: 1024> : tile<i32>
      %64 = constant <i32: -1> : tile<i32>
      %65 = constant <i32: 1024> : tile<i32>
      %66 = make_partition_view %28 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x1024xf32, strides=[1024,1]>>
      %67, %68 = load_view_tko weak %66[%37, %55] token = %27 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x1024xf32, strides=[1024,1]>>, tile<i32> -> tile<1x16xf32>, token
      %69 = constant <i32: 1> : tile<i32>
      %70 = constant <i32: 64> : tile<i32>
      %71 = constant <i32: 1> : tile<i32>
      %72 = constant <i32: 64> : tile<i32>
      %73 = constant <i32: 64> : tile<i32>
      %74 = make_partition_view %33 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>
      %75, %76 = load_view_tko weak %74[%55] token = %32 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %77 = constant <i32: 1> : tile<i32>
      %78 = reshape %75 : tile<1xi32> -> tile<i32>
      %79 = constant <i32: 4> : tile<i32>
      %80 = muli %78, %79 : tile<i32>
      %81 = addi %80, %41 : tile<i32>
      %82 = constant <i32: 0> : tile<i32>
      %83 = constant <i32: 16> : tile<i32>
      %84 = constant <i32: 64> : tile<i32>
      %85 = constant <i32: -1> : tile<i32>
      %86 = constant <i32: 64> : tile<i32>
      %87 = constant <i32: 16> : tile<i32>
      %88 = constant <i32: 64> : tile<i32>
      %89 = constant <i32: -1> : tile<i32>
      %90 = constant <i32: 64> : tile<i32>
      %91 = constant <i32: -1> : tile<i32>
      %92 = constant <i32: 64> : tile<i32>
      %93 = make_partition_view %31 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>
      %94, %95 = load_view_tko weak %93[%81, %82] token = %30 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>, tile<i32> -> tile<16x64xf16>, token
      %96 = ftof %94 : tile<16x64xf16> -> tile<16x64xf32>
      %97 = constant <i32: 1> : tile<i32>
      %98 = constant <i32: 16> : tile<i32>
      %99 = constant <i32: 16> : tile<i32>
      %100 = constant <i32: 1> : tile<i32>
      %101 = reshape %67 : tile<1x16xf32> -> tile<16x1xf32>
      %102 = constant <i32: 16> : tile<i32>
      %103 = constant <i32: 1> : tile<i32>
      %104 = constant <i32: 16> : tile<i32>
      %105 = constant <i32: 64> : tile<i32>
      %106 = broadcast %101 : tile<16x1xf32> -> tile<16x64xf32>
      %107 = mulf %106, %96 : tile<16x64xf32>
      %111 = reduce %107 dim=0 identities=[0] : tile<16x64xf32> -> tile<64xf32> {
      ^bb0(%108: tile<f32>, %109: tile<f32>):
        %110 = addf %108, %109 : tile<f32>
        yield %110 : tile<f32>
      }
      %112 = addf %54, %111 : tile<64xf32>
      continue %112 : tile<64xf32>
    }
    %114 = constant <i32: 64> : tile<i32>
    %115 = constant <i32: 1> : tile<i32>
    %116 = constant <i32: 64> : tile<i32>
    %117 = reshape %113 : tile<64xf32> -> tile<1x64xf32>
    %118 = constant <i32: 1> : tile<i32>
    %119 = constant <i32: 64> : tile<i32>
    %120 = constant <i32: 1> : tile<i32>
    %121 = constant <i32: 64> : tile<i32>
    %122, %123, %124 = get_tile_block_id : tile<i32>
    %125 = assume bounded<0, ?>, %122 : tile<i32>
    %126 = assume bounded<0, ?>, %123 : tile<i32>
    %127 = assume bounded<0, ?>, %124 : tile<i32>
    %128 = make_partition_view %25 : partition_view<tile=(1x64), tensor_view<?x?xf32, strides=[64,1]>>
    %129 = store_view_tko weak %117, %128[%125, %126] token = %24 : tile<1x64xf32>, partition_view<tile=(1x64), tensor_view<?x?xf32, strides=[64,1]>>, tile<i32> -> token
    return
  }
}
