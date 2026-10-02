cuda_tile.module @paged_nohint {
  entry @model_small_context_fp16_storage_rtable_1024_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>, %22: tile<ptr<f32>>, %23: tile<i32>, %24: tile<i32>, %25: tile<i32>, %26: tile<i32>) {
    %27 = assume bounded<0, ?>, %1 : tile<i32>
    %28 = assume bounded<0, ?>, %2 : tile<i32>
    %29 = make_token : token
    %30 = make_tensor_view %0, shape = [%27, %28], strides = [64, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[64,1]>
    %31 = assume bounded<0, ?>, %10 : tile<i32>
    %32 = make_token : token
    %33 = make_tensor_view %9, shape = [%31, 1024], strides = [1024, 1] : tile<i32> -> tensor_view<?x1024xf32, strides=[1024,1]>
    %34 = assume bounded<0, ?>, %15 : tile<i32>
    %35 = make_token : token
    %36 = make_tensor_view %14, shape = [%34, 32], strides = [32, 1] : tile<i32> -> tensor_view<?x32xf16, strides=[32,1]>
    %37 = make_token : token
    %38 = make_tensor_view %19, shape = [64], strides = [1] : tensor_view<64xi32, strides=[1]>
    %39 = assume bounded<0, ?>, %23 : tile<i32>
    %40 = make_token : token
    %41 = make_tensor_view %22, shape = [%39, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf32, strides=[64,1]>
    %42, %43, %44 = get_tile_block_id : tile<i32>
    %45 = assume bounded<0, ?>, %42 : tile<i32>
    %46 = assume bounded<0, ?>, %43 : tile<i32>
    %47 = assume bounded<0, ?>, %44 : tile<i32>
    %48 = constant <i32: 4> : tile<i32>
    %49 = divi %45, %48 signed rounding negative_inf : tile<i32>
    %50 = constant <f32: 0.0> : tile<f32>
    %51 = constant <i32: 32> : tile<i32>
    %52 = constant <i32: 1> : tile<i32>
    %53 = constant <i32: 1> : tile<i32>
    %54 = reshape %50 : tile<f32> -> tile<1xf32>
    %55 = constant <i32: 1> : tile<i32>
    %56 = constant <i32: 32> : tile<i32>
    %57 = broadcast %54 : tile<1xf32> -> tile<32xf32>
    %58 = constant <i32: 0> : tile<i32>
    %59 = constant <i32: 64> : tile<i32>
    %60 = constant <i32: 1> : tile<i32>
    %118 = for %61 in (%58 to %59, step %60) : tile<i32> iter_values(%62 = %57) -> (tile<32xf32>) {
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
      %75, %76 = load_view_tko weak %74[%45, %63] token = %32 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x1024xf32, strides=[1024,1]>>, tile<i32> -> tile<1x16xf32>, token
      %77 = constant <i32: 1> : tile<i32>
      %78 = constant <i32: 64> : tile<i32>
      %79 = constant <i32: 1> : tile<i32>
      %80 = constant <i32: 64> : tile<i32>
      %81 = constant <i32: 64> : tile<i32>
      %82 = make_partition_view %38 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>
      %83, %84 = load_view_tko weak %82[%63] token = %37 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %85 = constant <i32: 1> : tile<i32>
      %86 = reshape %83 : tile<1xi32> -> tile<i32>
      %87 = constant <i32: 0> : tile<i32>
      %88 = constant <i32: 16> : tile<i32>
      %89 = constant <i32: 32> : tile<i32>
      %90 = constant <i32: -1> : tile<i32>
      %91 = constant <i32: 32> : tile<i32>
      %92 = constant <i32: 16> : tile<i32>
      %93 = constant <i32: 32> : tile<i32>
      %94 = constant <i32: -1> : tile<i32>
      %95 = constant <i32: 32> : tile<i32>
      %96 = constant <i32: -1> : tile<i32>
      %97 = constant <i32: 32> : tile<i32>
      %98 = make_partition_view %36 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %99, %100 = load_view_tko weak %98[%86, %87] token = %35 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %101 = ftof %99 : tile<16x32xf16> -> tile<16x32xf32>
      %102 = constant <i32: 1> : tile<i32>
      %103 = constant <i32: 16> : tile<i32>
      %104 = constant <i32: 16> : tile<i32>
      %105 = constant <i32: 1> : tile<i32>
      %106 = reshape %75 : tile<1x16xf32> -> tile<16x1xf32>
      %107 = constant <i32: 16> : tile<i32>
      %108 = constant <i32: 1> : tile<i32>
      %109 = constant <i32: 16> : tile<i32>
      %110 = constant <i32: 32> : tile<i32>
      %111 = broadcast %106 : tile<16x1xf32> -> tile<16x32xf32>
      %112 = mulf %111, %101 : tile<16x32xf32>
      %116 = reduce %112 dim=0 identities=[0] : tile<16x32xf32> -> tile<32xf32> {
      ^bb0(%113: tile<f32>, %114: tile<f32>):
        %115 = addf %113, %114 : tile<f32>
        yield %115 : tile<f32>
      }
      %117 = addf %62, %116 : tile<32xf32>
      continue %117 : tile<32xf32>
    }
    %119 = constant <i32: 0> : tile<i32>
    %120 = constant <i32: 32> : tile<i32>
    %121 = constant <i32: 64> : tile<i32>
    %122 = constant <i32: -1> : tile<i32>
    %123 = constant <i32: 64> : tile<i32>
    %124 = constant <i32: 32> : tile<i32>
    %125 = constant <i32: 64> : tile<i32>
    %126 = constant <i32: -1> : tile<i32>
    %127 = constant <i32: 64> : tile<i32>
    %128 = constant <i32: -1> : tile<i32>
    %129 = constant <i32: 64> : tile<i32>
    %130 = make_partition_view %41 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>
    %131, %132 = load_view_tko weak %130[%49, %119] token = %40 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>, tile<i32> -> tile<32x64xf32>, token
    %133 = constant <i32: 32> : tile<i32>
    %134 = constant <i32: 32> : tile<i32>
    %135 = constant <i32: 1> : tile<i32>
    %136 = reshape %118 : tile<32xf32> -> tile<32x1xf32>
    %137 = constant <i32: 32> : tile<i32>
    %138 = constant <i32: 1> : tile<i32>
    %139 = constant <i32: 32> : tile<i32>
    %140 = constant <i32: 64> : tile<i32>
    %141 = broadcast %136 : tile<32x1xf32> -> tile<32x64xf32>
    %142 = mulf %141, %131 : tile<32x64xf32>
    %146 = reduce %142 dim=0 identities=[0] : tile<32x64xf32> -> tile<64xf32> {
    ^bb0(%143: tile<f32>, %144: tile<f32>):
      %145 = addf %143, %144 : tile<f32>
      yield %145 : tile<f32>
    }
    %147 = constant <i32: 64> : tile<i32>
    %148 = constant <i32: 1> : tile<i32>
    %149 = constant <i32: 64> : tile<i32>
    %150 = reshape %146 : tile<64xf32> -> tile<1x64xf32>
    %151 = constant <i32: 1> : tile<i32>
    %152 = constant <i32: 64> : tile<i32>
    %153 = constant <i32: 1> : tile<i32>
    %154 = constant <i32: 64> : tile<i32>
    %155, %156, %157 = get_tile_block_id : tile<i32>
    %158 = assume bounded<0, ?>, %155 : tile<i32>
    %159 = assume bounded<0, ?>, %156 : tile<i32>
    %160 = assume bounded<0, ?>, %157 : tile<i32>
    %161 = make_partition_view %30 : partition_view<tile=(1x64), tensor_view<?x?xf32, strides=[64,1]>>
    %162 = store_view_tko weak %150, %161[%158, %159] token = %29 : tile<1x64xf32>, partition_view<tile=(1x64), tensor_view<?x?xf32, strides=[64,1]>>, tile<i32> -> token
    return
  }
}
