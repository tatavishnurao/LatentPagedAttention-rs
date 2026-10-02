cuda_tile.module @paged_source_hint {
  entry @model_small_context_fp16_storage_rtable_1024_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>, %22: tile<ptr<f32>>, %23: tile<i32>, %24: tile<i32>, %25: tile<i32>, %26: tile<i32>) {
    %27 = assume bounded<0, ?>, %1 : tile<i32>
    %28 = assume div_by<16>, %27 : tile<i32>
    %29 = assume bounded<0, ?>, %2 : tile<i32>
    %30 = assume div_by<16>, %29 : tile<i32>
    %31 = make_token : token
    %32 = assume div_by<16>, %0 : tile<ptr<f32>>
    %33 = make_tensor_view %32, shape = [%28, %30], strides = [64, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[64,1]>
    %34 = assume bounded<0, ?>, %10 : tile<i32>
    %35 = assume div_by<16>, %34 : tile<i32>
    %36 = make_token : token
    %37 = assume div_by<16>, %9 : tile<ptr<f32>>
    %38 = make_tensor_view %37, shape = [%35, 1024], strides = [1024, 1] : tile<i32> -> tensor_view<?x1024xf32, strides=[1024,1]>
    %39 = assume bounded<0, ?>, %15 : tile<i32>
    %40 = assume div_by<16>, %39 : tile<i32>
    %41 = make_token : token
    %42 = assume div_by<16>, %14 : tile<ptr<f16>>
    %43 = make_tensor_view %42, shape = [%40, 32], strides = [32, 1] : tile<i32> -> tensor_view<?x32xf16, strides=[32,1]>
    %44 = make_token : token
    %45 = assume div_by<16>, %19 : tile<ptr<i32>>
    %46 = make_tensor_view %45, shape = [64], strides = [1] : tensor_view<64xi32, strides=[1]>
    %47 = assume bounded<0, ?>, %23 : tile<i32>
    %48 = assume div_by<16>, %47 : tile<i32>
    %49 = make_token : token
    %50 = assume div_by<16>, %22 : tile<ptr<f32>>
    %51 = make_tensor_view %50, shape = [%48, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf32, strides=[64,1]>
    %52, %53, %54 = get_tile_block_id : tile<i32>
    %55 = assume bounded<0, ?>, %52 : tile<i32>
    %56 = assume bounded<0, ?>, %53 : tile<i32>
    %57 = assume bounded<0, ?>, %54 : tile<i32>
    %58 = constant <i32: 4> : tile<i32>
    %59 = divi %55, %58 signed rounding negative_inf : tile<i32>
    %60 = constant <f32: 0.0> : tile<f32>
    %61 = constant <i32: 32> : tile<i32>
    %62 = constant <i32: 1> : tile<i32>
    %63 = constant <i32: 1> : tile<i32>
    %64 = reshape %60 : tile<f32> -> tile<1xf32>
    %65 = constant <i32: 1> : tile<i32>
    %66 = constant <i32: 32> : tile<i32>
    %67 = broadcast %64 : tile<1xf32> -> tile<32xf32>
    %68 = constant <i32: 0> : tile<i32>
    %69 = constant <i32: 64> : tile<i32>
    %70 = constant <i32: 1> : tile<i32>
    %128 = for %71 in (%68 to %69, step %70) : tile<i32> iter_values(%72 = %67) -> (tile<32xf32>) {
      %73 = assume bounded<0, 63>, %71 : tile<i32>
      %74 = constant <i32: 1> : tile<i32>
      %75 = constant <i32: 16> : tile<i32>
      %76 = constant <i32: -1> : tile<i32>
      %77 = constant <i32: 1024> : tile<i32>
      %78 = constant <i32: 1> : tile<i32>
      %79 = constant <i32: 16> : tile<i32>
      %80 = constant <i32: -1> : tile<i32>
      %81 = constant <i32: 1024> : tile<i32>
      %82 = constant <i32: -1> : tile<i32>
      %83 = constant <i32: 1024> : tile<i32>
      %84 = make_partition_view %38 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x1024xf32, strides=[1024,1]>>
      %85, %86 = load_view_tko weak %84[%55, %73] token = %36 : partition_view<tile=(1x16), padding_value = zero, tensor_view<?x1024xf32, strides=[1024,1]>>, tile<i32> -> tile<1x16xf32>, token
      %87 = constant <i32: 1> : tile<i32>
      %88 = constant <i32: 64> : tile<i32>
      %89 = constant <i32: 1> : tile<i32>
      %90 = constant <i32: 64> : tile<i32>
      %91 = constant <i32: 64> : tile<i32>
      %92 = make_partition_view %46 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>
      %93, %94 = load_view_tko weak %92[%73] token = %44 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %95 = constant <i32: 1> : tile<i32>
      %96 = reshape %93 : tile<1xi32> -> tile<i32>
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
      %108 = make_partition_view %43 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %109, %110 = load_view_tko weak %108[%96, %97] token = %41 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %111 = ftof %109 : tile<16x32xf16> -> tile<16x32xf32>
      %112 = constant <i32: 1> : tile<i32>
      %113 = constant <i32: 16> : tile<i32>
      %114 = constant <i32: 16> : tile<i32>
      %115 = constant <i32: 1> : tile<i32>
      %116 = reshape %85 : tile<1x16xf32> -> tile<16x1xf32>
      %117 = constant <i32: 16> : tile<i32>
      %118 = constant <i32: 1> : tile<i32>
      %119 = constant <i32: 16> : tile<i32>
      %120 = constant <i32: 32> : tile<i32>
      %121 = broadcast %116 : tile<16x1xf32> -> tile<16x32xf32>
      %122 = mulf %121, %111 : tile<16x32xf32>
      %126 = reduce %122 dim=0 identities=[0] : tile<16x32xf32> -> tile<32xf32> {
      ^bb0(%123: tile<f32>, %124: tile<f32>):
        %125 = addf %123, %124 : tile<f32>
        yield %125 : tile<f32>
      }
      %127 = addf %72, %126 : tile<32xf32>
      continue %127 : tile<32xf32>
    }
    %129 = constant <i32: 0> : tile<i32>
    %130 = constant <i32: 32> : tile<i32>
    %131 = constant <i32: 64> : tile<i32>
    %132 = constant <i32: -1> : tile<i32>
    %133 = constant <i32: 64> : tile<i32>
    %134 = constant <i32: 32> : tile<i32>
    %135 = constant <i32: 64> : tile<i32>
    %136 = constant <i32: -1> : tile<i32>
    %137 = constant <i32: 64> : tile<i32>
    %138 = constant <i32: -1> : tile<i32>
    %139 = constant <i32: 64> : tile<i32>
    %140 = make_partition_view %51 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>
    %141, %142 = load_view_tko weak %140[%59, %129] token = %49 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>, tile<i32> -> tile<32x64xf32>, token
    %143 = constant <i32: 32> : tile<i32>
    %144 = constant <i32: 32> : tile<i32>
    %145 = constant <i32: 1> : tile<i32>
    %146 = reshape %128 : tile<32xf32> -> tile<32x1xf32>
    %147 = constant <i32: 32> : tile<i32>
    %148 = constant <i32: 1> : tile<i32>
    %149 = constant <i32: 32> : tile<i32>
    %150 = constant <i32: 64> : tile<i32>
    %151 = broadcast %146 : tile<32x1xf32> -> tile<32x64xf32>
    %152 = mulf %151, %141 : tile<32x64xf32>
    %156 = reduce %152 dim=0 identities=[0] : tile<32x64xf32> -> tile<64xf32> {
    ^bb0(%153: tile<f32>, %154: tile<f32>):
      %155 = addf %153, %154 : tile<f32>
      yield %155 : tile<f32>
    }
    %157 = constant <i32: 64> : tile<i32>
    %158 = constant <i32: 1> : tile<i32>
    %159 = constant <i32: 64> : tile<i32>
    %160 = reshape %156 : tile<64xf32> -> tile<1x64xf32>
    %161 = constant <i32: 1> : tile<i32>
    %162 = constant <i32: 64> : tile<i32>
    %163 = constant <i32: 1> : tile<i32>
    %164 = constant <i32: 64> : tile<i32>
    %165, %166, %167 = get_tile_block_id : tile<i32>
    %168 = assume bounded<0, ?>, %165 : tile<i32>
    %169 = assume bounded<0, ?>, %166 : tile<i32>
    %170 = assume bounded<0, ?>, %167 : tile<i32>
    %171 = make_partition_view %33 : partition_view<tile=(1x64), tensor_view<?x?xf32, strides=[64,1]>>
    %172 = store_view_tko weak %160, %171[%168, %169] token = %31 : tile<1x64xf32>, partition_view<tile=(1x64), tensor_view<?x?xf32, strides=[64,1]>>, tile<i32> -> token
    return
  }
}
