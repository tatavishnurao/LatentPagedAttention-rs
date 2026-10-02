cuda_tile.module @paged_nohint {
  entry @model_small_scores_fp16_storage_rtable_1024_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>, %22: tile<ptr<i32>>, %23: tile<i32>, %24: tile<i32>, %25: tile<ptr<f32>>, %26: tile<i32>, %27: tile<i32>, %28: tile<i32>, %29: tile<i32>) {
    %30 = assume bounded<0, ?>, %1 : tile<i32>
    %31 = assume bounded<0, ?>, %2 : tile<i32>
    %32 = make_token : token
    %33 = make_tensor_view %0, shape = [%30, %31], strides = [1024, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[1024,1]>
    %34 = assume bounded<0, ?>, %10 : tile<i32>
    %35 = make_token : token
    %36 = make_tensor_view %9, shape = [%34, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf32, strides=[64,1]>
    %37 = assume bounded<0, ?>, %15 : tile<i32>
    %38 = make_token : token
    %39 = make_tensor_view %14, shape = [%37, 32], strides = [32, 1] : tile<i32> -> tensor_view<?x32xf16, strides=[32,1]>
    %40 = make_token : token
    %41 = make_tensor_view %19, shape = [64], strides = [1] : tensor_view<64xi32, strides=[1]>
    %42 = make_token : token
    %43 = make_tensor_view %22, shape = [1], strides = [1] : tensor_view<1xi32, strides=[1]>
    %44 = assume bounded<0, ?>, %26 : tile<i32>
    %45 = make_token : token
    %46 = make_tensor_view %25, shape = [%44, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf32, strides=[64,1]>
    %47, %48, %49 = get_tile_block_id : tile<i32>
    %50 = assume bounded<0, ?>, %47 : tile<i32>
    %51 = assume bounded<0, ?>, %48 : tile<i32>
    %52 = assume bounded<0, ?>, %49 : tile<i32>
    %53 = constant <i32: 4> : tile<i32>
    %54 = divi %50, %53 signed rounding negative_inf : tile<i32>
    %55 = constant <i32: 0> : tile<i32>
    %56 = constant <i32: 1> : tile<i32>
    %57 = constant <i32: 1> : tile<i32>
    %58 = constant <i32: 1> : tile<i32>
    %59 = constant <i32: 1> : tile<i32>
    %60 = constant <i32: 1> : tile<i32>
    %61 = make_partition_view %43 : partition_view<tile=(1), padding_value = zero, tensor_view<1xi32, strides=[1]>>
    %62, %63 = load_view_tko weak %61[%55] token = %42 : partition_view<tile=(1), padding_value = zero, tensor_view<1xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
    %64 = constant <i32: 1> : tile<i32>
    %65 = constant <i32: 64> : tile<i32>
    %66 = constant <i32: 1> : tile<i32>
    %67 = constant <i32: 64> : tile<i32>
    %68 = constant <i32: 64> : tile<i32>
    %69 = make_partition_view %41 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>
    %70, %71 = load_view_tko weak %69[%51] token = %40 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
    %72 = constant <i32: 1> : tile<i32>
    %73 = reshape %70 : tile<1xi32> -> tile<i32>
    %74 = constant <i32: 0> : tile<i32>
    %75 = constant <i32: 1> : tile<i32>
    %76 = constant <i32: 64> : tile<i32>
    %77 = constant <i32: -1> : tile<i32>
    %78 = constant <i32: 64> : tile<i32>
    %79 = constant <i32: 1> : tile<i32>
    %80 = constant <i32: 64> : tile<i32>
    %81 = constant <i32: -1> : tile<i32>
    %82 = constant <i32: 64> : tile<i32>
    %83 = constant <i32: -1> : tile<i32>
    %84 = constant <i32: 64> : tile<i32>
    %85 = make_partition_view %36 : partition_view<tile=(1x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>
    %86, %87 = load_view_tko weak %85[%50, %74] token = %35 : partition_view<tile=(1x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>, tile<i32> -> tile<1x64xf32>, token
    %88 = constant <i32: 0> : tile<i32>
    %89 = constant <i32: 32> : tile<i32>
    %90 = constant <i32: 64> : tile<i32>
    %91 = constant <i32: -1> : tile<i32>
    %92 = constant <i32: 64> : tile<i32>
    %93 = constant <i32: 32> : tile<i32>
    %94 = constant <i32: 64> : tile<i32>
    %95 = constant <i32: -1> : tile<i32>
    %96 = constant <i32: 64> : tile<i32>
    %97 = constant <i32: -1> : tile<i32>
    %98 = constant <i32: 64> : tile<i32>
    %99 = make_partition_view %46 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>
    %100, %101 = load_view_tko weak %99[%54, %88] token = %45 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>, tile<i32> -> tile<32x64xf32>, token
    %102 = constant <i32: 1> : tile<i32>
    %103 = constant <i32: 64> : tile<i32>
    %104 = constant <i32: 32> : tile<i32>
    %105 = constant <i32: 64> : tile<i32>
    %106 = broadcast %86 : tile<1x64xf32> -> tile<32x64xf32>
    %107 = mulf %100, %106 : tile<32x64xf32>
    %111 = reduce %107 dim=1 identities=[0] : tile<32x64xf32> -> tile<32xf32> {
    ^bb0(%108: tile<f32>, %109: tile<f32>):
      %110 = addf %108, %109 : tile<f32>
      yield %110 : tile<f32>
    }
    %112 = constant <i32: 0> : tile<i32>
    %113 = constant <i32: 16> : tile<i32>
    %114 = constant <i32: 32> : tile<i32>
    %115 = constant <i32: -1> : tile<i32>
    %116 = constant <i32: 32> : tile<i32>
    %117 = constant <i32: 16> : tile<i32>
    %118 = constant <i32: 32> : tile<i32>
    %119 = constant <i32: -1> : tile<i32>
    %120 = constant <i32: 32> : tile<i32>
    %121 = constant <i32: -1> : tile<i32>
    %122 = constant <i32: 32> : tile<i32>
    %123 = make_partition_view %39 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
    %124, %125 = load_view_tko weak %123[%73, %112] token = %38 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
    %126 = ftof %124 : tile<16x32xf16> -> tile<16x32xf32>
    %127 = constant <i32: 32> : tile<i32>
    %128 = constant <i32: 1> : tile<i32>
    %129 = constant <i32: 32> : tile<i32>
    %130 = reshape %111 : tile<32xf32> -> tile<1x32xf32>
    %131 = constant <i32: 1> : tile<i32>
    %132 = constant <i32: 32> : tile<i32>
    %133 = constant <i32: 16> : tile<i32>
    %134 = constant <i32: 32> : tile<i32>
    %135 = broadcast %130 : tile<1x32xf32> -> tile<16x32xf32>
    %136 = mulf %126, %135 : tile<16x32xf32>
    %140 = reduce %136 dim=1 identities=[0] : tile<16x32xf32> -> tile<16xf32> {
    ^bb0(%137: tile<f32>, %138: tile<f32>):
      %139 = addf %137, %138 : tile<f32>
      yield %139 : tile<f32>
    }
    %141 = constant <f32: 0.125> : tile<f32>
    %142 = constant <i32: 16> : tile<i32>
    %143 = constant <i32: 1> : tile<i32>
    %144 = constant <i32: 1> : tile<i32>
    %145 = reshape %141 : tile<f32> -> tile<1xf32>
    %146 = constant <i32: 1> : tile<i32>
    %147 = constant <i32: 16> : tile<i32>
    %148 = broadcast %145 : tile<1xf32> -> tile<16xf32>
    %149 = mulf %140, %148 : tile<16xf32>
    %150 = iota : tile<16xi32>
    %151 = constant <i32: 16> : tile<i32>
    %152 = muli %51, %151 : tile<i32>
    %153 = constant <i32: 16> : tile<i32>
    %154 = constant <i32: 1> : tile<i32>
    %155 = constant <i32: 1> : tile<i32>
    %156 = reshape %152 : tile<i32> -> tile<1xi32>
    %157 = constant <i32: 1> : tile<i32>
    %158 = constant <i32: 16> : tile<i32>
    %159 = broadcast %156 : tile<1xi32> -> tile<16xi32>
    %160 = addi %150, %159 : tile<16xi32>
    %161 = constant <i32: 1> : tile<i32>
    %162 = constant <i32: 16> : tile<i32>
    %163 = broadcast %62 : tile<1xi32> -> tile<16xi32>
    %164 = cmpi less_than %160, %163, signed : tile<16xi32> -> tile<16xi1>
    %165 = constant <f32: -340282346638528860000000000000000000000.0> : tile<f32>
    %166 = constant <i32: 16> : tile<i32>
    %167 = constant <i32: 1> : tile<i32>
    %168 = constant <i32: 1> : tile<i32>
    %169 = reshape %165 : tile<f32> -> tile<1xf32>
    %170 = constant <i32: 1> : tile<i32>
    %171 = constant <i32: 16> : tile<i32>
    %172 = broadcast %169 : tile<1xf32> -> tile<16xf32>
    %173 = select %164, %149, %172 : tile<16xi1>, tile<16xf32>
    %174 = constant <i32: 16> : tile<i32>
    %175 = constant <i32: 1> : tile<i32>
    %176 = constant <i32: 16> : tile<i32>
    %177 = reshape %173 : tile<16xf32> -> tile<1x16xf32>
    %178 = constant <i32: 1> : tile<i32>
    %179 = constant <i32: 16> : tile<i32>
    %180 = constant <i32: 1> : tile<i32>
    %181 = constant <i32: 16> : tile<i32>
    %182, %183, %184 = get_tile_block_id : tile<i32>
    %185 = assume bounded<0, ?>, %182 : tile<i32>
    %186 = assume bounded<0, ?>, %183 : tile<i32>
    %187 = assume bounded<0, ?>, %184 : tile<i32>
    %188 = make_partition_view %33 : partition_view<tile=(1x16), tensor_view<?x?xf32, strides=[1024,1]>>
    %189 = store_view_tko weak %177, %188[%185, %186] token = %32 : tile<1x16xf32>, partition_view<tile=(1x16), tensor_view<?x?xf32, strides=[1024,1]>>, tile<i32> -> token
    return
  }
}
