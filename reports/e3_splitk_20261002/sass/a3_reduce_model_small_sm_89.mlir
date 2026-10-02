cuda_tile.module @e3_kernels {
  entry @a3_reduce_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<i32>, %15: tile<i32>, %16: tile<ptr<f32>>, %17: tile<i32>, %18: tile<i32>, %19: tile<i32>, %20: tile<i32>, %21: tile<i32>, %22: tile<i32>, %23: tile<ptr<f32>>, %24: tile<i32>, %25: tile<i32>, %26: tile<i32>, %27: tile<i32>, %28: tile<i32>, %29: tile<i32>, %30: tile<i32>) {
    %31 = constant <i32: 16> : tile<i32>
    %32 = constant <i32: 4> : tile<i32>
    %33 = constant <i32: 64> : tile<i32>
    %34 = assume bounded<0, ?>, %1 : tile<i32>
    %35 = assume div_by<16>, %34 : tile<i32>
    %36 = assume bounded<0, ?>, %2 : tile<i32>
    %37 = assume div_by<16>, %36 : tile<i32>
    %38 = make_token : token
    %39 = assume div_by<16>, %0 : tile<ptr<f32>>
    %40 = make_tensor_view %39, shape = [%35, %37], strides = [64, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[64,1]>
    %41 = assume bounded<0, ?>, %10 : tile<i32>
    %42 = assume div_by<4>, %41 : tile<i32>
    %43 = make_token : token
    %44 = assume div_by<16>, %9 : tile<ptr<f32>>
    %45 = make_tensor_view %44, shape = [%42, 16, 64], strides = [1024, 64, 1] : tile<i32> -> tensor_view<?x16x64xf32, strides=[1024,64,1]>
    %46 = assume bounded<0, ?>, %17 : tile<i32>
    %47 = assume div_by<4>, %46 : tile<i32>
    %48 = make_token : token
    %49 = assume div_by<16>, %16 : tile<ptr<f32>>
    %50 = make_tensor_view %49, shape = [%47, 16, 1], strides = [16, 1, 1] : tile<i32> -> tensor_view<?x16x1xf32, strides=[16,1,1]>
    %51 = assume bounded<0, ?>, %24 : tile<i32>
    %52 = assume div_by<4>, %51 : tile<i32>
    %53 = make_token : token
    %54 = assume div_by<16>, %23 : tile<ptr<f32>>
    %55 = make_tensor_view %54, shape = [%52, 16, 1], strides = [16, 1, 1] : tile<i32> -> tensor_view<?x16x1xf32, strides=[16,1,1]>
    %56 = constant <i32: 16> : tile<i32>
    %57 = constant <i32: 4> : tile<i32>
    %58 = constant <i32: 64> : tile<i32>
    %59, %60, %61 = get_tile_block_id : tile<i32>
    %62 = assume bounded<0, ?>, %59 : tile<i32>
    %63 = assume bounded<0, ?>, %60 : tile<i32>
    %64 = assume bounded<0, ?>, %61 : tile<i32>
    %65 = constant <f32: -1000000015047466200000000000000.0> : tile<4x1xf32>
    %66 = constant <f32: 0.0> : tile<4x1xf32>
    %67 = constant <f32: 0.0> : tile<4x64xf32>
    %68 = constant <i32: 0> : tile<i32>
    %69 = constant <i32: 1> : tile<i32>
    %171, %172, %173 = for %70 in (%68 to %30, step %69) : tile<i32> iter_values(%71 = %67, %72 = %66, %73 = %65) -> (tile<4x64xf32>, tile<4x1xf32>, tile<4x1xf32>) {
      %74 = assume bounded<0, ?>, %70 : tile<i32>
      %75 = constant <i32: 0> : tile<i32>
      %76 = constant <i32: 1> : tile<i32>
      %77 = constant <i32: 4> : tile<i32>
      %78 = constant <i32: 64> : tile<i32>
      %79 = constant <i32: -1> : tile<i32>
      %80 = constant <i32: 16> : tile<i32>
      %81 = constant <i32: 64> : tile<i32>
      %82 = constant <i32: 1> : tile<i32>
      %83 = constant <i32: 4> : tile<i32>
      %84 = constant <i32: 64> : tile<i32>
      %85 = constant <i32: -1> : tile<i32>
      %86 = constant <i32: 16> : tile<i32>
      %87 = constant <i32: 64> : tile<i32>
      %88 = constant <i32: -1> : tile<i32>
      %89 = constant <i32: 16> : tile<i32>
      %90 = constant <i32: 64> : tile<i32>
      %91 = make_partition_view %45 : partition_view<tile=(1x4x64), padding_value = zero, tensor_view<?x16x64xf32, strides=[1024,64,1]>>
      %92, %93 = load_view_tko weak %91[%74, %62, %75] token = %43 : partition_view<tile=(1x4x64), padding_value = zero, tensor_view<?x16x64xf32, strides=[1024,64,1]>>, tile<i32> -> tile<1x4x64xf32>, token
      %94 = constant <i32: 0> : tile<i32>
      %95 = constant <i32: 1> : tile<i32>
      %96 = constant <i32: 4> : tile<i32>
      %97 = constant <i32: 1> : tile<i32>
      %98 = constant <i32: -1> : tile<i32>
      %99 = constant <i32: 16> : tile<i32>
      %100 = constant <i32: 1> : tile<i32>
      %101 = constant <i32: 1> : tile<i32>
      %102 = constant <i32: 4> : tile<i32>
      %103 = constant <i32: 1> : tile<i32>
      %104 = constant <i32: -1> : tile<i32>
      %105 = constant <i32: 16> : tile<i32>
      %106 = constant <i32: 1> : tile<i32>
      %107 = constant <i32: -1> : tile<i32>
      %108 = constant <i32: 16> : tile<i32>
      %109 = constant <i32: 1> : tile<i32>
      %110 = make_partition_view %50 : partition_view<tile=(1x4x1), padding_value = zero, tensor_view<?x16x1xf32, strides=[16,1,1]>>
      %111, %112 = load_view_tko weak %110[%74, %62, %94] token = %48 : partition_view<tile=(1x4x1), padding_value = zero, tensor_view<?x16x1xf32, strides=[16,1,1]>>, tile<i32> -> tile<1x4x1xf32>, token
      %113 = constant <i32: 0> : tile<i32>
      %114 = constant <i32: 1> : tile<i32>
      %115 = constant <i32: 4> : tile<i32>
      %116 = constant <i32: 1> : tile<i32>
      %117 = constant <i32: -1> : tile<i32>
      %118 = constant <i32: 16> : tile<i32>
      %119 = constant <i32: 1> : tile<i32>
      %120 = constant <i32: 1> : tile<i32>
      %121 = constant <i32: 4> : tile<i32>
      %122 = constant <i32: 1> : tile<i32>
      %123 = constant <i32: -1> : tile<i32>
      %124 = constant <i32: 16> : tile<i32>
      %125 = constant <i32: 1> : tile<i32>
      %126 = constant <i32: -1> : tile<i32>
      %127 = constant <i32: 16> : tile<i32>
      %128 = constant <i32: 1> : tile<i32>
      %129 = make_partition_view %55 : partition_view<tile=(1x4x1), padding_value = zero, tensor_view<?x16x1xf32, strides=[16,1,1]>>
      %130, %131 = load_view_tko weak %129[%74, %62, %113] token = %53 : partition_view<tile=(1x4x1), padding_value = zero, tensor_view<?x16x1xf32, strides=[16,1,1]>>, tile<i32> -> tile<1x4x1xf32>, token
      %132 = constant <i32: 1> : tile<i32>
      %133 = constant <i32: 4> : tile<i32>
      %134 = constant <i32: 1> : tile<i32>
      %135 = constant <i32: 4> : tile<i32>
      %136 = constant <i32: 1> : tile<i32>
      %137 = reshape %111 : tile<1x4x1xf32> -> tile<4x1xf32>
      %138 = maxf %73, %137 {rounding_mode = 0} : tile<4x1xf32>
      %139 = subf %73, %138 : tile<4x1xf32>
      %140 = exp %139 : tile<4x1xf32>
      %141 = subf %137, %138 : tile<4x1xf32>
      %142 = exp %141 : tile<4x1xf32>
      %143 = mulf %72, %140 : tile<4x1xf32>
      %144 = constant <i32: 1> : tile<i32>
      %145 = constant <i32: 4> : tile<i32>
      %146 = constant <i32: 1> : tile<i32>
      %147 = constant <i32: 4> : tile<i32>
      %148 = constant <i32: 1> : tile<i32>
      %149 = reshape %130 : tile<1x4x1xf32> -> tile<4x1xf32>
      %150 = mulf %149, %142 : tile<4x1xf32>
      %151 = addf %143, %150 : tile<4x1xf32>
      %152 = constant <i32: 4> : tile<i32>
      %153 = constant <i32: 1> : tile<i32>
      %154 = constant <i32: 4> : tile<i32>
      %155 = constant <i32: 64> : tile<i32>
      %156 = broadcast %140 : tile<4x1xf32> -> tile<4x64xf32>
      %157 = mulf %71, %156 : tile<4x64xf32>
      %158 = constant <i32: 1> : tile<i32>
      %159 = constant <i32: 4> : tile<i32>
      %160 = constant <i32: 64> : tile<i32>
      %161 = constant <i32: 4> : tile<i32>
      %162 = constant <i32: 64> : tile<i32>
      %163 = reshape %92 : tile<1x4x64xf32> -> tile<4x64xf32>
      %164 = constant <i32: 4> : tile<i32>
      %165 = constant <i32: 1> : tile<i32>
      %166 = constant <i32: 4> : tile<i32>
      %167 = constant <i32: 64> : tile<i32>
      %168 = broadcast %142 : tile<4x1xf32> -> tile<4x64xf32>
      %169 = mulf %163, %168 : tile<4x64xf32>
      %170 = addf %157, %169 : tile<4x64xf32>
      continue %170, %151, %138 : tile<4x64xf32>, tile<4x1xf32>, tile<4x1xf32>
    }
    %174 = constant <i32: 4> : tile<i32>
    %175 = constant <i32: 1> : tile<i32>
    %176 = constant <i32: 4> : tile<i32>
    %177 = constant <i32: 64> : tile<i32>
    %178 = broadcast %172 : tile<4x1xf32> -> tile<4x64xf32>
    %179 = divf %171, %178 rounding<approx> flush_to_zero : tile<4x64xf32>
    %180 = constant <i32: 4> : tile<i32>
    %181 = constant <i32: 64> : tile<i32>
    %182 = constant <i32: 4> : tile<i32>
    %183 = constant <i32: 64> : tile<i32>
    %184, %185, %186 = get_tile_block_id : tile<i32>
    %187 = assume bounded<0, ?>, %184 : tile<i32>
    %188 = assume bounded<0, ?>, %185 : tile<i32>
    %189 = assume bounded<0, ?>, %186 : tile<i32>
    %190 = make_partition_view %40 : partition_view<tile=(4x64), tensor_view<?x?xf32, strides=[64,1]>>
    %191 = store_view_tko weak %179, %190[%187, %188] token = %38 : tile<4x64xf32>, partition_view<tile=(4x64), tensor_view<?x?xf32, strides=[64,1]>>, tile<i32> -> token
    return
  }
}
