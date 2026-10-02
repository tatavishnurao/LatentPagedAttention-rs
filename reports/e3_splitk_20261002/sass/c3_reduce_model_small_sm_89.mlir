cuda_tile.module @e3_kernels {
  entry @c3_reduce_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<i32>, %15: tile<i32>, %16: tile<ptr<f32>>, %17: tile<i32>, %18: tile<i32>, %19: tile<i32>, %20: tile<i32>, %21: tile<i32>, %22: tile<i32>, %23: tile<ptr<f32>>, %24: tile<i32>, %25: tile<i32>, %26: tile<i32>, %27: tile<i32>, %28: tile<i32>, %29: tile<i32>, %30: tile<ptr<f32>>, %31: tile<i32>, %32: tile<i32>, %33: tile<i32>, %34: tile<i32>, %35: tile<i32>) {
    %36 = constant <i32: 16> : tile<i32>
    %37 = constant <i32: 4> : tile<i32>
    %38 = constant <i32: 64> : tile<i32>
    %39 = constant <i32: 32> : tile<i32>
    %40 = assume bounded<0, ?>, %1 : tile<i32>
    %41 = assume div_by<16>, %40 : tile<i32>
    %42 = assume bounded<0, ?>, %2 : tile<i32>
    %43 = assume div_by<16>, %42 : tile<i32>
    %44 = make_token : token
    %45 = assume div_by<16>, %0 : tile<ptr<f32>>
    %46 = make_tensor_view %45, shape = [%41, %43], strides = [64, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[64,1]>
    %47 = assume bounded<0, ?>, %10 : tile<i32>
    %48 = assume div_by<4>, %47 : tile<i32>
    %49 = make_token : token
    %50 = assume div_by<16>, %9 : tile<ptr<f32>>
    %51 = make_tensor_view %50, shape = [%48, 16, 32], strides = [512, 32, 1] : tile<i32> -> tensor_view<?x16x32xf32, strides=[512,32,1]>
    %52 = assume bounded<0, ?>, %17 : tile<i32>
    %53 = assume div_by<4>, %52 : tile<i32>
    %54 = make_token : token
    %55 = assume div_by<16>, %16 : tile<ptr<f32>>
    %56 = make_tensor_view %55, shape = [%53, 16, 1], strides = [16, 1, 1] : tile<i32> -> tensor_view<?x16x1xf32, strides=[16,1,1]>
    %57 = assume bounded<0, ?>, %24 : tile<i32>
    %58 = assume div_by<4>, %57 : tile<i32>
    %59 = make_token : token
    %60 = assume div_by<16>, %23 : tile<ptr<f32>>
    %61 = make_tensor_view %60, shape = [%58, 16, 1], strides = [16, 1, 1] : tile<i32> -> tensor_view<?x16x1xf32, strides=[16,1,1]>
    %62 = assume bounded<0, ?>, %31 : tile<i32>
    %63 = assume div_by<16>, %62 : tile<i32>
    %64 = make_token : token
    %65 = assume div_by<16>, %30 : tile<ptr<f32>>
    %66 = make_tensor_view %65, shape = [%63, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf32, strides=[64,1]>
    %67 = constant <i32: 16> : tile<i32>
    %68 = constant <i32: 4> : tile<i32>
    %69 = constant <i32: 64> : tile<i32>
    %70 = constant <i32: 32> : tile<i32>
    %71, %72, %73 = get_tile_block_id : tile<i32>
    %74 = assume bounded<0, ?>, %71 : tile<i32>
    %75 = assume bounded<0, ?>, %72 : tile<i32>
    %76 = assume bounded<0, ?>, %73 : tile<i32>
    %77 = constant <f32: -1000000015047466200000000000000.0> : tile<4x1xf32>
    %78 = constant <f32: 0.0> : tile<4x1xf32>
    %79 = constant <f32: 0.0> : tile<4x32xf32>
    %80 = constant <i32: 0> : tile<i32>
    %81 = constant <i32: 1> : tile<i32>
    %183, %184, %185 = for %82 in (%80 to %35, step %81) : tile<i32> iter_values(%83 = %79, %84 = %78, %85 = %77) -> (tile<4x32xf32>, tile<4x1xf32>, tile<4x1xf32>) {
      %86 = assume bounded<0, ?>, %82 : tile<i32>
      %87 = constant <i32: 0> : tile<i32>
      %88 = constant <i32: 1> : tile<i32>
      %89 = constant <i32: 4> : tile<i32>
      %90 = constant <i32: 32> : tile<i32>
      %91 = constant <i32: -1> : tile<i32>
      %92 = constant <i32: 16> : tile<i32>
      %93 = constant <i32: 32> : tile<i32>
      %94 = constant <i32: 1> : tile<i32>
      %95 = constant <i32: 4> : tile<i32>
      %96 = constant <i32: 32> : tile<i32>
      %97 = constant <i32: -1> : tile<i32>
      %98 = constant <i32: 16> : tile<i32>
      %99 = constant <i32: 32> : tile<i32>
      %100 = constant <i32: -1> : tile<i32>
      %101 = constant <i32: 16> : tile<i32>
      %102 = constant <i32: 32> : tile<i32>
      %103 = make_partition_view %51 : partition_view<tile=(1x4x32), padding_value = zero, tensor_view<?x16x32xf32, strides=[512,32,1]>>
      %104, %105 = load_view_tko weak %103[%86, %74, %87] token = %49 : partition_view<tile=(1x4x32), padding_value = zero, tensor_view<?x16x32xf32, strides=[512,32,1]>>, tile<i32> -> tile<1x4x32xf32>, token
      %106 = constant <i32: 0> : tile<i32>
      %107 = constant <i32: 1> : tile<i32>
      %108 = constant <i32: 4> : tile<i32>
      %109 = constant <i32: 1> : tile<i32>
      %110 = constant <i32: -1> : tile<i32>
      %111 = constant <i32: 16> : tile<i32>
      %112 = constant <i32: 1> : tile<i32>
      %113 = constant <i32: 1> : tile<i32>
      %114 = constant <i32: 4> : tile<i32>
      %115 = constant <i32: 1> : tile<i32>
      %116 = constant <i32: -1> : tile<i32>
      %117 = constant <i32: 16> : tile<i32>
      %118 = constant <i32: 1> : tile<i32>
      %119 = constant <i32: -1> : tile<i32>
      %120 = constant <i32: 16> : tile<i32>
      %121 = constant <i32: 1> : tile<i32>
      %122 = make_partition_view %56 : partition_view<tile=(1x4x1), padding_value = zero, tensor_view<?x16x1xf32, strides=[16,1,1]>>
      %123, %124 = load_view_tko weak %122[%86, %74, %106] token = %54 : partition_view<tile=(1x4x1), padding_value = zero, tensor_view<?x16x1xf32, strides=[16,1,1]>>, tile<i32> -> tile<1x4x1xf32>, token
      %125 = constant <i32: 0> : tile<i32>
      %126 = constant <i32: 1> : tile<i32>
      %127 = constant <i32: 4> : tile<i32>
      %128 = constant <i32: 1> : tile<i32>
      %129 = constant <i32: -1> : tile<i32>
      %130 = constant <i32: 16> : tile<i32>
      %131 = constant <i32: 1> : tile<i32>
      %132 = constant <i32: 1> : tile<i32>
      %133 = constant <i32: 4> : tile<i32>
      %134 = constant <i32: 1> : tile<i32>
      %135 = constant <i32: -1> : tile<i32>
      %136 = constant <i32: 16> : tile<i32>
      %137 = constant <i32: 1> : tile<i32>
      %138 = constant <i32: -1> : tile<i32>
      %139 = constant <i32: 16> : tile<i32>
      %140 = constant <i32: 1> : tile<i32>
      %141 = make_partition_view %61 : partition_view<tile=(1x4x1), padding_value = zero, tensor_view<?x16x1xf32, strides=[16,1,1]>>
      %142, %143 = load_view_tko weak %141[%86, %74, %125] token = %59 : partition_view<tile=(1x4x1), padding_value = zero, tensor_view<?x16x1xf32, strides=[16,1,1]>>, tile<i32> -> tile<1x4x1xf32>, token
      %144 = constant <i32: 1> : tile<i32>
      %145 = constant <i32: 4> : tile<i32>
      %146 = constant <i32: 1> : tile<i32>
      %147 = constant <i32: 4> : tile<i32>
      %148 = constant <i32: 1> : tile<i32>
      %149 = reshape %123 : tile<1x4x1xf32> -> tile<4x1xf32>
      %150 = maxf %85, %149 {rounding_mode = 0} : tile<4x1xf32>
      %151 = subf %85, %150 : tile<4x1xf32>
      %152 = exp %151 : tile<4x1xf32>
      %153 = subf %149, %150 : tile<4x1xf32>
      %154 = exp %153 : tile<4x1xf32>
      %155 = mulf %84, %152 : tile<4x1xf32>
      %156 = constant <i32: 1> : tile<i32>
      %157 = constant <i32: 4> : tile<i32>
      %158 = constant <i32: 1> : tile<i32>
      %159 = constant <i32: 4> : tile<i32>
      %160 = constant <i32: 1> : tile<i32>
      %161 = reshape %142 : tile<1x4x1xf32> -> tile<4x1xf32>
      %162 = mulf %161, %154 : tile<4x1xf32>
      %163 = addf %155, %162 : tile<4x1xf32>
      %164 = constant <i32: 4> : tile<i32>
      %165 = constant <i32: 1> : tile<i32>
      %166 = constant <i32: 4> : tile<i32>
      %167 = constant <i32: 32> : tile<i32>
      %168 = broadcast %152 : tile<4x1xf32> -> tile<4x32xf32>
      %169 = mulf %83, %168 : tile<4x32xf32>
      %170 = constant <i32: 1> : tile<i32>
      %171 = constant <i32: 4> : tile<i32>
      %172 = constant <i32: 32> : tile<i32>
      %173 = constant <i32: 4> : tile<i32>
      %174 = constant <i32: 32> : tile<i32>
      %175 = reshape %104 : tile<1x4x32xf32> -> tile<4x32xf32>
      %176 = constant <i32: 4> : tile<i32>
      %177 = constant <i32: 1> : tile<i32>
      %178 = constant <i32: 4> : tile<i32>
      %179 = constant <i32: 32> : tile<i32>
      %180 = broadcast %154 : tile<4x1xf32> -> tile<4x32xf32>
      %181 = mulf %175, %180 : tile<4x32xf32>
      %182 = addf %169, %181 : tile<4x32xf32>
      continue %182, %163, %150 : tile<4x32xf32>, tile<4x1xf32>, tile<4x1xf32>
    }
    %186 = constant <i32: 4> : tile<i32>
    %187 = constant <i32: 1> : tile<i32>
    %188 = constant <i32: 4> : tile<i32>
    %189 = constant <i32: 32> : tile<i32>
    %190 = broadcast %184 : tile<4x1xf32> -> tile<4x32xf32>
    %191 = divf %183, %190 rounding<approx> flush_to_zero : tile<4x32xf32>
    %192 = constant <i32: 0> : tile<i32>
    %193 = constant <i32: 32> : tile<i32>
    %194 = constant <i32: 64> : tile<i32>
    %195 = constant <i32: -1> : tile<i32>
    %196 = constant <i32: 64> : tile<i32>
    %197 = constant <i32: 32> : tile<i32>
    %198 = constant <i32: 64> : tile<i32>
    %199 = constant <i32: -1> : tile<i32>
    %200 = constant <i32: 64> : tile<i32>
    %201 = constant <i32: -1> : tile<i32>
    %202 = constant <i32: 64> : tile<i32>
    %203 = make_partition_view %66 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>
    %204, %205 = load_view_tko weak %203[%74, %192] token = %64 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>, tile<i32> -> tile<32x64xf32>, token
    %206 = constant <i32: 4> : tile<i32>
    %207 = constant <i32: 32> : tile<i32>
    %208 = constant <i32: 4> : tile<i32>
    %209 = constant <i32: 32> : tile<i32>
    %210 = constant <i32: 1> : tile<i32>
    %211 = reshape %191 : tile<4x32xf32> -> tile<4x32x1xf32>
    %212 = constant <i32: 4> : tile<i32>
    %213 = constant <i32: 32> : tile<i32>
    %214 = constant <i32: 1> : tile<i32>
    %215 = constant <i32: 4> : tile<i32>
    %216 = constant <i32: 32> : tile<i32>
    %217 = constant <i32: 64> : tile<i32>
    %218 = broadcast %211 : tile<4x32x1xf32> -> tile<4x32x64xf32>
    %219 = constant <i32: 32> : tile<i32>
    %220 = constant <i32: 64> : tile<i32>
    %221 = constant <i32: 1> : tile<i32>
    %222 = constant <i32: 32> : tile<i32>
    %223 = constant <i32: 64> : tile<i32>
    %224 = reshape %204 : tile<32x64xf32> -> tile<1x32x64xf32>
    %225 = constant <i32: 1> : tile<i32>
    %226 = constant <i32: 32> : tile<i32>
    %227 = constant <i32: 64> : tile<i32>
    %228 = constant <i32: 4> : tile<i32>
    %229 = constant <i32: 32> : tile<i32>
    %230 = constant <i32: 64> : tile<i32>
    %231 = broadcast %224 : tile<1x32x64xf32> -> tile<4x32x64xf32>
    %232 = mulf %218, %231 : tile<4x32x64xf32>
    %236 = reduce %232 dim=1 identities=[0] : tile<4x32x64xf32> -> tile<4x64xf32> {
    ^bb0(%233: tile<f32>, %234: tile<f32>):
      %235 = addf %233, %234 : tile<f32>
      yield %235 : tile<f32>
    }
    %237 = constant <i32: 4> : tile<i32>
    %238 = constant <i32: 64> : tile<i32>
    %239 = constant <i32: 4> : tile<i32>
    %240 = constant <i32: 64> : tile<i32>
    %241, %242, %243 = get_tile_block_id : tile<i32>
    %244 = assume bounded<0, ?>, %241 : tile<i32>
    %245 = assume bounded<0, ?>, %242 : tile<i32>
    %246 = assume bounded<0, ?>, %243 : tile<i32>
    %247 = make_partition_view %46 : partition_view<tile=(4x64), tensor_view<?x?xf32, strides=[64,1]>>
    %248 = store_view_tko weak %236, %247[%244, %245] token = %44 : tile<4x64xf32>, partition_view<tile=(4x64), tensor_view<?x?xf32, strides=[64,1]>>, tile<i32> -> token
    return
  }
}
