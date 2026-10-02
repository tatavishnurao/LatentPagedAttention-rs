cuda_tile.module @e3_kernels {
  entry @a3_split_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<i32>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<ptr<f32>>, %14: tile<i32>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<i32>, %20: tile<i32>, %21: tile<i32>, %22: tile<i32>, %23: tile<i32>, %24: tile<i32>, %25: tile<i32>, %26: tile<ptr<f32>>, %27: tile<i32>, %28: tile<i32>, %29: tile<i32>, %30: tile<i32>, %31: tile<i32>, %32: tile<i32>, %33: tile<i32>, %34: tile<i32>, %35: tile<i32>, %36: tile<i32>, %37: tile<i32>, %38: tile<i32>, %39: tile<ptr<f32>>, %40: tile<i32>, %41: tile<i32>, %42: tile<i32>, %43: tile<i32>, %44: tile<ptr<f16>>, %45: tile<i32>, %46: tile<i32>, %47: tile<i32>, %48: tile<i32>, %49: tile<ptr<f16>>, %50: tile<i32>, %51: tile<i32>, %52: tile<i32>, %53: tile<i32>, %54: tile<ptr<i32>>, %55: tile<i32>, %56: tile<i32>, %57: tile<i32>, %58: tile<i32>, %59: tile<f32>) {
    %60 = constant <i32: 16> : tile<i32>
    %61 = constant <i32: 4> : tile<i32>
    %62 = constant <i32: 4> : tile<i32>
    %63 = constant <i32: 64> : tile<i32>
    %64 = constant <i32: 16> : tile<i32>
    %65 = constant <i32: 32> : tile<i32>
    %66 = assume bounded<0, ?>, %1 : tile<i32>
    %67 = assume div_by<4>, %66 : tile<i32>
    %68 = assume bounded<0, ?>, %2 : tile<i32>
    %69 = assume div_by<16>, %68 : tile<i32>
    %70 = assume bounded<0, ?>, %3 : tile<i32>
    %71 = assume div_by<16>, %70 : tile<i32>
    %72 = make_token : token
    %73 = assume div_by<16>, %0 : tile<ptr<f32>>
    %74 = make_tensor_view %73, shape = [%67, %69, %71], strides = [1024, 64, 1] : tile<i32> -> tensor_view<?x?x?xf32, strides=[1024,64,1]>
    %75 = assume bounded<0, ?>, %14 : tile<i32>
    %76 = assume div_by<4>, %75 : tile<i32>
    %77 = assume bounded<0, ?>, %15 : tile<i32>
    %78 = assume div_by<16>, %77 : tile<i32>
    %79 = assume bounded<0, ?>, %16 : tile<i32>
    %80 = make_token : token
    %81 = assume div_by<16>, %13 : tile<ptr<f32>>
    %82 = make_tensor_view %81, shape = [%76, %78, %79], strides = [16, 1, 1] : tile<i32> -> tensor_view<?x?x?xf32, strides=[16,1,1]>
    %83 = assume bounded<0, ?>, %27 : tile<i32>
    %84 = assume div_by<4>, %83 : tile<i32>
    %85 = assume bounded<0, ?>, %28 : tile<i32>
    %86 = assume div_by<16>, %85 : tile<i32>
    %87 = assume bounded<0, ?>, %29 : tile<i32>
    %88 = make_token : token
    %89 = assume div_by<16>, %26 : tile<ptr<f32>>
    %90 = make_tensor_view %89, shape = [%84, %86, %87], strides = [16, 1, 1] : tile<i32> -> tensor_view<?x?x?xf32, strides=[16,1,1]>
    %91 = make_token : token
    %92 = assume div_by<16>, %39 : tile<ptr<f32>>
    %93 = make_tensor_view %92, shape = [16, 64], strides = [64, 1] : tensor_view<16x64xf32, strides=[64,1]>
    %94 = assume bounded<0, ?>, %45 : tile<i32>
    %95 = assume div_by<16>, %94 : tile<i32>
    %96 = make_token : token
    %97 = assume div_by<16>, %44 : tile<ptr<f16>>
    %98 = make_tensor_view %97, shape = [%95, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf16, strides=[64,1]>
    %99 = assume bounded<0, ?>, %50 : tile<i32>
    %100 = assume div_by<16>, %99 : tile<i32>
    %101 = make_token : token
    %102 = assume div_by<16>, %49 : tile<ptr<f16>>
    %103 = make_tensor_view %102, shape = [%100, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf16, strides=[64,1]>
    %104 = assume bounded<0, ?>, %55 : tile<i32>
    %105 = assume div_by<16>, %104 : tile<i32>
    %106 = make_token : token
    %107 = assume div_by<16>, %54 : tile<ptr<i32>>
    %108 = make_tensor_view %107, shape = [%105], strides = [1] : tile<i32> -> tensor_view<?xi32, strides=[1]>
    %109 = constant <i32: 16> : tile<i32>
    %110 = constant <i32: 4> : tile<i32>
    %111 = constant <i32: 4> : tile<i32>
    %112 = constant <i32: 64> : tile<i32>
    %113 = constant <i32: 16> : tile<i32>
    %114 = constant <i32: 32> : tile<i32>
    %115, %116, %117 = get_tile_block_id : tile<i32>
    %118 = assume bounded<0, ?>, %115 : tile<i32>
    %119 = assume bounded<0, ?>, %116 : tile<i32>
    %120 = assume bounded<0, ?>, %117 : tile<i32>
    %121 = constant <i32: 0> : tile<i32>
    %122 = constant <i32: 4> : tile<i32>
    %123 = constant <i32: 64> : tile<i32>
    %124 = constant <i32: 16> : tile<i32>
    %125 = constant <i32: 64> : tile<i32>
    %126 = constant <i32: 4> : tile<i32>
    %127 = constant <i32: 64> : tile<i32>
    %128 = constant <i32: 16> : tile<i32>
    %129 = constant <i32: 64> : tile<i32>
    %130 = constant <i32: 16> : tile<i32>
    %131 = constant <i32: 64> : tile<i32>
    %132 = make_partition_view %93 : partition_view<tile=(4x64), padding_value = zero, tensor_view<16x64xf32, strides=[64,1]>>
    %133, %134 = load_view_tko weak %132[%119, %121] token = %91 : partition_view<tile=(4x64), padding_value = zero, tensor_view<16x64xf32, strides=[64,1]>>, tile<i32> -> tile<4x64xf32>, token
    %135 = ftof %133 : tile<4x64xf32> -> tile<4x64xf16>
    %136 = constant <f32: -1000000015047466200000000000000.0> : tile<4x1xf32>
    %137 = constant <f32: 0.0> : tile<4x1xf32>
    %138 = constant <f32: 0.0> : tile<4x64xf32>
    %139 = muli %118, %58 : tile<i32>
    %140 = constant <i32: 2> : tile<i32>
    %141 = muli %139, %140 : tile<i32>
    %142 = constant <i32: 0> : tile<i32>
    %143 = constant <i32: 1> : tile<i32>
    %344, %345, %346 = for %144 in (%142 to %58, step %143) : tile<i32> iter_values(%145 = %138, %146 = %137, %147 = %136) -> (tile<4x64xf32>, tile<4x1xf32>, tile<4x1xf32>) {
      %148 = assume bounded<0, ?>, %144 : tile<i32>
      %149 = constant <i32: 2> : tile<i32>
      %150 = muli %148, %149 : tile<i32>
      %151 = addi %141, %150 : tile<i32>
      %152 = constant <i32: 1> : tile<i32>
      %153 = constant <i32: -1> : tile<i32>
      %154 = constant <i32: 1> : tile<i32>
      %155 = constant <i32: -1> : tile<i32>
      %156 = constant <i32: -1> : tile<i32>
      %157 = make_partition_view %108 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %158, %159 = load_view_tko weak %157[%151] token = %106 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %160 = constant <i32: 1> : tile<i32>
      %161 = addi %151, %160 : tile<i32>
      %162 = constant <i32: 1> : tile<i32>
      %163 = constant <i32: -1> : tile<i32>
      %164 = constant <i32: 1> : tile<i32>
      %165 = constant <i32: -1> : tile<i32>
      %166 = constant <i32: -1> : tile<i32>
      %167 = make_partition_view %108 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %168, %169 = load_view_tko weak %167[%161] token = %106 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %170 = constant <i32: 1> : tile<i32>
      %171 = reshape %158 : tile<1xi32> -> tile<i32>
      %172 = muli %171, %110 : tile<i32>
      %173 = addi %172, %119 : tile<i32>
      %174 = constant <i32: 1> : tile<i32>
      %175 = reshape %168 : tile<1xi32> -> tile<i32>
      %176 = muli %175, %110 : tile<i32>
      %177 = addi %176, %119 : tile<i32>
      %178 = constant <i32: 0> : tile<i32>
      %179 = constant <i32: 16> : tile<i32>
      %180 = constant <i32: 64> : tile<i32>
      %181 = constant <i32: -1> : tile<i32>
      %182 = constant <i32: 64> : tile<i32>
      %183 = constant <i32: 16> : tile<i32>
      %184 = constant <i32: 64> : tile<i32>
      %185 = constant <i32: -1> : tile<i32>
      %186 = constant <i32: 64> : tile<i32>
      %187 = constant <i32: -1> : tile<i32>
      %188 = constant <i32: 64> : tile<i32>
      %189 = make_partition_view %98 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>
      %190, %191 = load_view_tko weak %189[%173, %178] token = %96 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>, tile<i32> -> tile<16x64xf16>, token
      %192 = constant <i32: 0> : tile<i32>
      %193 = constant <i32: 16> : tile<i32>
      %194 = constant <i32: 64> : tile<i32>
      %195 = constant <i32: -1> : tile<i32>
      %196 = constant <i32: 64> : tile<i32>
      %197 = constant <i32: 16> : tile<i32>
      %198 = constant <i32: 64> : tile<i32>
      %199 = constant <i32: -1> : tile<i32>
      %200 = constant <i32: 64> : tile<i32>
      %201 = constant <i32: -1> : tile<i32>
      %202 = constant <i32: 64> : tile<i32>
      %203 = make_partition_view %98 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>
      %204, %205 = load_view_tko weak %203[%177, %192] token = %96 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>, tile<i32> -> tile<16x64xf16>, token
      %206 = constant <i32: 0> : tile<i32>
      %207 = cat %190, %204 dim = 0 : tile<16x64xf16>, tile<16x64xf16> -> tile<32x64xf16>
      %208 = constant <i32: 32> : tile<i32>
      %209 = constant <i32: 64> : tile<i32>
      %210 = permute %207 [1, 0] : tile<32x64xf16> -> tile<64x32xf16>
      %211 = constant <f32: 0.0> : tile<4x32xf32>
      %212 = mmaf %135, %210, %211 : tile<4x64xf16>, tile<64x32xf16>, tile<4x32xf32>
      %213 = constant <i32: 4> : tile<i32>
      %214 = constant <i32: 32> : tile<i32>
      %215 = constant <i32: 1> : tile<i32>
      %216 = constant <i32: 1> : tile<i32>
      %217 = constant <i32: 1> : tile<i32>
      %218 = reshape %59 : tile<f32> -> tile<1x1xf32>
      %219 = constant <i32: 1> : tile<i32>
      %220 = constant <i32: 1> : tile<i32>
      %221 = constant <i32: 4> : tile<i32>
      %222 = constant <i32: 32> : tile<i32>
      %223 = broadcast %218 : tile<1x1xf32> -> tile<4x32xf32>
      %224 = mulf %212, %223 : tile<4x32xf32>
      %225 = iota : tile<32xi32>
      %226 = muli %151, %113 : tile<i32>
      %227 = constant <i32: 32> : tile<i32>
      %228 = constant <i32: 1> : tile<i32>
      %229 = constant <i32: 1> : tile<i32>
      %230 = reshape %226 : tile<i32> -> tile<1xi32>
      %231 = constant <i32: 1> : tile<i32>
      %232 = constant <i32: 32> : tile<i32>
      %233 = broadcast %230 : tile<1xi32> -> tile<32xi32>
      %234 = addi %225, %233 : tile<32xi32>
      %235 = constant <i32: 32> : tile<i32>
      %236 = constant <i32: 1> : tile<i32>
      %237 = constant <i32: 1> : tile<i32>
      %238 = reshape %57 : tile<i32> -> tile<1xi32>
      %239 = constant <i32: 1> : tile<i32>
      %240 = constant <i32: 32> : tile<i32>
      %241 = broadcast %238 : tile<1xi32> -> tile<32xi32>
      %242 = cmpi less_than %234, %241, signed : tile<32xi32> -> tile<32xi1>
      %243 = constant <i32: 32> : tile<i32>
      %244 = constant <i32: 1> : tile<i32>
      %245 = constant <i32: 32> : tile<i32>
      %246 = reshape %242 : tile<32xi1> -> tile<1x32xi1>
      %247 = constant <i32: 1> : tile<i32>
      %248 = constant <i32: 32> : tile<i32>
      %249 = constant <i32: 4> : tile<i32>
      %250 = constant <i32: 32> : tile<i32>
      %251 = broadcast %246 : tile<1x32xi1> -> tile<4x32xi1>
      %252 = constant <f32: -1000000015047466200000000000000.0> : tile<f32>
      %253 = constant <i32: 4> : tile<i32>
      %254 = constant <i32: 32> : tile<i32>
      %255 = constant <i32: 1> : tile<i32>
      %256 = constant <i32: 1> : tile<i32>
      %257 = constant <i32: 1> : tile<i32>
      %258 = reshape %252 : tile<f32> -> tile<1x1xf32>
      %259 = constant <i32: 1> : tile<i32>
      %260 = constant <i32: 1> : tile<i32>
      %261 = constant <i32: 4> : tile<i32>
      %262 = constant <i32: 32> : tile<i32>
      %263 = broadcast %258 : tile<1x1xf32> -> tile<4x32xf32>
      %264 = select %251, %224, %263 : tile<4x32xi1>, tile<4x32xf32>
      %268 = reduce %264 dim=1 identities=[-inf] : tile<4x32xf32> -> tile<4xf32> {
      ^bb0(%265: tile<f32>, %266: tile<f32>):
        %267 = maxf %265, %266 {rounding_mode = 0} : tile<f32>
        yield %267 : tile<f32>
      }
      %269 = constant <i32: 4> : tile<i32>
      %270 = constant <i32: 4> : tile<i32>
      %271 = constant <i32: 1> : tile<i32>
      %272 = reshape %268 : tile<4xf32> -> tile<4x1xf32>
      %273 = maxf %147, %272 {rounding_mode = 0} : tile<4x1xf32>
      %274 = constant <i32: 4> : tile<i32>
      %275 = constant <i32: 1> : tile<i32>
      %276 = constant <i32: 4> : tile<i32>
      %277 = constant <i32: 32> : tile<i32>
      %278 = broadcast %273 : tile<4x1xf32> -> tile<4x32xf32>
      %279 = subf %264, %278 : tile<4x32xf32>
      %280 = exp %279 : tile<4x32xf32>
      %281 = constant <f32: 0.0> : tile<f32>
      %282 = constant <i32: 4> : tile<i32>
      %283 = constant <i32: 32> : tile<i32>
      %284 = constant <i32: 1> : tile<i32>
      %285 = constant <i32: 1> : tile<i32>
      %286 = constant <i32: 1> : tile<i32>
      %287 = reshape %281 : tile<f32> -> tile<1x1xf32>
      %288 = constant <i32: 1> : tile<i32>
      %289 = constant <i32: 1> : tile<i32>
      %290 = constant <i32: 4> : tile<i32>
      %291 = constant <i32: 32> : tile<i32>
      %292 = broadcast %287 : tile<1x1xf32> -> tile<4x32xf32>
      %293 = select %251, %280, %292 : tile<4x32xi1>, tile<4x32xf32>
      %297 = reduce %293 dim=1 identities=[0] : tile<4x32xf32> -> tile<4xf32> {
      ^bb0(%294: tile<f32>, %295: tile<f32>):
        %296 = addf %294, %295 : tile<f32>
        yield %296 : tile<f32>
      }
      %298 = subf %147, %273 : tile<4x1xf32>
      %299 = exp %298 : tile<4x1xf32>
      %300 = mulf %146, %299 : tile<4x1xf32>
      %301 = constant <i32: 4> : tile<i32>
      %302 = constant <i32: 4> : tile<i32>
      %303 = constant <i32: 1> : tile<i32>
      %304 = reshape %297 : tile<4xf32> -> tile<4x1xf32>
      %305 = addf %300, %304 : tile<4x1xf32>
      %306 = constant <i32: 4> : tile<i32>
      %307 = constant <i32: 1> : tile<i32>
      %308 = constant <i32: 4> : tile<i32>
      %309 = constant <i32: 64> : tile<i32>
      %310 = broadcast %299 : tile<4x1xf32> -> tile<4x64xf32>
      %311 = mulf %145, %310 : tile<4x64xf32>
      %312 = constant <i32: 0> : tile<i32>
      %313 = constant <i32: 16> : tile<i32>
      %314 = constant <i32: 64> : tile<i32>
      %315 = constant <i32: -1> : tile<i32>
      %316 = constant <i32: 64> : tile<i32>
      %317 = constant <i32: 16> : tile<i32>
      %318 = constant <i32: 64> : tile<i32>
      %319 = constant <i32: -1> : tile<i32>
      %320 = constant <i32: 64> : tile<i32>
      %321 = constant <i32: -1> : tile<i32>
      %322 = constant <i32: 64> : tile<i32>
      %323 = make_partition_view %103 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>
      %324, %325 = load_view_tko weak %323[%173, %312] token = %101 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>, tile<i32> -> tile<16x64xf16>, token
      %326 = constant <i32: 0> : tile<i32>
      %327 = constant <i32: 16> : tile<i32>
      %328 = constant <i32: 64> : tile<i32>
      %329 = constant <i32: -1> : tile<i32>
      %330 = constant <i32: 64> : tile<i32>
      %331 = constant <i32: 16> : tile<i32>
      %332 = constant <i32: 64> : tile<i32>
      %333 = constant <i32: -1> : tile<i32>
      %334 = constant <i32: 64> : tile<i32>
      %335 = constant <i32: -1> : tile<i32>
      %336 = constant <i32: 64> : tile<i32>
      %337 = make_partition_view %103 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>
      %338, %339 = load_view_tko weak %337[%177, %326] token = %101 : partition_view<tile=(16x64), padding_value = zero, tensor_view<?x64xf16, strides=[64,1]>>, tile<i32> -> tile<16x64xf16>, token
      %340 = constant <i32: 0> : tile<i32>
      %341 = cat %324, %338 dim = 0 : tile<16x64xf16>, tile<16x64xf16> -> tile<32x64xf16>
      %342 = ftof %293 : tile<4x32xf32> -> tile<4x32xf16>
      %343 = mmaf %342, %341, %311 : tile<4x32xf16>, tile<32x64xf16>, tile<4x64xf32>
      continue %343, %305, %273 : tile<4x64xf32>, tile<4x1xf32>, tile<4x1xf32>
    }
    %347 = constant <i32: 4> : tile<i32>
    %348 = constant <i32: 64> : tile<i32>
    %349 = constant <i32: 1> : tile<i32>
    %350 = constant <i32: 4> : tile<i32>
    %351 = constant <i32: 64> : tile<i32>
    %352 = reshape %344 : tile<4x64xf32> -> tile<1x4x64xf32>
    %353 = constant <i32: 1> : tile<i32>
    %354 = constant <i32: 4> : tile<i32>
    %355 = constant <i32: 64> : tile<i32>
    %356 = constant <i32: 1> : tile<i32>
    %357 = constant <i32: 4> : tile<i32>
    %358 = constant <i32: 64> : tile<i32>
    %359, %360, %361 = get_tile_block_id : tile<i32>
    %362 = assume bounded<0, ?>, %359 : tile<i32>
    %363 = assume bounded<0, ?>, %360 : tile<i32>
    %364 = assume bounded<0, ?>, %361 : tile<i32>
    %365 = make_partition_view %74 : partition_view<tile=(1x4x64), tensor_view<?x?x?xf32, strides=[1024,64,1]>>
    %366 = store_view_tko weak %352, %365[%362, %363, %364] token = %72 : tile<1x4x64xf32>, partition_view<tile=(1x4x64), tensor_view<?x?x?xf32, strides=[1024,64,1]>>, tile<i32> -> token
    %367 = constant <i32: 4> : tile<i32>
    %368 = constant <i32: 1> : tile<i32>
    %369 = constant <i32: 1> : tile<i32>
    %370 = constant <i32: 4> : tile<i32>
    %371 = constant <i32: 1> : tile<i32>
    %372 = reshape %346 : tile<4x1xf32> -> tile<1x4x1xf32>
    %373 = constant <i32: 1> : tile<i32>
    %374 = constant <i32: 4> : tile<i32>
    %375 = constant <i32: 1> : tile<i32>
    %376 = constant <i32: 1> : tile<i32>
    %377 = constant <i32: 4> : tile<i32>
    %378 = constant <i32: 1> : tile<i32>
    %379, %380, %381 = get_tile_block_id : tile<i32>
    %382 = assume bounded<0, ?>, %379 : tile<i32>
    %383 = assume bounded<0, ?>, %380 : tile<i32>
    %384 = assume bounded<0, ?>, %381 : tile<i32>
    %385 = make_partition_view %82 : partition_view<tile=(1x4x1), tensor_view<?x?x?xf32, strides=[16,1,1]>>
    %386 = store_view_tko weak %372, %385[%382, %383, %384] token = %80 : tile<1x4x1xf32>, partition_view<tile=(1x4x1), tensor_view<?x?x?xf32, strides=[16,1,1]>>, tile<i32> -> token
    %387 = constant <i32: 4> : tile<i32>
    %388 = constant <i32: 1> : tile<i32>
    %389 = constant <i32: 1> : tile<i32>
    %390 = constant <i32: 4> : tile<i32>
    %391 = constant <i32: 1> : tile<i32>
    %392 = reshape %345 : tile<4x1xf32> -> tile<1x4x1xf32>
    %393 = constant <i32: 1> : tile<i32>
    %394 = constant <i32: 4> : tile<i32>
    %395 = constant <i32: 1> : tile<i32>
    %396 = constant <i32: 1> : tile<i32>
    %397 = constant <i32: 4> : tile<i32>
    %398 = constant <i32: 1> : tile<i32>
    %399, %400, %401 = get_tile_block_id : tile<i32>
    %402 = assume bounded<0, ?>, %399 : tile<i32>
    %403 = assume bounded<0, ?>, %400 : tile<i32>
    %404 = assume bounded<0, ?>, %401 : tile<i32>
    %405 = make_partition_view %90 : partition_view<tile=(1x4x1), tensor_view<?x?x?xf32, strides=[16,1,1]>>
    %406 = store_view_tko weak %392, %405[%402, %403, %404] token = %88 : tile<1x4x1xf32>, partition_view<tile=(1x4x1), tensor_view<?x?x?xf32, strides=[16,1,1]>>, tile<i32> -> token
    return
  }
}
