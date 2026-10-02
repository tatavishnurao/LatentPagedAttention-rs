cuda_tile.module @paged_source_hint {
  entry @model_small_scores_fp16_storage_rtable_1024_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>, %22: tile<ptr<i32>>, %23: tile<i32>, %24: tile<i32>, %25: tile<ptr<f32>>, %26: tile<i32>, %27: tile<i32>, %28: tile<i32>, %29: tile<i32>) {
    %30 = assume bounded<0, ?>, %1 : tile<i32>
    %31 = assume div_by<16>, %30 : tile<i32>
    %32 = assume bounded<0, ?>, %2 : tile<i32>
    %33 = assume div_by<16>, %32 : tile<i32>
    %34 = make_token : token
    %35 = assume div_by<16>, %0 : tile<ptr<f32>>
    %36 = make_tensor_view %35, shape = [%31, %33], strides = [1024, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[1024,1]>
    %37 = assume bounded<0, ?>, %10 : tile<i32>
    %38 = assume div_by<16>, %37 : tile<i32>
    %39 = make_token : token
    %40 = assume div_by<16>, %9 : tile<ptr<f32>>
    %41 = make_tensor_view %40, shape = [%38, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf32, strides=[64,1]>
    %42 = assume bounded<0, ?>, %15 : tile<i32>
    %43 = assume div_by<16>, %42 : tile<i32>
    %44 = make_token : token
    %45 = assume div_by<16>, %14 : tile<ptr<f16>>
    %46 = make_tensor_view %45, shape = [%43, 32], strides = [32, 1] : tile<i32> -> tensor_view<?x32xf16, strides=[32,1]>
    %47 = make_token : token
    %48 = assume div_by<16>, %19 : tile<ptr<i32>>
    %49 = make_tensor_view %48, shape = [64], strides = [1] : tensor_view<64xi32, strides=[1]>
    %50 = make_token : token
    %51 = assume div_by<16>, %22 : tile<ptr<i32>>
    %52 = make_tensor_view %51, shape = [1], strides = [1] : tensor_view<1xi32, strides=[1]>
    %53 = assume bounded<0, ?>, %26 : tile<i32>
    %54 = assume div_by<16>, %53 : tile<i32>
    %55 = make_token : token
    %56 = assume div_by<16>, %25 : tile<ptr<f32>>
    %57 = make_tensor_view %56, shape = [%54, 64], strides = [64, 1] : tile<i32> -> tensor_view<?x64xf32, strides=[64,1]>
    %58, %59, %60 = get_tile_block_id : tile<i32>
    %61 = assume bounded<0, ?>, %58 : tile<i32>
    %62 = assume bounded<0, ?>, %59 : tile<i32>
    %63 = assume bounded<0, ?>, %60 : tile<i32>
    %64 = constant <i32: 4> : tile<i32>
    %65 = divi %61, %64 signed rounding negative_inf : tile<i32>
    %66 = constant <i32: 0> : tile<i32>
    %67 = constant <i32: 1> : tile<i32>
    %68 = constant <i32: 1> : tile<i32>
    %69 = constant <i32: 1> : tile<i32>
    %70 = constant <i32: 1> : tile<i32>
    %71 = constant <i32: 1> : tile<i32>
    %72 = make_partition_view %52 : partition_view<tile=(1), padding_value = zero, tensor_view<1xi32, strides=[1]>>
    %73, %74 = load_view_tko weak %72[%66] token = %50 : partition_view<tile=(1), padding_value = zero, tensor_view<1xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
    %75 = constant <i32: 1> : tile<i32>
    %76 = constant <i32: 64> : tile<i32>
    %77 = constant <i32: 1> : tile<i32>
    %78 = constant <i32: 64> : tile<i32>
    %79 = constant <i32: 64> : tile<i32>
    %80 = make_partition_view %49 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>
    %81, %82 = load_view_tko weak %80[%62] token = %47 : partition_view<tile=(1), padding_value = zero, tensor_view<64xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
    %83 = constant <i32: 1> : tile<i32>
    %84 = reshape %81 : tile<1xi32> -> tile<i32>
    %85 = constant <i32: 0> : tile<i32>
    %86 = constant <i32: 1> : tile<i32>
    %87 = constant <i32: 64> : tile<i32>
    %88 = constant <i32: -1> : tile<i32>
    %89 = constant <i32: 64> : tile<i32>
    %90 = constant <i32: 1> : tile<i32>
    %91 = constant <i32: 64> : tile<i32>
    %92 = constant <i32: -1> : tile<i32>
    %93 = constant <i32: 64> : tile<i32>
    %94 = constant <i32: -1> : tile<i32>
    %95 = constant <i32: 64> : tile<i32>
    %96 = make_partition_view %41 : partition_view<tile=(1x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>
    %97, %98 = load_view_tko weak %96[%61, %85] token = %39 : partition_view<tile=(1x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>, tile<i32> -> tile<1x64xf32>, token
    %99 = constant <i32: 0> : tile<i32>
    %100 = constant <i32: 32> : tile<i32>
    %101 = constant <i32: 64> : tile<i32>
    %102 = constant <i32: -1> : tile<i32>
    %103 = constant <i32: 64> : tile<i32>
    %104 = constant <i32: 32> : tile<i32>
    %105 = constant <i32: 64> : tile<i32>
    %106 = constant <i32: -1> : tile<i32>
    %107 = constant <i32: 64> : tile<i32>
    %108 = constant <i32: -1> : tile<i32>
    %109 = constant <i32: 64> : tile<i32>
    %110 = make_partition_view %57 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>
    %111, %112 = load_view_tko weak %110[%65, %99] token = %55 : partition_view<tile=(32x64), padding_value = zero, tensor_view<?x64xf32, strides=[64,1]>>, tile<i32> -> tile<32x64xf32>, token
    %113 = constant <i32: 1> : tile<i32>
    %114 = constant <i32: 64> : tile<i32>
    %115 = constant <i32: 32> : tile<i32>
    %116 = constant <i32: 64> : tile<i32>
    %117 = broadcast %97 : tile<1x64xf32> -> tile<32x64xf32>
    %118 = mulf %111, %117 : tile<32x64xf32>
    %122 = reduce %118 dim=1 identities=[0] : tile<32x64xf32> -> tile<32xf32> {
    ^bb0(%119: tile<f32>, %120: tile<f32>):
      %121 = addf %119, %120 : tile<f32>
      yield %121 : tile<f32>
    }
    %123 = constant <i32: 0> : tile<i32>
    %124 = constant <i32: 16> : tile<i32>
    %125 = constant <i32: 32> : tile<i32>
    %126 = constant <i32: -1> : tile<i32>
    %127 = constant <i32: 32> : tile<i32>
    %128 = constant <i32: 16> : tile<i32>
    %129 = constant <i32: 32> : tile<i32>
    %130 = constant <i32: -1> : tile<i32>
    %131 = constant <i32: 32> : tile<i32>
    %132 = constant <i32: -1> : tile<i32>
    %133 = constant <i32: 32> : tile<i32>
    %134 = make_partition_view %46 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
    %135, %136 = load_view_tko weak %134[%84, %123] token = %44 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
    %137 = ftof %135 : tile<16x32xf16> -> tile<16x32xf32>
    %138 = constant <i32: 32> : tile<i32>
    %139 = constant <i32: 1> : tile<i32>
    %140 = constant <i32: 32> : tile<i32>
    %141 = reshape %122 : tile<32xf32> -> tile<1x32xf32>
    %142 = constant <i32: 1> : tile<i32>
    %143 = constant <i32: 32> : tile<i32>
    %144 = constant <i32: 16> : tile<i32>
    %145 = constant <i32: 32> : tile<i32>
    %146 = broadcast %141 : tile<1x32xf32> -> tile<16x32xf32>
    %147 = mulf %137, %146 : tile<16x32xf32>
    %151 = reduce %147 dim=1 identities=[0] : tile<16x32xf32> -> tile<16xf32> {
    ^bb0(%148: tile<f32>, %149: tile<f32>):
      %150 = addf %148, %149 : tile<f32>
      yield %150 : tile<f32>
    }
    %152 = constant <f32: 0.125> : tile<f32>
    %153 = constant <i32: 16> : tile<i32>
    %154 = constant <i32: 1> : tile<i32>
    %155 = constant <i32: 1> : tile<i32>
    %156 = reshape %152 : tile<f32> -> tile<1xf32>
    %157 = constant <i32: 1> : tile<i32>
    %158 = constant <i32: 16> : tile<i32>
    %159 = broadcast %156 : tile<1xf32> -> tile<16xf32>
    %160 = mulf %151, %159 : tile<16xf32>
    %161 = iota : tile<16xi32>
    %162 = constant <i32: 16> : tile<i32>
    %163 = muli %62, %162 : tile<i32>
    %164 = constant <i32: 16> : tile<i32>
    %165 = constant <i32: 1> : tile<i32>
    %166 = constant <i32: 1> : tile<i32>
    %167 = reshape %163 : tile<i32> -> tile<1xi32>
    %168 = constant <i32: 1> : tile<i32>
    %169 = constant <i32: 16> : tile<i32>
    %170 = broadcast %167 : tile<1xi32> -> tile<16xi32>
    %171 = addi %161, %170 : tile<16xi32>
    %172 = constant <i32: 1> : tile<i32>
    %173 = constant <i32: 16> : tile<i32>
    %174 = broadcast %73 : tile<1xi32> -> tile<16xi32>
    %175 = cmpi less_than %171, %174, signed : tile<16xi32> -> tile<16xi1>
    %176 = constant <f32: -340282346638528860000000000000000000000.0> : tile<f32>
    %177 = constant <i32: 16> : tile<i32>
    %178 = constant <i32: 1> : tile<i32>
    %179 = constant <i32: 1> : tile<i32>
    %180 = reshape %176 : tile<f32> -> tile<1xf32>
    %181 = constant <i32: 1> : tile<i32>
    %182 = constant <i32: 16> : tile<i32>
    %183 = broadcast %180 : tile<1xf32> -> tile<16xf32>
    %184 = select %175, %160, %183 : tile<16xi1>, tile<16xf32>
    %185 = constant <i32: 16> : tile<i32>
    %186 = constant <i32: 1> : tile<i32>
    %187 = constant <i32: 16> : tile<i32>
    %188 = reshape %184 : tile<16xf32> -> tile<1x16xf32>
    %189 = constant <i32: 1> : tile<i32>
    %190 = constant <i32: 16> : tile<i32>
    %191 = constant <i32: 1> : tile<i32>
    %192 = constant <i32: 16> : tile<i32>
    %193, %194, %195 = get_tile_block_id : tile<i32>
    %196 = assume bounded<0, ?>, %193 : tile<i32>
    %197 = assume bounded<0, ?>, %194 : tile<i32>
    %198 = assume bounded<0, ?>, %195 : tile<i32>
    %199 = make_partition_view %36 : partition_view<tile=(1x16), tensor_view<?x?xf32, strides=[1024,1]>>
    %200 = store_view_tko weak %188, %199[%196, %197] token = %34 : tile<1x16xf32>, partition_view<tile=(1x16), tensor_view<?x?xf32, strides=[1024,1]>>, tile<i32> -> token
    return
  }
}
