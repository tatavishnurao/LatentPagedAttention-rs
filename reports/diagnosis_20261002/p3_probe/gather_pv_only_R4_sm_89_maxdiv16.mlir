cuda_tile.module @gather_mma_module {
  entry @latent_gather_pv_only_entry(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f16>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<ptr<i32>>, %20: tile<i32>, %21: tile<i32>) {
    %22 = constant <i32: 4> : tile<i32>
    %23 = assume bounded<0, ?>, %1 : tile<i32>
    %24 = assume div_by<16>, %23 : tile<i32>
    %25 = assume bounded<0, ?>, %2 : tile<i32>
    %26 = assume div_by<16>, %25 : tile<i32>
    %27 = make_token : token
    %28 = assume div_by<16>, %0 : tile<ptr<f32>>
    %29 = make_tensor_view %28, shape = [%24, %26], strides = [32, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[32,1]>
    %30 = assume bounded<0, ?>, %11 : tile<i32>
    %31 = assume div_by<16>, %30 : tile<i32>
    %32 = make_token : token
    %33 = assume div_by<16>, %9 : tile<ptr<f16>>
    %34 = make_tensor_view %33, shape = [16, %31], strides = [8192, 1] : tile<i32> -> tensor_view<16x?xf16, strides=[8192,1]>
    %35 = assume bounded<0, ?>, %15 : tile<i32>
    %36 = assume div_by<16>, %35 : tile<i32>
    %37 = make_token : token
    %38 = assume div_by<16>, %14 : tile<ptr<f16>>
    %39 = make_tensor_view %38, shape = [%36, 32], strides = [32, 1] : tile<i32> -> tensor_view<?x32xf16, strides=[32,1]>
    %40 = assume bounded<0, ?>, %20 : tile<i32>
    %41 = assume div_by<16>, %40 : tile<i32>
    %42 = make_token : token
    %43 = assume div_by<16>, %19 : tile<ptr<i32>>
    %44 = make_tensor_view %43, shape = [%41], strides = [1] : tile<i32> -> tensor_view<?xi32, strides=[1]>
    %45 = constant <i32: 4> : tile<i32>
    %46, %47, %48 = get_tile_block_id : tile<i32>
    %49 = assume bounded<0, ?>, %46 : tile<i32>
    %50 = assume bounded<0, ?>, %47 : tile<i32>
    %51 = assume bounded<0, ?>, %48 : tile<i32>
    %52 = constant <f32: 0.0> : tile<16x32xf32>
    %53 = constant <i32: 0> : tile<i32>
    %54 = constant <i32: 1> : tile<i32>
    %99 = for %55 in (%53 to %45, step %54) : tile<i32> iter_values(%56 = %52) -> (tile<16x32xf32>) {
      %57 = assume bounded<0, 3>, %55 : tile<i32>
      %58 = muli %49, %45 : tile<i32>
      %59 = addi %58, %57 : tile<i32>
      %60 = constant <i32: 1> : tile<i32>
      %61 = constant <i32: -1> : tile<i32>
      %62 = constant <i32: 1> : tile<i32>
      %63 = constant <i32: -1> : tile<i32>
      %64 = constant <i32: -1> : tile<i32>
      %65 = make_partition_view %44 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      %66, %67 = load_view_tko weak %65[%59] token = %42 : partition_view<tile=(1), padding_value = zero, tensor_view<?xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %68 = constant <i32: 1> : tile<i32>
      %69 = reshape %66 : tile<1xi32> -> tile<i32>
      %70 = constant <i32: 0> : tile<i32>
      %71 = constant <i32: 16> : tile<i32>
      %72 = constant <i32: 32> : tile<i32>
      %73 = constant <i32: -1> : tile<i32>
      %74 = constant <i32: 32> : tile<i32>
      %75 = constant <i32: 16> : tile<i32>
      %76 = constant <i32: 32> : tile<i32>
      %77 = constant <i32: -1> : tile<i32>
      %78 = constant <i32: 32> : tile<i32>
      %79 = constant <i32: -1> : tile<i32>
      %80 = constant <i32: 32> : tile<i32>
      %81 = make_partition_view %39 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>
      %82, %83 = load_view_tko weak %81[%69, %70] token = %37 : partition_view<tile=(16x32), padding_value = zero, tensor_view<?x32xf16, strides=[32,1]>>, tile<i32> -> tile<16x32xf16>, token
      %84 = constant <i32: 0> : tile<i32>
      %85 = constant <i32: 16> : tile<i32>
      %86 = constant <i32: 16> : tile<i32>
      %87 = constant <i32: 16> : tile<i32>
      %88 = constant <i32: -1> : tile<i32>
      %89 = constant <i32: 16> : tile<i32>
      %90 = constant <i32: 16> : tile<i32>
      %91 = constant <i32: 16> : tile<i32>
      %92 = constant <i32: -1> : tile<i32>
      %93 = constant <i32: 16> : tile<i32>
      %94 = constant <i32: -1> : tile<i32>
      %95 = make_partition_view %34 : partition_view<tile=(16x16), padding_value = zero, tensor_view<16x?xf16, strides=[8192,1]>>
      %96, %97 = load_view_tko weak %95[%84, %59] token = %32 : partition_view<tile=(16x16), padding_value = zero, tensor_view<16x?xf16, strides=[8192,1]>>, tile<i32> -> tile<16x16xf16>, token
      %98 = mmaf %96, %82, %56 : tile<16x16xf16>, tile<16x32xf16>, tile<16x32xf32>
      continue %98 : tile<16x32xf32>
    }
    %100 = constant <i32: 16> : tile<i32>
    %101 = constant <i32: 32> : tile<i32>
    %102 = constant <i32: 16> : tile<i32>
    %103 = constant <i32: 32> : tile<i32>
    %104, %105, %106 = get_tile_block_id : tile<i32>
    %107 = assume bounded<0, ?>, %104 : tile<i32>
    %108 = assume bounded<0, ?>, %105 : tile<i32>
    %109 = assume bounded<0, ?>, %106 : tile<i32>
    %110 = make_partition_view %29 : partition_view<tile=(16x32), tensor_view<?x?xf32, strides=[32,1]>>
    %111 = store_view_tko weak %99, %110[%107, %108] token = %27 : tile<16x32xf32>, partition_view<tile=(16x32), tensor_view<?x?xf32, strides=[32,1]>>, tile<i32> -> token
    return
  }
}
