//! E3 split-K paged decode (docs/E3_SPLITK_DESIGN.md). Correctness-first.
//!
//! Each split CTA walks `steps_per_split` steps of two logical blocks (TOK = 2 * BLK tokens;
//! the score GEMM needs >= 512 outputs for HMMA, §0(b)), gathers the blocks through the block
//! table, runs an online softmax, and writes an unnormalized partial (acc, m, l). A reduce kernel
//! merges the splits with log-sum-exp and normalizes.
//!
//! - C3 (latent): one CTA covers all HQ query heads of the shared latent tile. The query
//!   projection q' = q P^K is computed once per CTA in FP32 and amortized over its blocks.
//!   The reduce kernel applies the V projection once per head.
//! - A3 (full KV): one CTA covers the G query heads of one KV head; same two-block steps.
//!
//! Precision: FP16 cache storage; `mma` operands are FP16 (q' or q, and P), accumulation FP32.
//! Masked scores use a finite sentinel (-1e30) so fully masked steps keep m finite and add
//! nothing (p is selected to 0, alpha = 1).
//!
//! Generic parameters (all powers of two): HQ query heads, HKV KV heads, G = HQ / HKV,
//! D head dim, L latent dim, BLK tokens per block, TOK = 2 * BLK, DC projection chunk of D,
//! HKVL = HKV * L.

// Kernel entries take many tensor and scalar arguments by design.
#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod e3_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn c3_split<
        const HQ: i32,
        const HKV: i32,
        const G: i32,
        const D: i32,
        const L: i32,
        const BLK: i32,
        const TOK: i32,
        const DC: i32,
        const HKVL: i32,
    >(
        out_acc: &mut Tensor<f32, { [1, HQ, L] }>,
        out_m: &mut Tensor<f32, { [1, HQ, 1] }>,
        out_l: &mut Tensor<f32, { [1, HQ, 1] }>,
        q: &Tensor<f32, { [HQ, D] }>,
        k_projection: &Tensor<f32, { [HKVL, D] }>,
        latent: &Tensor<f16, { [-1, L] }>,
        table: &Tensor<i32, { [-1] }>,
        active_seq_len: i32,
        steps_per_split: i32,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let split = pid.0;

        // q'[h, l] = sum_d q[h, d] * P^K_{h / G}[l, d], FP32, chunked over D.
        let mut qp: Tile<f32, { [HQ, L] }> = constant(0.0f32, const_shape![HQ, L]);
        let chunks: i32 = D / DC;
        for c in 0i32..chunks {
            let qc: Tile<f32, { [HQ, DC] }> = q.load_tile(const_shape![HQ, DC], [0, c]);
            let kc: Tile<f32, { [HKVL, DC] }> =
                k_projection.load_tile(const_shape![HKVL, DC], [0, c]);
            let qb: Tile<f32, { [HKV, G, L, DC] }> = qc
                .reshape(const_shape![HKV, G, 1, DC])
                .broadcast(const_shape![HKV, G, L, DC]);
            let kb: Tile<f32, { [HKV, G, L, DC] }> = kc
                .reshape(const_shape![HKV, 1, L, DC])
                .broadcast(const_shape![HKV, G, L, DC]);
            let part: Tile<f32, { [HKV, G, L] }> = reduce_sum(qb * kb, 3i32);
            let part2: Tile<f32, { [HQ, L] }> = part.reshape(const_shape![HQ, L]);
            qp = qp + part2;
        }
        let q16: Tile<f16, { [HQ, L] }> = convert_tile(qp);

        let mut m_i: Tile<f32, { [HQ, 1] }> = constant(-1.0e30f32, const_shape![HQ, 1]);
        let mut l_i: Tile<f32, { [HQ, 1] }> = constant(0.0f32, const_shape![HQ, 1]);
        let mut acc: Tile<f32, { [HQ, L] }> = constant(0.0f32, const_shape![HQ, L]);
        let base_block = split * steps_per_split * 2i32;
        for step in 0i32..steps_per_split {
            let b0 = base_block + step * 2i32;
            let e0: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0]);
            let e1: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 1i32]);
            let p0: i32 = tile_to_scalar(e0.reshape(const_shape![]));
            let p1: i32 = tile_to_scalar(e1.reshape(const_shape![]));
            let za: Tile<f16, { [BLK, L] }> = latent.load_tile(const_shape![BLK, L], [p0, 0]);
            let zb: Tile<f16, { [BLK, L] }> = latent.load_tile(const_shape![BLK, L], [p1, 0]);
            let z: Tile<f16, { [TOK, L] }> = cat(za, zb, 0i32);
            let zt: Tile<f16, { [L, TOK] }> = z.transpose();
            let zero: Tile<f32, { [HQ, TOK] }> = constant(0.0f32, const_shape![HQ, TOK]);
            let raw: Tile<f32, { [HQ, TOK] }> = mma(q16, zt, zero);
            let s = raw * broadcast_scalar(scale, const_shape![HQ, TOK]);

            let tokens: Tile<i32, { [TOK] }> =
                iota(const_shape![TOK]) + broadcast_scalar(b0 * BLK, const_shape![TOK]);
            let valid_1d: Tile<bool, { [TOK] }> = cmpi(
                tokens,
                broadcast_scalar(active_seq_len, const_shape![TOK]),
                predicate::LessThan,
            );
            let valid: Tile<bool, { [HQ, TOK] }> = valid_1d
                .reshape(const_shape![1, TOK])
                .broadcast(const_shape![HQ, TOK]);
            let s = select(
                valid,
                s,
                broadcast_scalar(-1.0e30f32, const_shape![HQ, TOK]),
            );

            let row_max: Tile<f32, { [HQ] }> = reduce_max(s, 1i32);
            let m_ij: Tile<f32, { [HQ, 1] }> = max_tile(m_i, row_max.reshape(const_shape![HQ, 1]));
            let shifted = s - m_ij.broadcast(const_shape![HQ, TOK]);
            let p: Tile<f32, { [HQ, TOK] }> = select(
                valid,
                exp(shifted),
                broadcast_scalar(0.0f32, const_shape![HQ, TOK]),
            );
            let l_ij: Tile<f32, { [HQ] }> = reduce_sum(p, 1i32);
            let alpha: Tile<f32, { [HQ, 1] }> = exp(m_i - m_ij);
            l_i = l_i * alpha + l_ij.reshape(const_shape![HQ, 1]);
            acc = acc * alpha.broadcast(const_shape![HQ, L]);
            let p16: Tile<f16, { [HQ, TOK] }> = convert_tile(p);
            acc = mma(p16, z, acc);
            m_i = m_ij;
        }
        out_acc.store(acc.reshape(const_shape![1, HQ, L]));
        out_m.store(m_i.reshape(const_shape![1, HQ, 1]));
        out_l.store(l_i.reshape(const_shape![1, HQ, 1]));
    }

    /// Merges C3 partials for the G heads of KV head `pid.0` and applies the V projection.
    #[cutile::entry()]
    pub fn c3_reduce<const HQ: i32, const G: i32, const D: i32, const L: i32>(
        out: &mut Tensor<f32, { [G, D] }>,
        partial_acc: &Tensor<f32, { [-1, HQ, L] }>,
        partial_m: &Tensor<f32, { [-1, HQ, 1] }>,
        partial_l: &Tensor<f32, { [-1, HQ, 1] }>,
        v_projection: &Tensor<f32, { [-1, D] }>,
        num_splits: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let g = pid.0;
        let mut big_m: Tile<f32, { [G, 1] }> = constant(-1.0e30f32, const_shape![G, 1]);
        let mut big_l: Tile<f32, { [G, 1] }> = constant(0.0f32, const_shape![G, 1]);
        let mut acc: Tile<f32, { [G, L] }> = constant(0.0f32, const_shape![G, L]);
        for s in 0i32..num_splits {
            let a: Tile<f32, { [1, G, L] }> =
                partial_acc.load_tile(const_shape![1, G, L], [s, g, 0]);
            let m: Tile<f32, { [1, G, 1] }> = partial_m.load_tile(const_shape![1, G, 1], [s, g, 0]);
            let l: Tile<f32, { [1, G, 1] }> = partial_l.load_tile(const_shape![1, G, 1], [s, g, 0]);
            let m_s: Tile<f32, { [G, 1] }> = m.reshape(const_shape![G, 1]);
            let m_new: Tile<f32, { [G, 1] }> = max_tile(big_m, m_s);
            let w_old: Tile<f32, { [G, 1] }> = exp(big_m - m_new);
            let w_s: Tile<f32, { [G, 1] }> = exp(m_s - m_new);
            big_l = big_l * w_old + l.reshape(const_shape![G, 1]) * w_s;
            acc = acc * w_old.broadcast(const_shape![G, L])
                + a.reshape(const_shape![G, L]) * w_s.broadcast(const_shape![G, L]);
            big_m = m_new;
        }
        let ctx: Tile<f32, { [G, L] }> = true_div(acc, big_l.broadcast(const_shape![G, L]));
        let vp: Tile<f32, { [L, D] }> = v_projection.load_tile(const_shape![L, D], [g, 0]);
        let prod: Tile<f32, { [G, L, D] }> = ctx
            .reshape(const_shape![G, L, 1])
            .broadcast(const_shape![G, L, D])
            * vp.reshape(const_shape![1, L, D])
                .broadcast(const_shape![G, L, D]);
        let context: Tile<f32, { [G, D] }> = reduce_sum(prod, 1i32);
        out.store(context);
    }

    #[cutile::entry()]
    pub fn a3_split<
        const HQ: i32,
        const HKV: i32,
        const G: i32,
        const D: i32,
        const BLK: i32,
        const TOK: i32,
    >(
        out_acc: &mut Tensor<f32, { [1, G, D] }>,
        out_m: &mut Tensor<f32, { [1, G, 1] }>,
        out_l: &mut Tensor<f32, { [1, G, 1] }>,
        q: &Tensor<f32, { [HQ, D] }>,
        k_full: &Tensor<f16, { [-1, D] }>,
        v_full: &Tensor<f16, { [-1, D] }>,
        table: &Tensor<i32, { [-1] }>,
        active_seq_len: i32,
        steps_per_split: i32,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let split = pid.0;
        let kv = pid.1;
        let qg: Tile<f32, { [G, D] }> = q.load_tile(const_shape![G, D], [kv, 0]);
        let q16: Tile<f16, { [G, D] }> = convert_tile(qg);

        let mut m_i: Tile<f32, { [G, 1] }> = constant(-1.0e30f32, const_shape![G, 1]);
        let mut l_i: Tile<f32, { [G, 1] }> = constant(0.0f32, const_shape![G, 1]);
        let mut acc: Tile<f32, { [G, D] }> = constant(0.0f32, const_shape![G, D]);
        let base_block = split * steps_per_split * 2i32;
        for step in 0i32..steps_per_split {
            let b0 = base_block + step * 2i32;
            let e0: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0]);
            let e1: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 1i32]);
            // Physical layout (block, kv_head, token, dim): tile index = block * HKV + kv.
            let phys0: i32 = tile_to_scalar(e0.reshape(const_shape![]));
            let t0 = phys0 * HKV + kv;
            let phys1: i32 = tile_to_scalar(e1.reshape(const_shape![]));
            let t1 = phys1 * HKV + kv;
            let ka: Tile<f16, { [BLK, D] }> = k_full.load_tile(const_shape![BLK, D], [t0, 0]);
            let kb: Tile<f16, { [BLK, D] }> = k_full.load_tile(const_shape![BLK, D], [t1, 0]);
            let k: Tile<f16, { [TOK, D] }> = cat(ka, kb, 0i32);
            let kt: Tile<f16, { [D, TOK] }> = k.transpose();
            let zero: Tile<f32, { [G, TOK] }> = constant(0.0f32, const_shape![G, TOK]);
            let raw: Tile<f32, { [G, TOK] }> = mma(q16, kt, zero);
            let s = raw * broadcast_scalar(scale, const_shape![G, TOK]);

            let tokens: Tile<i32, { [TOK] }> =
                iota(const_shape![TOK]) + broadcast_scalar(b0 * BLK, const_shape![TOK]);
            let valid_1d: Tile<bool, { [TOK] }> = cmpi(
                tokens,
                broadcast_scalar(active_seq_len, const_shape![TOK]),
                predicate::LessThan,
            );
            let valid: Tile<bool, { [G, TOK] }> = valid_1d
                .reshape(const_shape![1, TOK])
                .broadcast(const_shape![G, TOK]);
            let s = select(valid, s, broadcast_scalar(-1.0e30f32, const_shape![G, TOK]));

            let row_max: Tile<f32, { [G] }> = reduce_max(s, 1i32);
            let m_ij: Tile<f32, { [G, 1] }> = max_tile(m_i, row_max.reshape(const_shape![G, 1]));
            let shifted = s - m_ij.broadcast(const_shape![G, TOK]);
            let p: Tile<f32, { [G, TOK] }> = select(
                valid,
                exp(shifted),
                broadcast_scalar(0.0f32, const_shape![G, TOK]),
            );
            let l_ij: Tile<f32, { [G] }> = reduce_sum(p, 1i32);
            let alpha: Tile<f32, { [G, 1] }> = exp(m_i - m_ij);
            l_i = l_i * alpha + l_ij.reshape(const_shape![G, 1]);
            acc = acc * alpha.broadcast(const_shape![G, D]);
            let va: Tile<f16, { [BLK, D] }> = v_full.load_tile(const_shape![BLK, D], [t0, 0]);
            let vb: Tile<f16, { [BLK, D] }> = v_full.load_tile(const_shape![BLK, D], [t1, 0]);
            let v: Tile<f16, { [TOK, D] }> = cat(va, vb, 0i32);
            let p16: Tile<f16, { [G, TOK] }> = convert_tile(p);
            acc = mma(p16, v, acc);
            m_i = m_ij;
        }
        out_acc.store(acc.reshape(const_shape![1, G, D]));
        out_m.store(m_i.reshape(const_shape![1, G, 1]));
        out_l.store(l_i.reshape(const_shape![1, G, 1]));
    }

    /// A3 with eight gathered blocks per step (T8 = 8 * BLK tokens) so the score GEMM output
    /// (G x T8 = 4 x 128 = 512 at model_small) meets the 512-output rule for tensor-core MMA.
    /// The PV GEMM output is G x D regardless of step size. Partials and the reduce kernel are
    /// shared with `a3_split`.
    #[cutile::entry()]
    pub fn a3_split8<
        const HQ: i32,
        const HKV: i32,
        const G: i32,
        const D: i32,
        const BLK: i32,
        const T2: i32,
        const T4: i32,
        const T8: i32,
    >(
        out_acc: &mut Tensor<f32, { [1, G, D] }>,
        out_m: &mut Tensor<f32, { [1, G, 1] }>,
        out_l: &mut Tensor<f32, { [1, G, 1] }>,
        q: &Tensor<f32, { [HQ, D] }>,
        k_full: &Tensor<f16, { [-1, D] }>,
        v_full: &Tensor<f16, { [-1, D] }>,
        table: &Tensor<i32, { [-1] }>,
        active_seq_len: i32,
        steps_per_split: i32,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let split = pid.0;
        let kv = pid.1;
        let qg: Tile<f32, { [G, D] }> = q.load_tile(const_shape![G, D], [kv, 0]);
        let q16: Tile<f16, { [G, D] }> = convert_tile(qg);

        let mut m_i: Tile<f32, { [G, 1] }> = constant(-1.0e30f32, const_shape![G, 1]);
        let mut l_i: Tile<f32, { [G, 1] }> = constant(0.0f32, const_shape![G, 1]);
        let mut acc: Tile<f32, { [G, D] }> = constant(0.0f32, const_shape![G, D]);
        let base_block = split * steps_per_split * 8i32;
        for step in 0i32..steps_per_split {
            let b0 = base_block + step * 8i32;
            let e0: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0]);
            let e1: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 1i32]);
            let e2: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 2i32]);
            let e3: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 3i32]);
            let e4: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 4i32]);
            let e5: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 5i32]);
            let e6: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 6i32]);
            let e7: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 7i32]);
            let p0: i32 = tile_to_scalar(e0.reshape(const_shape![]));
            let p1: i32 = tile_to_scalar(e1.reshape(const_shape![]));
            let p2: i32 = tile_to_scalar(e2.reshape(const_shape![]));
            let p3: i32 = tile_to_scalar(e3.reshape(const_shape![]));
            let p4: i32 = tile_to_scalar(e4.reshape(const_shape![]));
            let p5: i32 = tile_to_scalar(e5.reshape(const_shape![]));
            let p6: i32 = tile_to_scalar(e6.reshape(const_shape![]));
            let p7: i32 = tile_to_scalar(e7.reshape(const_shape![]));
            // Physical layout (block, kv_head, token, dim): tile index = block * HKV + kv.
            let t0 = p0 * HKV + kv;
            let t1 = p1 * HKV + kv;
            let t2 = p2 * HKV + kv;
            let t3 = p3 * HKV + kv;
            let t4 = p4 * HKV + kv;
            let t5 = p5 * HKV + kv;
            let t6 = p6 * HKV + kv;
            let t7 = p7 * HKV + kv;
            let k01: Tile<f16, { [T2, D] }> = cat(
                k_full.load_tile(const_shape![BLK, D], [t0, 0]),
                k_full.load_tile(const_shape![BLK, D], [t1, 0]),
                0i32,
            );
            let k23: Tile<f16, { [T2, D] }> = cat(
                k_full.load_tile(const_shape![BLK, D], [t2, 0]),
                k_full.load_tile(const_shape![BLK, D], [t3, 0]),
                0i32,
            );
            let k45: Tile<f16, { [T2, D] }> = cat(
                k_full.load_tile(const_shape![BLK, D], [t4, 0]),
                k_full.load_tile(const_shape![BLK, D], [t5, 0]),
                0i32,
            );
            let k67: Tile<f16, { [T2, D] }> = cat(
                k_full.load_tile(const_shape![BLK, D], [t6, 0]),
                k_full.load_tile(const_shape![BLK, D], [t7, 0]),
                0i32,
            );
            let k03: Tile<f16, { [T4, D] }> = cat(k01, k23, 0i32);
            let k47: Tile<f16, { [T4, D] }> = cat(k45, k67, 0i32);
            let k: Tile<f16, { [T8, D] }> = cat(k03, k47, 0i32);
            let kt: Tile<f16, { [D, T8] }> = k.transpose();
            let zero: Tile<f32, { [G, T8] }> = constant(0.0f32, const_shape![G, T8]);
            let raw: Tile<f32, { [G, T8] }> = mma(q16, kt, zero);
            let s = raw * broadcast_scalar(scale, const_shape![G, T8]);

            let tokens: Tile<i32, { [T8] }> =
                iota(const_shape![T8]) + broadcast_scalar(b0 * BLK, const_shape![T8]);
            let valid_1d: Tile<bool, { [T8] }> = cmpi(
                tokens,
                broadcast_scalar(active_seq_len, const_shape![T8]),
                predicate::LessThan,
            );
            let valid: Tile<bool, { [G, T8] }> = valid_1d
                .reshape(const_shape![1, T8])
                .broadcast(const_shape![G, T8]);
            let s = select(valid, s, broadcast_scalar(-1.0e30f32, const_shape![G, T8]));

            let row_max: Tile<f32, { [G] }> = reduce_max(s, 1i32);
            let m_ij: Tile<f32, { [G, 1] }> = max_tile(m_i, row_max.reshape(const_shape![G, 1]));
            let shifted = s - m_ij.broadcast(const_shape![G, T8]);
            let p: Tile<f32, { [G, T8] }> = select(
                valid,
                exp(shifted),
                broadcast_scalar(0.0f32, const_shape![G, T8]),
            );
            let l_ij: Tile<f32, { [G] }> = reduce_sum(p, 1i32);
            let alpha: Tile<f32, { [G, 1] }> = exp(m_i - m_ij);
            l_i = l_i * alpha + l_ij.reshape(const_shape![G, 1]);
            acc = acc * alpha.broadcast(const_shape![G, D]);
            let v01: Tile<f16, { [T2, D] }> = cat(
                v_full.load_tile(const_shape![BLK, D], [t0, 0]),
                v_full.load_tile(const_shape![BLK, D], [t1, 0]),
                0i32,
            );
            let v23: Tile<f16, { [T2, D] }> = cat(
                v_full.load_tile(const_shape![BLK, D], [t2, 0]),
                v_full.load_tile(const_shape![BLK, D], [t3, 0]),
                0i32,
            );
            let v45: Tile<f16, { [T2, D] }> = cat(
                v_full.load_tile(const_shape![BLK, D], [t4, 0]),
                v_full.load_tile(const_shape![BLK, D], [t5, 0]),
                0i32,
            );
            let v67: Tile<f16, { [T2, D] }> = cat(
                v_full.load_tile(const_shape![BLK, D], [t6, 0]),
                v_full.load_tile(const_shape![BLK, D], [t7, 0]),
                0i32,
            );
            let v03: Tile<f16, { [T4, D] }> = cat(v01, v23, 0i32);
            let v47: Tile<f16, { [T4, D] }> = cat(v45, v67, 0i32);
            let v: Tile<f16, { [T8, D] }> = cat(v03, v47, 0i32);
            let p16: Tile<f16, { [G, T8] }> = convert_tile(p);
            acc = mma(p16, v, acc);
            m_i = m_ij;
        }
        out_acc.store(acc.reshape(const_shape![1, G, D]));
        out_m.store(m_i.reshape(const_shape![1, G, 1]));
        out_l.store(l_i.reshape(const_shape![1, G, 1]));
    }

    /// A3 with the G query rows of one KV head zero-padded to MP rows (MP == HQ == 16 at
    /// model_small), so both GEMMs have tensor-core shapes: scores MP x TOK (16 x 32 = 512) and
    /// PV MP x D (16 x 64 = 1024), the same shapes as C3. The CTA loads all HQ query rows and
    /// zeroes the rows outside KV head `kv`'s group (tile dims must be powers of two, so a
    /// [MP - G, D] pad tile is not expressible). Real rows sit at kv * G .. kv * G + G inside the
    /// CTA's MP-row partial block; padding rows cost (MP / G)x MACs and are never read back.
    #[cutile::entry()]
    pub fn a3_split_pad<
        const HQ: i32,
        const HKV: i32,
        const G: i32,
        const D: i32,
        const BLK: i32,
        const TOK: i32,
        const MP: i32,
    >(
        out_acc: &mut Tensor<f32, { [1, MP, D] }>,
        out_m: &mut Tensor<f32, { [1, MP, 1] }>,
        out_l: &mut Tensor<f32, { [1, MP, 1] }>,
        q: &Tensor<f32, { [HQ, D] }>,
        k_full: &Tensor<f16, { [-1, D] }>,
        v_full: &Tensor<f16, { [-1, D] }>,
        table: &Tensor<i32, { [-1] }>,
        active_seq_len: i32,
        steps_per_split: i32,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let split = pid.0;
        let kv = pid.1;
        let q_all: Tile<f32, { [MP, D] }> = q.load_tile(const_shape![MP, D], [0, 0]);
        let rows: Tile<i32, { [MP] }> = iota(const_shape![MP]);
        let lo: Tile<bool, { [MP] }> = cmpi(
            rows,
            broadcast_scalar(kv * G, const_shape![MP]),
            predicate::GreaterThanOrEqual,
        );
        let hi: Tile<bool, { [MP] }> = cmpi(
            rows,
            broadcast_scalar(kv * G + G, const_shape![MP]),
            predicate::LessThan,
        );
        let zero_q: Tile<f32, { [MP, D] }> = constant(0.0f32, const_shape![MP, D]);
        let keep_lo: Tile<bool, { [MP, D] }> = lo
            .reshape(const_shape![MP, 1])
            .broadcast(const_shape![MP, D]);
        let keep_hi: Tile<bool, { [MP, D] }> = hi
            .reshape(const_shape![MP, 1])
            .broadcast(const_shape![MP, D]);
        let qp: Tile<f32, { [MP, D] }> = select(keep_hi, select(keep_lo, q_all, zero_q), zero_q);
        let q16: Tile<f16, { [MP, D] }> = convert_tile(qp);

        let mut m_i: Tile<f32, { [MP, 1] }> = constant(-1.0e30f32, const_shape![MP, 1]);
        let mut l_i: Tile<f32, { [MP, 1] }> = constant(0.0f32, const_shape![MP, 1]);
        let mut acc: Tile<f32, { [MP, D] }> = constant(0.0f32, const_shape![MP, D]);
        let base_block = split * steps_per_split * 2i32;
        for step in 0i32..steps_per_split {
            let b0 = base_block + step * 2i32;
            let e0: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0]);
            let e1: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [b0 + 1i32]);
            let phys0: i32 = tile_to_scalar(e0.reshape(const_shape![]));
            let phys1: i32 = tile_to_scalar(e1.reshape(const_shape![]));
            let t0 = phys0 * HKV + kv;
            let t1 = phys1 * HKV + kv;
            let ka: Tile<f16, { [BLK, D] }> = k_full.load_tile(const_shape![BLK, D], [t0, 0]);
            let kb: Tile<f16, { [BLK, D] }> = k_full.load_tile(const_shape![BLK, D], [t1, 0]);
            let k: Tile<f16, { [TOK, D] }> = cat(ka, kb, 0i32);
            let kt: Tile<f16, { [D, TOK] }> = k.transpose();
            let zero: Tile<f32, { [MP, TOK] }> = constant(0.0f32, const_shape![MP, TOK]);
            let raw: Tile<f32, { [MP, TOK] }> = mma(q16, kt, zero);
            let s = raw * broadcast_scalar(scale, const_shape![MP, TOK]);

            let tokens: Tile<i32, { [TOK] }> =
                iota(const_shape![TOK]) + broadcast_scalar(b0 * BLK, const_shape![TOK]);
            let valid_1d: Tile<bool, { [TOK] }> = cmpi(
                tokens,
                broadcast_scalar(active_seq_len, const_shape![TOK]),
                predicate::LessThan,
            );
            let valid: Tile<bool, { [MP, TOK] }> = valid_1d
                .reshape(const_shape![1, TOK])
                .broadcast(const_shape![MP, TOK]);
            let s = select(
                valid,
                s,
                broadcast_scalar(-1.0e30f32, const_shape![MP, TOK]),
            );

            let row_max: Tile<f32, { [MP] }> = reduce_max(s, 1i32);
            let m_ij: Tile<f32, { [MP, 1] }> = max_tile(m_i, row_max.reshape(const_shape![MP, 1]));
            let shifted = s - m_ij.broadcast(const_shape![MP, TOK]);
            let p: Tile<f32, { [MP, TOK] }> = select(
                valid,
                exp(shifted),
                broadcast_scalar(0.0f32, const_shape![MP, TOK]),
            );
            let l_ij: Tile<f32, { [MP] }> = reduce_sum(p, 1i32);
            let alpha: Tile<f32, { [MP, 1] }> = exp(m_i - m_ij);
            l_i = l_i * alpha + l_ij.reshape(const_shape![MP, 1]);
            acc = acc * alpha.broadcast(const_shape![MP, D]);
            let va: Tile<f16, { [BLK, D] }> = v_full.load_tile(const_shape![BLK, D], [t0, 0]);
            let vb: Tile<f16, { [BLK, D] }> = v_full.load_tile(const_shape![BLK, D], [t1, 0]);
            let v: Tile<f16, { [TOK, D] }> = cat(va, vb, 0i32);
            let p16: Tile<f16, { [MP, TOK] }> = convert_tile(p);
            acc = mma(p16, v, acc);
            m_i = m_ij;
        }
        out_acc.store(acc.reshape(const_shape![1, MP, D]));
        out_m.store(m_i.reshape(const_shape![1, MP, 1]));
        out_l.store(l_i.reshape(const_shape![1, MP, 1]));
    }

    /// Merges `a3_split_pad` partials: for KV head g, reads only the G real rows, which sit at
    /// rows g * MP + g * G of the padded partial layout (tile index g * (MP / G) + g).
    #[cutile::entry()]
    pub fn a3_reduce_pad<const G: i32, const D: i32, const MP: i32>(
        out: &mut Tensor<f32, { [G, D] }>,
        partial_acc: &Tensor<f32, { [-1, -1, D] }>,
        partial_m: &Tensor<f32, { [-1, -1, 1] }>,
        partial_l: &Tensor<f32, { [-1, -1, 1] }>,
        num_splits: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_tile = pid.0 * (MP / G) + pid.0;
        let mut big_m: Tile<f32, { [G, 1] }> = constant(-1.0e30f32, const_shape![G, 1]);
        let mut big_l: Tile<f32, { [G, 1] }> = constant(0.0f32, const_shape![G, 1]);
        let mut acc: Tile<f32, { [G, D] }> = constant(0.0f32, const_shape![G, D]);
        for s in 0i32..num_splits {
            let a: Tile<f32, { [1, G, D] }> =
                partial_acc.load_tile(const_shape![1, G, D], [s, row_tile, 0]);
            let m: Tile<f32, { [1, G, 1] }> =
                partial_m.load_tile(const_shape![1, G, 1], [s, row_tile, 0]);
            let l: Tile<f32, { [1, G, 1] }> =
                partial_l.load_tile(const_shape![1, G, 1], [s, row_tile, 0]);
            let m_s: Tile<f32, { [G, 1] }> = m.reshape(const_shape![G, 1]);
            let m_new: Tile<f32, { [G, 1] }> = max_tile(big_m, m_s);
            let w_old: Tile<f32, { [G, 1] }> = exp(big_m - m_new);
            let w_s: Tile<f32, { [G, 1] }> = exp(m_s - m_new);
            big_l = big_l * w_old + l.reshape(const_shape![G, 1]) * w_s;
            acc = acc * w_old.broadcast(const_shape![G, D])
                + a.reshape(const_shape![G, D]) * w_s.broadcast(const_shape![G, D]);
            big_m = m_new;
        }
        out.store(true_div(acc, big_l.broadcast(const_shape![G, D])));
    }

    /// Merges A3 partials for the G heads of KV head `pid.0`.
    #[cutile::entry()]
    pub fn a3_reduce<const HQ: i32, const G: i32, const D: i32>(
        out: &mut Tensor<f32, { [G, D] }>,
        partial_acc: &Tensor<f32, { [-1, HQ, D] }>,
        partial_m: &Tensor<f32, { [-1, HQ, 1] }>,
        partial_l: &Tensor<f32, { [-1, HQ, 1] }>,
        num_splits: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let g = pid.0;
        let mut big_m: Tile<f32, { [G, 1] }> = constant(-1.0e30f32, const_shape![G, 1]);
        let mut big_l: Tile<f32, { [G, 1] }> = constant(0.0f32, const_shape![G, 1]);
        let mut acc: Tile<f32, { [G, D] }> = constant(0.0f32, const_shape![G, D]);
        for s in 0i32..num_splits {
            let a: Tile<f32, { [1, G, D] }> =
                partial_acc.load_tile(const_shape![1, G, D], [s, g, 0]);
            let m: Tile<f32, { [1, G, 1] }> = partial_m.load_tile(const_shape![1, G, 1], [s, g, 0]);
            let l: Tile<f32, { [1, G, 1] }> = partial_l.load_tile(const_shape![1, G, 1], [s, g, 0]);
            let m_s: Tile<f32, { [G, 1] }> = m.reshape(const_shape![G, 1]);
            let m_new: Tile<f32, { [G, 1] }> = max_tile(big_m, m_s);
            let w_old: Tile<f32, { [G, 1] }> = exp(big_m - m_new);
            let w_s: Tile<f32, { [G, 1] }> = exp(m_s - m_new);
            big_l = big_l * w_old + l.reshape(const_shape![G, 1]) * w_s;
            acc = acc * w_old.broadcast(const_shape![G, D])
                + a.reshape(const_shape![G, D]) * w_s.broadcast(const_shape![G, D]);
            big_m = m_new;
        }
        out.store(true_div(acc, big_l.broadcast(const_shape![G, D])));
    }
}
