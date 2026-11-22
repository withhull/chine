//! Correctness and lifetime tests for the TuckCore Metal path.
//!
//! GPU tests skip with a loud note when no Metal device is present and fail
//! instead when `TUCKMILL_REQUIRE_METAL` is set (the repository gate on
//! owned hardware). Comparisons against the CPU scalar reference are
//! tolerance-registered: GPU reduction order differs from the CPU sequence,
//! so exact bit equality is not claimed anywhere.

use super::*;

/// Skip helper: `None` means "no GPU"; with TUCKMILL_REQUIRE_METAL set that
/// is a test failure, never a silent pass.
fn metal_context() -> Option<MetalContext> {
    let probe = probe();
    if !probe.available {
        assert!(
            std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
            "TUCKMILL_REQUIRE_METAL is set but no Metal device is available ({})",
            probe.info
        );
        eprintln!("skipping: no Metal device ({})", probe.info);
        return None;
    }
    Some(MetalContext::open().expect("metal context"))
}

// ------------------------------------------------------------ references --

/// Deterministic xorshift64* stream (the tests never use OS randomness).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// IEEE binary16 round-to-nearest-even, the semantics the Metal `half`
/// conversion is required to match.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if ((bits >> 23) & 0xff) == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = 1u32 << (shift - 1);
        let mut sub = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (sub & 1) != 0) {
            sub += 1;
        }
        return sign | sub;
    }
    let mut half = (((exp as u32) << 10) | (mantissa >> 13)) as u16;
    let rem = mantissa & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0) {
        half += 1;
    }
    sign | half
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal half: value = mantissa * 2^-24; normalize into f32.
            let p = 31 - mantissa.leading_zeros(); // floor(log2(mantissa))
            let m = (mantissa << (23 - p)) & 0x7f_ffff;
            sign | ((p + 103) << 23) | m
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exp + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

/// CPU scalar reference for the packed Q8_0 gemv (sequential f32
/// accumulation, exact same dequantization: d * q).
fn cpu_q8_0_gemv(w: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let blocks = cols / 32;
    let mut y = vec![0f32; rows];
    for (r, slot) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for b in 0..blocks {
            let off = (r * blocks + b) * 34;
            let d = f16_bits_to_f32(u16::from_le_bytes([w[off], w[off + 1]]));
            for (l, xv) in x[b * 32..(b + 1) * 32].iter().enumerate() {
                let q = w[off + 2 + l] as i8 as f32;
                acc += d * q * xv;
            }
        }
        *slot = acc;
    }
    y
}

/// Q8_0 weights with boundary scales and quantized extremes.
fn boundary_weights(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / 32;
    let mut rng = Rng(seed);
    let scales = [
        1.0f32,
        -0.5,
        2f32.powi(-12),
        65504.0,
        -65504.0,
        5.96e-8,
        0.0,
        -0.0,
        448.0,
        6e-5,
    ];
    let mut bytes = Vec::with_capacity(rows * blocks * 34);
    for _ in 0..rows * blocks {
        let scale = scales[(rng.next() % scales.len() as u64) as usize];
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for l in 0..32 {
            let q: i8 = match (rng.next() % 8, l) {
                (0, _) => -128,
                (1, _) => 127,
                (2, _) => 0,
                _ => (rng.next() % 256) as u8 as i8,
            };
            bytes.push(q as u8);
        }
    }
    bytes
}

fn boundary_input(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..cols)
        .map(|i| match i % 7 {
            0 => 0.0,
            1 => -1.0,
            2 => 1e3,
            3 => -1e3,
            _ => ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0,
        })
        .collect()
}

/// Boundary weights restricted to the metal-f16-act domain: every
/// dequantized value (scale x quant, |quant| <= 128) stays inside half
/// range, so staging rounds it to a finite half. The +-65504 scales of
/// `boundary_weights` dequantize past half range and are out of profile
/// domain (real artifact bytes are in domain); products and accumulation
/// stay f32 in the matrix units, so only the staged operands must fit.
fn half_domain_weights(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / 32;
    let mut rng = Rng(seed);
    let scales = [
        1.0f32,
        -0.5,
        2f32.powi(-12),
        448.0,
        -448.0,
        5.96e-8,
        0.0,
        -0.0,
        6e-5,
    ];
    let mut bytes = Vec::with_capacity(rows * blocks * 34);
    for _ in 0..rows * blocks {
        let scale = scales[(rng.next() % scales.len() as u64) as usize];
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for l in 0..32 {
            let q: i8 = match (rng.next() % 8, l) {
                (0, _) => -128,
                (1, _) => 127,
                (2, _) => 0,
                _ => (rng.next() % 256) as u8 as i8,
            };
            bytes.push(q as u8);
        }
    }
    bytes
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---------------------------------------------------------------- tests ---

#[test]
fn probe_reports_device_presence() {
    let probe = probe();
    assert!(!probe.info.is_empty());
    if !probe.available {
        assert!(probe.info.contains("no Metal"));
    } else {
        // The figure is Metal's recommended work-set size, not the
        // machine's total unified memory; the label must say so.
        assert!(
            probe.info.contains("recommended GPU working set"),
            "info must label the working-set figure honestly: {}",
            probe.info
        );
    }
    eprintln!(
        "metal probe: available={} info={}",
        probe.available, probe.info
    );
}

#[test]
fn context_creation_reports_bad_msl_with_a_typed_error() {
    if !probe().available {
        if std::env::var("TUCKMILL_REQUIRE_METAL").is_ok() {
            panic!("TUCKMILL_REQUIRE_METAL is set but no Metal device is available");
        }
        eprintln!("skipping: no Metal device");
        return;
    }
    let err = match MetalContext::open_with_source("kernel void broken( [[ buffer(0) ]] float x {")
    {
        Ok(_) => panic!("malformed MSL must fail context creation"),
        Err(e) => e,
    };
    let text = format!("{err}");
    assert!(
        text.contains("MSL compilation failed") || text.contains("compilation"),
        "the error must name the compilation failure: {text}"
    );
}

#[test]
fn f32_utility_ops_match_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut rng = Rng(7);
    for n in [1usize, 3, 32, 33, 4096, 100_003] {
        let x: Vec<f32> = (0..n)
            .map(|_| ((rng.next() % 200_001) as f32 - 100_000.0) / 1000.0)
            .collect();
        let a: Vec<f32> = (0..n)
            .map(|_| ((rng.next() % 200_001) as f32 - 100_000.0) / 1000.0)
            .collect();
        let factor = -3.25f32;
        let alpha = 0.125f32;

        let xb = ctx.buffer(n * 4, true).unwrap();
        let ab = ctx.buffer(n * 4, true).unwrap();
        let yb = ctx.buffer(n * 4, true).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        ab.upload(0, &f32_bytes(&a)).unwrap();

        ctx.f32_scale(&xb, &yb, factor, n as u32).unwrap();
        ctx.f32_axpy(&ab, &yb, alpha, n as u32).unwrap();
        ctx.fence().unwrap();

        let mut out = vec![0u8; n * 4];
        yb.download(0, &mut out).unwrap();
        for (i, chunk) in out.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes(chunk.try_into().unwrap());
            let want = x[i] * factor + alpha * a[i];
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "n={n} i={i}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn f16_conversions_match_ieee_round_to_nearest_even() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // Values chosen across representable ranges, subnormals, and overflow.
    let values: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        65504.0,
        65520.0,
        100_000.0,
        -100_000.0,
        5.960_464_5e-8, // min positive subnormal
        2.980_232_2e-8, // half of it: rounds to zero (ties to even)
        1.0e-7,
        -6.0e-8,
        44.856,
        std::f32::consts::PI,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let n = values.len();
    let xb = ctx.buffer(n * 4, true).unwrap();
    let hb = ctx.buffer(n * 2, true).unwrap();
    let back = ctx.buffer(n * 4, true).unwrap();
    xb.upload(0, &f32_bytes(&values)).unwrap();
    ctx.cast_f32_to_f16(&xb, &hb, n as u32).unwrap();
    ctx.fence().unwrap();

    let mut half_bytes = vec![0u8; n * 2];
    hb.download(0, &mut half_bytes).unwrap();
    for (i, chunk) in half_bytes.chunks_exact(2).enumerate() {
        let got = u16::from_le_bytes(chunk.try_into().unwrap());
        let want = f32_to_f16_bits(values[i]);
        assert_eq!(got, want, "f32->f16 mismatch for {}", values[i]);
    }

    // And back: f16 -> f32 must reproduce the reference decode exactly.
    ctx.cast_f16_to_f32(&hb, &back, n as u32).unwrap();
    ctx.fence().unwrap();
    let mut out = vec![0u8; n * 4];
    back.download(0, &mut out).unwrap();
    for (i, chunk) in out.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes(chunk.try_into().unwrap());
        let want = f16_bits_to_f32(f32_to_f16_bits(values[i]));
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "f16->f32 mismatch for input {}",
            values[i]
        );
    }
}

#[test]
fn q8_0_gemv_minimal_ones_case() {
    // One row, one block, all q = 1, d = 1.0, x = 2.0: y[0] must be 64.
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut w = Vec::new();
    w.extend_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
    w.extend_from_slice(&[1u8; 32]);
    let x = vec![2.0f32; 32];
    let wb = ctx.buffer(w.len(), true).unwrap();
    let xb = ctx.buffer(32 * 4, true).unwrap();
    let yb = ctx.buffer(4, true).unwrap();
    wb.upload(0, &w).unwrap();
    xb.upload(0, &f32_bytes(&x)).unwrap();
    ctx.q8_0_gemv(&wb, &xb, &yb, 1, 32).unwrap();
    ctx.fence().unwrap();
    let mut out = [0u8; 4];
    yb.download(0, &mut out).unwrap();
    let got = f32::from_le_bytes(out);
    assert_eq!(got, 64.0, "gmm ones case");
}

#[test]
fn q8_0_gemv_matches_the_scalar_reference_across_real_shapes() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // Real Qwen3-0.6B shapes plus small tail/Odd row counts; cols always a
    // multiple of 32 by the format, rows exercise 1, primes and large grids.
    let shapes: [(usize, usize, u64); 7] = [
        (1, 32, 1),
        (3, 32, 2),
        (17, 64, 3),
        (1024, 1024, 4),    // o_proj / embed-like
        (2048, 1024, 5),    // q_proj
        (3072, 1024, 6),    // ffn gate/up/down
        (151_936, 1024, 7), // token embedding / lm head
    ];
    for (rows, cols, seed) in shapes {
        let w = boundary_weights(rows, cols, seed);
        let x = boundary_input(cols, seed ^ 0x9e37);
        let wb = ctx.buffer(w.len(), true).unwrap();
        let xb = ctx.buffer(cols * 4, true).unwrap();
        let yb = ctx.buffer(rows * 4, true).unwrap();
        wb.upload(0, &w).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        ctx.q8_0_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
            .unwrap();
        ctx.fence().unwrap();

        let mut out = vec![0u8; rows * 4];
        yb.download(0, &mut out).unwrap();
        let gpu: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let cpu = cpu_q8_0_gemv(&w, &x, rows, cols);
        let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
        let check_rows: Vec<usize> = if rows <= 32 {
            (0..rows).collect()
        } else {
            vec![0, 1, 2, rows / 2, rows - 3, rows - 2, rows - 1]
        };
        for &r in &check_rows {
            let diff = (gpu[r] - cpu[r]).abs();
            assert!(
                diff <= 1e-4 * scale,
                "shape ({rows},{cols}) row {r}: gpu {}, cpu {}, diff {diff} over scale {scale}",
                gpu[r],
                cpu[r]
            );
        }
    }
}

#[test]
fn q8_0_gemv_rejects_malformed_arguments() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let wb = ctx.buffer(2 * 34, true).unwrap(); // one row of 32 cols
    let xb = ctx.buffer(32 * 4, true).unwrap();
    let yb = ctx.buffer(4, true).unwrap();
    // cols not a multiple of 32.
    assert!(ctx.q8_0_gemv(&wb, &xb, &yb, 2, 30).is_err());
    // zero rows.
    assert!(ctx.q8_0_gemv(&wb, &xb, &yb, 0, 32).is_err());
    // Output buffer too small for two rows.
    let err = ctx.q8_0_gemv(&wb, &xb, &yb, 2, 32).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidRequest);
    // Weight buffer too small.
    let big_w = ctx.buffer(34, true).unwrap();
    let big_y = ctx.buffer(8, true).unwrap();
    assert!(ctx.q8_0_gemv(&big_w, &xb, &big_y, 2, 32).is_err());
    // Zero-length allocation is refused outright.
    assert!(ctx.buffer(0, true).is_err());
}

#[test]
fn buffers_are_context_scoped_and_accounted() {
    let Some(ctx) = metal_context() else {
        return;
    };
    assert_eq!(ctx.allocated_bytes(), 0);
    let a = ctx.buffer(1024, true).unwrap();
    let b = ctx.buffer(2048, false).unwrap();
    assert_eq!(ctx.allocated_bytes(), 3072);
    assert_eq!(a.len(), 1024);
    assert_eq!(b.len(), 2048);

    // A buffer from another context is refused by every operation.
    let other = MetalContext::open().unwrap();
    let foreign = other.buffer(64, true).unwrap();
    let err = ctx.f32_scale(&foreign, &a, 1.0, 4).unwrap_err();
    assert!(format!("{err}").contains("different context"), "{err}");
    assert!(ctx.q8_0_gemv(&foreign, &foreign, &a, 1, 32).is_err());

    // Private buffers have no host copy.
    let mut out = [0u8; 8];
    assert!(b.download(0, &mut out).is_err());

    drop(a);
    drop(b);
    assert_eq!(ctx.allocated_bytes(), 0);
    assert_eq!(other.allocated_bytes(), 64);
    drop(foreign);
    assert_eq!(other.allocated_bytes(), 0);
}

#[test]
fn dispatch_completion_accounting_is_exact() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let x = ctx.buffer(64, true).unwrap();
    let y = ctx.buffer(64, true).unwrap();
    x.upload(0, &f32_bytes(&[1.0; 16])).unwrap();
    for _ in 0..5 {
        ctx.f32_scale(&x, &y, 2.0, 16).unwrap();
    }
    assert_eq!(ctx.committed_dispatches(), 5);
    ctx.fence().unwrap();
    assert_eq!(ctx.completed_dispatches(), 5);
    assert!(ctx.last_gpu_ms() >= 0.0);
    let mut out = vec![0u8; 64];
    y.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 2.0);
    }
}

#[test]
fn upload_download_respect_bounds_and_offsets() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let buf = ctx.buffer(16, true).unwrap();
    buf.upload(8, &[9u8; 8]).unwrap();
    let mut out = [0u8; 16];
    buf.download(0, &mut out).unwrap();
    assert_eq!(&out[8..], &[9u8; 8]);
    assert_eq!(&out[..8], &[0u8; 8]);
    assert!(buf.upload(12, &[0u8; 8]).is_err(), "range past the end");
    assert!(buf.download(16, &mut [0u8; 1]).is_err());
}

#[test]
fn copy_contract_synchronizes_with_in_flight_compute() {
    // A download immediately after asynchronous compute must observe the
    // finished result (the copy contract fences internally); an upload must
    // not race in-flight work either.
    let Some(ctx) = metal_context() else {
        return;
    };
    let x = ctx.buffer(64, true).unwrap();
    let y = ctx.buffer(64, true).unwrap();
    x.upload(0, &f32_bytes(&[3.0; 16])).unwrap();
    ctx.f32_scale(&x, &y, 4.0, 16).unwrap();
    // No explicit fence here: the copy contract orders it.
    let mut out = vec![0u8; 64];
    y.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 12.0);
    }
    // Upload-while-pending: overwrite the input, run again, read back.
    x.upload(0, &f32_bytes(&[5.0; 16])).unwrap();
    ctx.f32_scale(&x, &y, 2.0, 16).unwrap();
    y.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 10.0);
    }
}

#[test]
fn dropping_buffers_before_fence_is_safe() {
    // In-flight leases must keep device resources alive: the handles are
    // freed here before any fence, and the GPU work must still complete
    // correctly with the allocation accounting returning to zero.
    let Some(ctx) = metal_context() else {
        return;
    };
    let x = ctx.buffer(64, true).unwrap();
    let y = ctx.buffer(64, true).unwrap();
    let out_keep = ctx.buffer(64, true).unwrap();
    x.upload(0, &f32_bytes(&[7.0; 16])).unwrap();
    ctx.f32_scale(&x, &out_keep, 3.0, 16).unwrap();
    drop(x); // GPU work on `x` is still pending
    ctx.fence().unwrap();
    let mut out = vec![0u8; 64];
    out_keep.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 21.0);
    }
    drop(y);
    drop(out_keep);
    assert_eq!(ctx.allocated_bytes(), 0);
}

#[test]
fn grouped_encoding_matches_individual_dispatch_and_aborts_cleanly() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let a = ctx.buffer(64, true).unwrap();
    let b = ctx.buffer(64, true).unwrap();
    a.upload(0, &f32_bytes(&[2.0; 16])).unwrap();

    // Grouped: two operations, one commit.
    let committed_before = ctx.committed_dispatches();
    ctx.begin_group().unwrap();
    ctx.f32_scale(&a, &b, 3.0, 16).unwrap();
    ctx.f32_axpy(&a, &b, 0.5, 16).unwrap();
    ctx.end_group().unwrap();
    assert_eq!(ctx.committed_dispatches(), committed_before + 1);
    ctx.fence().unwrap();
    let mut grouped = vec![0u8; 64];
    b.download(0, &mut grouped).unwrap();
    for chunk in grouped.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 7.0);
    }

    // Individual: same math, two commits.
    ctx.f32_scale(&a, &b, 3.0, 16).unwrap();
    ctx.f32_axpy(&a, &b, 0.5, 16).unwrap();
    ctx.fence().unwrap();
    let mut individual = vec![0u8; 64];
    b.download(0, &mut individual).unwrap();
    assert_eq!(grouped, individual);

    // Abort: the discarded group never commits and never writes.
    b.upload(0, &f32_bytes(&[0.0; 16])).unwrap();
    let committed_before = ctx.committed_dispatches();
    ctx.begin_group().unwrap();
    ctx.f32_scale(&a, &b, 99.0, 16).unwrap();
    ctx.abort_group().unwrap();
    assert_eq!(ctx.committed_dispatches(), committed_before);
    let mut aborted = vec![0u8; 64];
    b.download(0, &mut aborted).unwrap();
    assert!(aborted.iter().all(|byte| *byte == 0));

    // A fence with an open group is refused rather than guessed.
    ctx.begin_group().unwrap();
    assert!(ctx.fence().is_err());
    ctx.abort_group().unwrap();
}

/// CPU references for the transformer ops (mirroring tuckcore's scalar
/// executor semantics: f32, sequential accumulation).
fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0f32 / (ss / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, wv)| v * inv * wv).collect()
}

#[test]
fn grouped_profiling_attributes_per_operator() {
    let Some(ctx) = metal_context() else {
        return;
    };
    if !ctx.grouped_profiling_supported() {
        eprintln!(
            "skipping: this GPU does not expose dispatch-boundary counter sampling \
             (grouped mode keeps the single 'group' label)"
        );
        return;
    }
    // The same operator sequence, ungrouped (per-op command buffers) and
    // grouped (one shared command buffer, counter samples at dispatch
    // boundaries). Every operator must appear with the same count in both
    // regimes, and the grouped time must not exceed the ungrouped time:
    // counter sampling measures the dispatch only, while ungrouped
    // attribution includes the command-buffer overhead the earlier
    // projections mistook for operator cost.
    let rows = 128usize;
    let cols = 1152usize;
    let xb = ctx.buffer(rows * cols * 4, true).unwrap();
    let resb = ctx.buffer(rows * cols * 4, true).unwrap();
    let wb = ctx.buffer(cols * 4, true).unwrap();
    let yb = ctx.buffer(rows * cols * 4, true).unwrap();
    let xb2 = ctx.buffer(cols * 4, true).unwrap();
    let yb2 = ctx.buffer(cols * 4, true).unwrap();
    let data = f32_bytes(&vec![0.5f32; cols * 4]);
    xb.upload(0, &data).unwrap();
    resb.upload(0, &data).unwrap();
    wb.upload(0, &data).unwrap();
    xb2.upload(0, &data).unwrap();
    let run = || {
        ctx.rmsnorm(&xb, &wb, &yb, rows as u32, cols as u32, 1e-6)
            .unwrap();
        ctx.norm_add(&xb, &wb, &resb, &yb, rows as u32, cols as u32, 1e-6)
            .unwrap();
        ctx.f32_scale(&xb2, &yb2, 2.0, (cols * 4) as u32).unwrap();
    };
    let gather = |name: &str| -> std::collections::BTreeMap<String, (u64, u64)> {
        ctx.set_profiling(true);
        run();
        if name == "grouped" {
            ctx.begin_group().unwrap();
            run();
            ctx.end_group().unwrap();
        }
        ctx.fence().unwrap();
        let snap = ctx.profile_snapshot();
        ctx.set_profiling(false);
        snap.into_iter()
            .map(|(l, n, c)| (l, (n, c)))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let ungrouped = gather("ungrouped");
    let grouped = gather("grouped");
    for label in ["rmsnorm", "norm_add", "scale"] {
        let u = ungrouped
            .get(label)
            .unwrap_or_else(|| panic!("ungrouped {label}"));
        let g = grouped
            .get(label)
            .unwrap_or_else(|| panic!("grouped {label}"));
        assert_eq!(u.1, g.1, "{label}: dispatch counts");
        assert!(
            g.0 <= u.0,
            "{label}: grouped {g0} ns exceeded ungrouped {u0} ns",
            g0 = g.0,
            u0 = u.0
        );
        assert!(
            g.0 >= u.0 / 8,
            "{label}: grouped {g0} ns implausibly below ungrouped {u0} ns",
            g0 = g.0,
            u0 = u.0
        );
        eprintln!(
            "{label}: grouped {g0} ns vs ungrouped {u0} ns (x{runs})",
            g0 = g.0,
            u0 = u.0,
            runs = g.1
        );
    }
    // The grouped command buffer's own total must appear as the "group"
    // label (folded from GPU timestamps by the completion handler).
    let g_total = grouped.get("group").map(|(n, _)| *n);
    if let Some(total) = g_total {
        let ops: u64 = ["rmsnorm", "norm_add", "scale"]
            .iter()
            .filter_map(|l| grouped.get(*l).map(|(n, _)| *n))
            .sum();
        assert!(
            ops <= total,
            "grouped per-op sum {ops} exceeded the command buffer total {total}"
        );
        eprintln!("group total {total} ns, per-op sum {ops} ns");
    }
}

#[test]
fn norm_add_matches_the_sequential_pair() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // The fused kernel replaces the engine's post-block pair:
    // `rmsnorm(x)` then `y += normed` where y is the residual stream.
    // Registered: bit-identical to the pair (measured 0 ulps on every
    // shape). The kernel routes the normed product through an optnone,
    // no-inline helper — a real call boundary that rounds to f32 before
    // the residual add, exactly the pair's device-memory round-trip.
    // That is what the chunk-vs-per-token engine bit-identity gates rely
    // on (either mode may run fused). The `aliased` passes use the
    // engine's in-place wiring (the output buffer is the residual input).
    // Shapes cover Gemma 3's 1152 hidden, the 2B's 2048, decode rows and
    // partial chunks.
    let mut rng = Rng(29);
    for (rows, cols, aliased) in [
        (1usize, 1024usize, false),
        (1, 1152, false),
        (128, 1152, false),
        (37, 1152, false),
        (37, 1152, true),
        (128, 2048, false),
        (5, 2048, true),
        (128, 6912, false),
    ] {
        let x: Vec<f32> = (0..rows * cols)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let res: Vec<f32> = (0..rows * cols)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let w: Vec<f32> = (0..cols)
            .map(|_| 0.5 + (rng.next() % 1000) as f32 / 1000.0)
            .collect();
        let eps = 1e-6f32;
        let n = rows * cols;
        let xb = ctx.buffer(n * 4, true).unwrap();
        let resb = ctx.buffer(n * 4, true).unwrap();
        let wb = ctx.buffer(cols * 4, true).unwrap();
        let nb = ctx.buffer(n * 4, true).unwrap();
        let refb = ctx.buffer(n * 4, true).unwrap();
        let fb = ctx.buffer(n * 4, true).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        resb.upload(0, &f32_bytes(&res)).unwrap();
        wb.upload(0, &f32_bytes(&w)).unwrap();
        // Sequential pair: norm into nb, then the reference accumulator
        // (a copy of the residual) += nb.
        ctx.rmsnorm(&xb, &wb, &nb, rows as u32, cols as u32, eps)
            .unwrap();
        refb.upload(0, &f32_bytes(&res)).unwrap();
        ctx.f32_axpy(&nb, &refb, 1.0, n as u32).unwrap();
        // Fused: one pass from x and the residual.
        if aliased {
            ctx.norm_add(&xb, &wb, &resb, &resb, rows as u32, cols as u32, eps)
                .unwrap();
        } else {
            ctx.norm_add(&xb, &wb, &resb, &fb, rows as u32, cols as u32, eps)
                .unwrap();
        }
        ctx.fence().unwrap();
        let mut seq = vec![0u8; n * 4];
        let mut fus = vec![0u8; n * 4];
        refb.download(0, &mut seq).unwrap();
        if aliased {
            resb.download(0, &mut fus).unwrap();
        } else {
            fb.download(0, &mut fus).unwrap();
        }
        for i in 0..n {
            let a = f32::from_le_bytes(seq[i * 4..i * 4 + 4].try_into().unwrap());
            let b = f32::from_le_bytes(fus[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "({rows},{cols},aliased={aliased}) element {i}: fused {a} vs sequential {b}"
            );
        }
    }
    eprintln!("norm_add vs sequential pair: bit-identical (all shapes, incl. aliased)");
}

#[test]
fn norm_add_pair_timing_when_requested() {
    if std::env::var("TUCKMILL_PROFILE").ok().as_deref() != Some("1") {
        eprintln!("skipping: TUCKMILL_PROFILE=1 not set (measurement only)");
        return;
    }
    let Some(ctx) = metal_context() else {
        return;
    };
    // The engine's post-block sites: Gemma 3's 1152 hidden, the 2B's 2048,
    // at the 128-token chunk row count.
    for (rows, cols) in [(128usize, 1152usize), (128, 2048), (1, 2048), (1, 1152)] {
        let n = (rows * cols) as u32;
        let xb = ctx.buffer(n as usize * 4, true).unwrap();
        let resb = ctx.buffer(n as usize * 4, true).unwrap();
        let wb = ctx.buffer(cols * 4, true).unwrap();
        let yb = ctx.buffer(n as usize * 4, true).unwrap();
        let group = |fused: bool| {
            ctx.begin_group().unwrap();
            for _ in 0..16 {
                if fused {
                    ctx.norm_add(&xb, &wb, &resb, &yb, rows as u32, cols as u32, 1e-6)
                        .unwrap();
                } else {
                    ctx.rmsnorm(&xb, &wb, &yb, rows as u32, cols as u32, 1e-6)
                        .unwrap();
                    ctx.f32_axpy(&resb, &yb, 1.0, n).unwrap();
                }
            }
            ctx.end_group().unwrap();
            ctx.fence().unwrap();
        };
        for fused in [false, true] {
            for _ in 0..3 {
                group(fused);
            }
            let iters = 40;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                group(fused);
            }
            let per = t0.elapsed().as_secs_f64() / (iters * 16) as f64;
            eprintln!(
                "norm-add {} {rows:>3}x{cols:<4}: {:>7.2} us per row-set",
                if fused { "fused " } else { "pair  " },
                per * 1e6
            );
        }
    }
}

#[test]
fn qk_norm_matches_the_two_rmsnorm_launches_bit_for_bit() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // The merged Q/K norm must be bit-identical to the two rmsnorm
    // launches it replaces (no add in the expression, so no contraction).
    // Shapes: the Gemma 3 chunk core (512 q rows, 128 k rows of 256), the
    // 2B's per-token core, and odd row counts.
    let mut rng = Rng(41);
    for (q_rows, k_rows, cols) in [
        (512usize, 128usize, 256usize),
        (4, 1, 256),
        (128, 32, 128),
        (7, 3, 96),
    ] {
        let q: Vec<f32> = (0..q_rows * cols)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let k: Vec<f32> = (0..k_rows * cols)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let qw: Vec<f32> = (0..cols)
            .map(|_| 0.5 + (rng.next() % 1000) as f32 / 1000.0)
            .collect();
        let kw: Vec<f32> = (0..cols)
            .map(|_| 0.5 + (rng.next() % 1000) as f32 / 1000.0)
            .collect();
        let eps = 1e-6f32;
        let qb = ctx.buffer(q_rows * cols * 4, true).unwrap();
        let qwb = ctx.buffer(cols * 4, true).unwrap();
        let qout = ctx.buffer(q_rows * cols * 4, true).unwrap();
        let kb = ctx.buffer(k_rows * cols * 4, true).unwrap();
        let kwb = ctx.buffer(cols * 4, true).unwrap();
        let kout = ctx.buffer(k_rows * cols * 4, true).unwrap();
        let mqout = ctx.buffer(q_rows * cols * 4, true).unwrap();
        let mkout = ctx.buffer(k_rows * cols * 4, true).unwrap();
        qb.upload(0, &f32_bytes(&q)).unwrap();
        qwb.upload(0, &f32_bytes(&qw)).unwrap();
        kb.upload(0, &f32_bytes(&k)).unwrap();
        kwb.upload(0, &f32_bytes(&kw)).unwrap();
        // Sequential: two rmsnorm launches.
        ctx.rmsnorm(&qb, &qwb, &qout, q_rows as u32, cols as u32, eps)
            .unwrap();
        ctx.rmsnorm(&kb, &kwb, &kout, k_rows as u32, cols as u32, eps)
            .unwrap();
        // Merged: one launch.
        ctx.qk_norm(
            &qb,
            &qwb,
            &mqout,
            &kb,
            &kwb,
            &mkout,
            q_rows as u32,
            k_rows as u32,
            cols as u32,
            eps,
        )
        .unwrap();
        ctx.fence().unwrap();
        let mut seq_q = vec![0u8; q_rows * cols * 4];
        let mut m_q = vec![0u8; q_rows * cols * 4];
        let mut seq_k = vec![0u8; k_rows * cols * 4];
        let mut m_k = vec![0u8; k_rows * cols * 4];
        qout.download(0, &mut seq_q).unwrap();
        mqout.download(0, &mut m_q).unwrap();
        kout.download(0, &mut seq_k).unwrap();
        mkout.download(0, &mut m_k).unwrap();
        assert_eq!(seq_q, m_q, "({q_rows},{k_rows},{cols}) q bits");
        assert_eq!(seq_k, m_k, "({q_rows},{k_rows},{cols}) k bits");
    }
}

#[test]
fn rmsnorm_matches_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut rng = Rng(11);
    for (rows, cols) in [(1usize, 1024usize), (16, 128), (8, 128), (1, 7)] {
        let x: Vec<f32> = (0..rows * cols)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let w: Vec<f32> = (0..cols)
            .map(|_| 0.5 + (rng.next() % 1000) as f32 / 1000.0)
            .collect();
        let eps = 1e-6f32;
        let xb = ctx.buffer(rows * cols * 4, true).unwrap();
        let wb = ctx.buffer(cols * 4, true).unwrap();
        let yb = ctx.buffer(rows * cols * 4, true).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        wb.upload(0, &f32_bytes(&w)).unwrap();
        ctx.rmsnorm(&xb, &wb, &yb, rows as u32, cols as u32, eps)
            .unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; rows * cols * 4];
        yb.download(0, &mut out).unwrap();
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for r in 0..rows {
            let want = cpu_rmsnorm(&x[r * cols..(r + 1) * cols], &w, eps);
            for c in 0..cols {
                let (g, t) = (got[r * cols + c], want[c]);
                assert!(
                    (g - t).abs() <= 1e-5 * t.abs().max(1.0),
                    "({rows},{cols}) [{r},{c}]: got {g}, want {t}"
                );
            }
        }
    }
}

#[test]
fn silu_mul_and_row_dequant_match_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut rng = Rng(19);
    for n in [3072usize, 3071, 1] {
        let gate: Vec<f32> = (0..n)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let up: Vec<f32> = (0..n)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let gb = ctx.buffer(n * 4, true).unwrap();
        let ub = ctx.buffer(n * 4, true).unwrap();
        let ob = ctx.buffer(n * 4, true).unwrap();
        gb.upload(0, &f32_bytes(&gate)).unwrap();
        ub.upload(0, &f32_bytes(&up)).unwrap();
        ctx.silu_mul(&gb, &ub, &ob, n as u32).unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; n * 4];
        ob.download(0, &mut out).unwrap();
        for (i, chunk) in out.chunks_exact(4).enumerate() {
            let g = f32::from_le_bytes(chunk.try_into().unwrap());
            let want = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            assert!(
                (g - want).abs() <= 1e-5 * want.abs().max(1.0),
                "n={n} i={i}"
            );
        }
    }

    // Q8_0 row dequantization on boundary weights.
    let (rows, cols) = (8usize, 64usize);
    let w = boundary_weights(rows, cols, 23);
    let wb = ctx.buffer(w.len(), true).unwrap();
    let ob = ctx.buffer(cols * 4, true).unwrap();
    wb.upload(0, &w).unwrap();
    for row in [0u32, 5, 7] {
        ctx.q8_0_row(&wb, &ob, row, cols as u32, 0).unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; cols * 4];
        ob.download(0, &mut out).unwrap();
        let blocks = cols / 32;
        for d in 0..cols {
            let b = d / 32;
            let lane = d % 32;
            let off = (row as usize * blocks + b) * 34;
            let scale = f16_bits_to_f32(u16::from_le_bytes([w[off], w[off + 1]]));
            let q = w[off + 2 + lane] as i8 as f32;
            let got = f32::from_le_bytes(out[d * 4..d * 4 + 4].try_into().unwrap());
            assert_eq!(got, scale * q, "row {row} dim {d}");
        }
    }
}

#[test]
fn f16_gemv_matches_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut rng = Rng(29);
    for (rows, cols, seed) in [(1usize, 32usize, 1u64), (17, 64, 2), (1024, 1024, 3)] {
        let wf: Vec<f32> = (0..rows * cols)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let w: Vec<u8> = wf
            .iter()
            .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
            .collect();
        let x = boundary_input(cols, seed);
        let wb = ctx.buffer(w.len(), true).unwrap();
        let xb = ctx.buffer(cols * 4, true).unwrap();
        let yb = ctx.buffer(rows * 4, true).unwrap();
        wb.upload(0, &w).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        ctx.f16_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
            .unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; rows * 4];
        yb.download(0, &mut out).unwrap();
        let gpu: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut scale = 1.0f32;
        for r in [0usize, rows / 2, rows - 1] {
            let row = &w[r * cols * 2..(r + 1) * cols * 2];
            let want: f32 = row
                .chunks_exact(2)
                .zip(&x)
                .map(|(c, xv)| f16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())) * xv)
                .sum();
            scale = scale.max(want.abs());
            assert!(
                (gpu[r] - want).abs() <= 1e-4 * scale,
                "({rows},{cols}) row {r}: {} vs {want}",
                gpu[r]
            );
        }
    }
}

/// Real artifact operator check (machine-local): the packed gemv must match
/// the scalar reference on real Q8_0 tensor bytes, not just synthetic data.
#[test]
fn q8_0_gemv_matches_on_real_artifact_bytes() {
    let Ok(gguf) = std::env::var("TUCKMILL_REAL_GGUF") else {
        eprintln!("skipping: TUCKMILL_REAL_GGUF not set (real-weight label)");
        return;
    };
    let Some(ctx) = metal_context() else {
        return;
    };
    let path = std::path::PathBuf::from(&gguf);
    let inspection =
        tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
            .expect("inspect real artifact");
    let tensor = inspection
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight present");
    assert_eq!(tensor.type_id, 8, "expected Q8_0");
    let dims = &tensor.dims; // [in, out]
    let (cols, rows) = (dims[0] as usize, dims[1] as usize);
    assert_eq!(tensor.byte_size as usize, rows * (cols / 32) * 34);

    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(&path).unwrap();
    file.seek(SeekFrom::Start(inspection.data_start + tensor.offset))
        .unwrap();
    let mut w = vec![0u8; tensor.byte_size as usize];
    file.read_exact(&mut w).unwrap();

    let x = boundary_input(cols, 42);
    let wb = ctx.buffer(w.len(), true).unwrap();
    let xb = ctx.buffer(cols * 4, true).unwrap();
    let yb = ctx.buffer(rows * 4, true).unwrap();
    wb.upload(0, &w).unwrap();
    xb.upload(0, &f32_bytes(&x)).unwrap();
    ctx.q8_0_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
        .unwrap();
    ctx.fence().unwrap();

    let mut out = vec![0u8; rows * 4];
    yb.download(0, &mut out).unwrap();
    let gpu: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let cpu = cpu_q8_0_gemv(&w, &x, rows, cols);
    let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    for r in [0usize, 1, rows / 2, rows - 2, rows - 1] {
        let diff = (gpu[r] - cpu[r]).abs();
        assert!(
            diff <= 1e-4 * scale,
            "row {r}: gpu {}, cpu {}, diff {diff} over scale {scale}",
            gpu[r],
            cpu[r]
        );
    }
}

#[test]
fn in_flight_bytes_stay_visible_until_completion() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let x = ctx.buffer(4096, true).unwrap();
    let y = ctx.buffer(4096, true).unwrap();
    x.upload(0, &f32_bytes(&[1.0; 1024])).unwrap();
    assert_eq!(ctx.in_flight_bytes(), 0);
    ctx.f32_scale(&x, &y, 2.0, 1024).unwrap();
    // Handles dropped, GPU work still pending: the lease keeps the storage
    // physically alive and visible to accounting.
    drop(x);
    assert_eq!(ctx.allocated_bytes(), 4096, "only y is still handle-live");
    drop(y);
    assert_eq!(
        ctx.allocated_bytes(),
        0,
        "handle bytes are released on drop"
    );
    let leased_before = ctx.in_flight_bytes();
    assert!(
        leased_before >= 8192,
        "dropped-but-leased buffers must remain visible: {leased_before}"
    );
    assert!(ctx.live_gpu_bytes() >= leased_before);
    ctx.fence().unwrap();
    assert_eq!(ctx.in_flight_bytes(), 0, "leases release at completion");

    // Correctness after the fully-lease-only run: same math, kept handles.
    let x2 = ctx.buffer(4096, true).unwrap();
    let y2 = ctx.buffer(4096, true).unwrap();
    x2.upload(0, &f32_bytes(&[3.0; 1024])).unwrap();
    ctx.f32_scale(&x2, &y2, 2.0, 1024).unwrap();
    ctx.fence().unwrap();
    let mut out = vec![0u8; 4096];
    y2.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4).take(1024) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 6.0);
    }
    drop(x2);
    drop(y2);
    assert_eq!(ctx.allocated_bytes(), 0);
}

#[test]
fn batched_gemm_matches_the_scalar_reference_for_all_rows() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // (rows, cols, all-rows?) — realistic projection shapes; the 151936-row
    // vocabulary projection is spot-checked to keep the reference bounded.
    let shapes: [(usize, usize, bool); 5] = [
        (32, 32, true),
        (64, 1024, true),
        (2048, 1024, true),
        (3072, 1024, true),
        (151_936, 1024, false),
    ];
    for (rows, cols, all_rows) in shapes {
        let w = boundary_weights(rows, cols, 101 + rows as u64);
        for batch in [1usize, 4, 8, 16] {
            let xs: Vec<Vec<f32>> = (0..batch)
                .map(|b| boundary_input(cols, 200 + b as u64))
                .collect();
            let mut flat = Vec::with_capacity(batch * cols);
            for x in &xs {
                flat.extend_from_slice(x);
            }
            let wb = ctx.buffer(w.len(), true).unwrap();
            let xb = ctx.buffer(flat.len() * 4, true).unwrap();
            let yb = ctx.buffer(batch * rows * 4, true).unwrap();
            wb.upload(0, &w).unwrap();
            xb.upload(0, &f32_bytes(&flat)).unwrap();
            ctx.q8_0_gemm(&wb, &xb, &yb, rows as u32, cols as u32, batch as u32, 0, 0)
                .unwrap();
            ctx.fence().unwrap();
            let mut out = vec![0u8; batch * rows * 4];
            yb.download(0, &mut out).unwrap();
            let gpu: Vec<f32> = out
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let check_rows: Vec<usize> = if all_rows {
                (0..rows).collect()
            } else {
                vec![0, 1, rows / 2, rows - 2, rows - 1]
            };
            for (b, x) in xs.iter().enumerate() {
                let cpu = cpu_q8_0_gemv(&w, x, rows, cols);
                let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
                for &r in &check_rows {
                    let diff = (gpu[b * rows + r] - cpu[r]).abs();
                    assert!(
                        diff <= 1e-4 * scale,
                        "({rows},{cols}) batch {batch} b {b} row {r}: {} vs {}",
                        gpu[b * rows + r],
                        cpu[r]
                    );
                }
            }
        }
    }
}

#[test]
fn batched_gemm_batch_one_is_bit_identical_to_the_single_token_gemv() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (rows, cols) = (512usize, 1024usize);
    let w = boundary_weights(rows, cols, 7);
    let x = boundary_input(cols, 9);
    let wb = ctx.buffer(w.len(), true).unwrap();
    let xb = ctx.buffer(cols * 4, true).unwrap();
    let y_single = ctx.buffer(rows * 4, true).unwrap();
    let y_batch = ctx.buffer(rows * 4, true).unwrap();
    wb.upload(0, &w).unwrap();
    xb.upload(0, &f32_bytes(&x)).unwrap();
    ctx.q8_0_gemv(&wb, &xb, &y_single, rows as u32, cols as u32)
        .unwrap();
    ctx.q8_0_gemm(&wb, &xb, &y_batch, rows as u32, cols as u32, 1, 0, 0)
        .unwrap();
    ctx.fence().unwrap();
    let mut a = vec![0u8; rows * 4];
    let mut b = vec![0u8; rows * 4];
    y_single.download(0, &mut a).unwrap();
    y_batch.download(0, &mut b).unwrap();
    assert_eq!(
        a, b,
        "batch=1 must be bit-identical to the single-token GEMV"
    );
}

#[test]
fn f16_gemm_matches_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (rows, cols) = (128usize, 256usize);
    let mut rng = Rng(31);
    let wf: Vec<f32> = (0..rows * cols)
        .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
        .collect();
    let w: Vec<u8> = wf
        .iter()
        .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
        .collect();
    let batch = 8usize;
    let xs: Vec<Vec<f32>> = (0..batch)
        .map(|b| boundary_input(cols, 41 + b as u64))
        .collect();
    let mut flat = Vec::new();
    for x in &xs {
        flat.extend_from_slice(x);
    }
    let wb = ctx.buffer(w.len(), true).unwrap();
    let xb = ctx.buffer(flat.len() * 4, true).unwrap();
    let yb = ctx.buffer(batch * rows * 4, true).unwrap();
    wb.upload(0, &w).unwrap();
    xb.upload(0, &f32_bytes(&flat)).unwrap();
    ctx.f16_gemm(&wb, &xb, &yb, rows as u32, cols as u32, batch as u32, 0, 0)
        .unwrap();
    ctx.fence().unwrap();
    let mut out = vec![0u8; batch * rows * 4];
    yb.download(0, &mut out).unwrap();
    let gpu: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (b, x) in xs.iter().enumerate() {
        for r in [0usize, rows / 2, rows - 1] {
            let row = &w[r * cols * 2..(r + 1) * cols * 2];
            let want: f32 = row
                .chunks_exact(2)
                .zip(x)
                .map(|(c, xv)| f16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())) * xv)
                .sum();
            let got = gpu[b * rows + r];
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "batch {b} row {r}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn gpu_time_is_accumulated_and_attributed_per_operator_when_profiling() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let n = 1u32 << 16;
    let a = ctx.buffer(n as usize * 4, true).unwrap();
    let y = ctx.buffer(n as usize * 4, true).unwrap();

    // Off by default: work runs, the total grows, nothing is attributed.
    let before = ctx.gpu_nanos_total();
    ctx.f32_axpy(&a, &y, 1.0, n).unwrap();
    ctx.fence().unwrap();
    assert!(ctx.gpu_nanos_total() >= before);
    assert!(ctx.profile_snapshot().is_empty());

    // On: every command buffer is attributed to its operator label, and
    // a dispatch group counts as one "group" buffer.
    ctx.set_profiling(true);
    for _ in 0..3 {
        ctx.f32_axpy(&a, &y, 1.0, n).unwrap();
    }
    ctx.begin_group().unwrap();
    ctx.f32_axpy(&a, &y, 1.0, n).unwrap();
    ctx.f32_axpy(&a, &y, 1.0, n).unwrap();
    ctx.end_group().unwrap();
    ctx.fence().unwrap();
    let table = ctx.profile_snapshot();
    let count = |label: &str| {
        table
            .iter()
            .find(|(name, _, _)| name == label)
            .map(|(_, _, c)| *c)
            .unwrap_or(0)
    };
    assert_eq!(count("f32_axpy"), 3, "{table:?}");
    assert_eq!(count("group"), 1, "{table:?}");

    // Disabling clears the table.
    ctx.set_profiling(false);
    assert!(ctx.profile_snapshot().is_empty());
}

/// Phase 1C measurement (opt-in, `TUCKMILL_PROFILE=1`): GPU time and
/// effective weight bandwidth of the Q8_0 GEMV on the pinned Qwen3.5-0.8B
/// matrix shapes. Prints only; asserts nothing about speed.
#[test]
fn q8_0_gemv_shape_bandwidth_when_requested() {
    if std::env::var("TUCKMILL_PROFILE").ok().as_deref() != Some("1") {
        eprintln!("skipping: TUCKMILL_PROFILE=1 not set (measurement only)");
        return;
    }
    let Some(ctx) = metal_context() else {
        return;
    };
    // (rows, cols, calls per decoded token) for the pinned 0.8B.
    let shapes = [
        (248_320usize, 1024usize, 1usize),
        (6144, 1024, 18),
        (4096, 1024, 6),
        (3584, 1024, 48),
        (2048, 1024, 18),
        (1024, 3584, 24),
        (1024, 2048, 24),
        (512, 1024, 12),
        (16, 1024, 36),
    ];
    let mut per_token_us = 0.0f64;
    for (rows, cols, calls) in shapes {
        let w = boundary_weights(rows, cols, 3);
        let x = boundary_input(cols, 5);
        let wb = ctx.buffer(w.len(), true).unwrap();
        wb.upload(0, &w).unwrap();
        let xb = ctx.buffer(cols * 4, true).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        let yb = ctx.buffer(rows * 4, true).unwrap();
        for _ in 0..3 {
            ctx.q8_0_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
                .unwrap();
        }
        ctx.fence().unwrap();
        let iters = 30u32;
        let g0 = ctx.gpu_nanos_total();
        for _ in 0..iters {
            ctx.q8_0_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
                .unwrap();
        }
        ctx.fence().unwrap();
        let us = (ctx.gpu_nanos_total() - g0) as f64 / 1e3 / iters as f64;
        let gbs = w.len() as f64 / (us * 1e-6) / 1e9;
        per_token_us += us * calls as f64;
        eprintln!(
            "q8_0_gemv {rows:>6}x{cols:<5} {us:>9.1} us  {gbs:>6.1} GB/s  x{calls:<3} = {:>8.1} us/token",
            us * calls as f64
        );
    }
    eprintln!("q8_0_gemv total per token: {:.2} ms", per_token_us / 1e3);
}

#[test]
fn copy_f32_moves_exact_slices_and_refuses_out_of_range() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let src: Vec<f32> = (0..64)
        .map(|i| (i as f32).sin() * 1e-30 + i as f32)
        .collect();
    let a = ctx.buffer(64 * 4, true).unwrap();
    let b = ctx.buffer(64 * 4, true).unwrap();
    a.upload(0, &f32_bytes(&src)).unwrap();
    ctx.copy_f32(&a, 8, &b, 40, 16).unwrap();
    ctx.fence().unwrap();
    let mut out = vec![0u8; 64 * 4];
    b.download(0, &mut out).unwrap();
    let src_bytes = f32_bytes(&src);
    assert_eq!(
        &out[40 * 4..56 * 4],
        &src_bytes[8 * 4..24 * 4],
        "bit-exact slice"
    );
    assert!(
        out[..40 * 4].iter().all(|v| *v == 0),
        "nothing outside the slice"
    );
    assert!(
        ctx.copy_f32(&a, 60, &b, 0, 8).is_err(),
        "source overrun refused"
    );
    assert!(
        ctx.copy_f32(&a, 0, &b, 60, 8).is_err(),
        "destination overrun refused"
    );
    assert!(
        ctx.copy_f32(&a, 0, &b, 0, 0).is_err(),
        "zero length refused"
    );
}

/// Phase 1C measurement (opt-in, `TUCKMILL_PROFILE=1`): batched Q8_0 GEMM
/// time per shape at prefill batch sizes, against the single-token GEMV.
#[test]
fn q8_0_gemm_batch_timing_when_requested() {
    if std::env::var("TUCKMILL_PROFILE").ok().as_deref() != Some("1") {
        eprintln!("skipping: TUCKMILL_PROFILE=1 not set (measurement only)");
        return;
    }
    let Some(ctx) = metal_context() else {
        return;
    };
    // `TUCKMILL_GEMM_MMA=1` times the matrix-unit kernel at prefill chunk
    // sizes instead of the per-row GEMM.
    let mma = std::env::var("TUCKMILL_GEMM_MMA").is_ok();
    let batches: &[usize] = if mma { &[32, 128] } else { &[1, 4, 8, 16] };
    for (rows, cols) in [(3584usize, 1024usize), (6144, 1024), (1024, 3584)] {
        let w = boundary_weights(rows, cols, 3);
        let wb = ctx.buffer(w.len(), true).unwrap();
        wb.upload(0, &w).unwrap();
        for &batch in batches {
            let mma = mma && batch > 1;
            let x = boundary_input(cols * batch, 5);
            let xb = ctx.buffer(cols * batch * 4, true).unwrap();
            xb.upload(0, &f32_bytes(&x)).unwrap();
            let yb = ctx.buffer(rows * batch * 4, true).unwrap();
            let run = || {
                if mma {
                    ctx.q8_0_gemm_mma(&wb, &xb, &yb, rows as u32, cols as u32, batch as u32, 0, 0)
                        .unwrap()
                } else {
                    ctx.q8_0_gemm(&wb, &xb, &yb, rows as u32, cols as u32, batch as u32, 0, 0)
                        .unwrap()
                }
            };
            for _ in 0..20 {
                run();
            }
            ctx.fence().unwrap();
            let iters = 100u32;
            let g0 = ctx.gpu_nanos_total();
            for _ in 0..iters {
                run();
            }
            ctx.fence().unwrap();
            let us = (ctx.gpu_nanos_total() - g0) as f64 / 1e3 / iters as f64;
            let gflops = 2.0 * (rows * cols * batch) as f64 / (us * 1e-6) / 1e9;
            eprintln!(
                "q8_0_gemm {rows:>5}x{cols:<5} batch {batch:>2}: {us:>8.1} us  {:>6.1} us/token  {gflops:>6.0} GFLOP/s",
                us / batch as f64
            );
        }
    }
}

#[test]
fn q8_0_gemv_timing_when_requested() {
    if std::env::var("TUCKMILL_PROFILE").ok().as_deref() != Some("1") {
        eprintln!("skipping: TUCKMILL_PROFILE=1 not set (measurement only)");
        return;
    }
    let Some(ctx) = metal_context() else {
        return;
    };
    // `TUCKMILL_GEMV_SHAPES=rowsxcols,...` overrides the Gemma-decode
    // default list. Per-dispatch wall over 16 serial dispatches; the
    // printed number is the achieved DRAM-equivalent bandwidth on the
    // weight bytes (measurement only; ADR-011 engine note records runs).
    let shapes = std::env::var("TUCKMILL_GEMV_SHAPES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|p| {
                    let (r, c) = p.split_once('x').expect("rowsxcols");
                    (
                        r.trim().parse::<usize>().expect("rows"),
                        c.trim().parse::<usize>().expect("cols"),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            vec![
                (6912usize, 1152usize),
                (1152, 6912),
                (1024, 1152),
                (1152, 1024),
                (256, 1152),
                (262144, 1152),
                (3584, 1024),
                (1024, 3584),
            ]
        });
    for (rows, cols) in shapes {
        let w = boundary_weights(rows, cols, 3);
        let x = boundary_input(cols, 5);
        let wb = ctx.buffer(w.len(), true).unwrap();
        wb.upload(0, &w).unwrap();
        let xb = ctx.buffer(cols * 4, true).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        let yb = ctx.buffer(rows * 4, true).unwrap();
        let group = || {
            ctx.begin_group().unwrap();
            for _ in 0..16 {
                ctx.q8_0_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
                    .unwrap();
            }
            ctx.end_group().unwrap();
            ctx.fence().unwrap();
        };
        for _ in 0..3 {
            group();
        }
        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            group();
        }
        let per = t0.elapsed().as_secs_f64() / (iters * 16) as f64;
        let wbytes = (rows * cols / 32 * 34) as f64;
        eprintln!(
            "ours gemv {rows:>6}x{cols:<5}: {per:>8.1} us  {gbw:>6.1} GB/s on weights",
            gbw = wbytes / per / 1e9
        );
    }
}

#[test]
fn split_k_attention_matches_the_cpu_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // The split-K pair against a scalar CPU reference, across the
    // engine's geometries: the tiny fixture (head_dim 16, where EPL is 1
    // and lanes past HD must contribute zero), the gemma3 fixture shape
    // with its window, dense-Qwen3 and Qwen3.5 shapes, and the real
    // Gemma 3 decode geometry with the sliding window. Covers decode
    // rows (n = 1), multi-row chunks, full chunks at pos 0, and
    // mid-context positions.
    let mut rng = Rng(97);
    let mut worst_all = 0f32;
    for (heads, kv_heads, hd, cap, pos, n, window) in [
        (2u32, 1u32, 16u32, 8u32, 5u32, 1u32, 0u32),
        (2, 1, 16, 8, 5, 1, 8),
        (4, 2, 16, 32, 0, 5, 8),
        (4, 2, 16, 144, 0, 8, 8),
        (4, 2, 16, 144, 0, 16, 8),
        (4, 2, 16, 144, 60, 1, 8),
        (4, 2, 128, 64, 33, 1, 0),
        (4, 2, 128, 48, 40, 5, 0),
        (8, 4, 128, 160, 80, 32, 0),
        (4, 1, 256, 64, 33, 1, 0),
        (4, 1, 256, 1024, 530, 1, 512),
        (4, 1, 256, 1024, 530, 128, 512),
        (4, 1, 256, 1024, 896, 128, 512),
    ] {
        let group = heads / kv_heads;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..(n as usize) * (heads as usize) * (hd as usize))
            .map(|_| ((rng.next() % 2001) as f32 - 1000.0) / 1000.0)
            .collect();
        let k: Vec<f32> = (0..(cap as usize) * (kv_heads as usize) * (hd as usize))
            .map(|_| ((rng.next() % 2001) as f32 - 1000.0) / 1000.0)
            .collect();
        let v: Vec<f32> = (0..(cap as usize) * (kv_heads as usize) * (hd as usize))
            .map(|_| ((rng.next() % 2001) as f32 - 1000.0) / 1000.0)
            .collect();
        let f16 = |x: &[f32]| -> Vec<u8> {
            x.iter()
                .flat_map(|f| f32_to_f16_bits(*f).to_le_bytes())
                .collect()
        };
        let qb = ctx.buffer(q.len() * 4, true).unwrap();
        qb.upload(0, &f32_bytes(&q)).unwrap();
        let kc = ctx.buffer(k.len() * 2, true).unwrap();
        kc.upload(0, &f16(&k)).unwrap();
        let vc = ctx.buffer(v.len() * 2, true).unwrap();
        vc.upload(0, &f16(&v)).unwrap();
        let out = ctx.buffer(q.len() * 4, true).unwrap();
        let scratch = ctx
            .buffer(
                (n as usize)
                    * (heads as usize)
                    * ATTENTION_SK_SLICES as usize
                    * (hd as usize + 2)
                    * 4,
                true,
            )
            .unwrap();
        ctx.qwen35_attention_chunk(
            &qb, &kc, &vc, &out, &scratch, heads, pos, cap, hd, group, scale, n, window, false,
        )
        .unwrap();
        ctx.fence().unwrap();
        let mut got = vec![0u8; q.len() * 4];
        out.download(0, &mut got).unwrap();
        let got: Vec<f32> = got
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let kf: Vec<f32> = k
            .iter()
            .map(|x| f16_bits_to_f32(f32_to_f16_bits(*x)))
            .collect();
        let vf: Vec<f32> = v
            .iter()
            .map(|x| f16_bits_to_f32(f32_to_f16_bits(*x)))
            .collect();
        let mut worst = 0f32;
        for t in 0..n as usize {
            let seq = pos as usize + t + 1;
            let start = if window == 0 {
                0
            } else {
                seq.saturating_sub(window as usize)
            };
            for h in 0..heads as usize {
                let kv_head = h / group as usize;
                let mut sc = Vec::new();
                for j in start..seq {
                    let mut dota = 0f32;
                    for d in 0..hd as usize {
                        dota += q[(t * heads as usize + h) * hd as usize + d]
                            * kf[(kv_head * cap as usize + j) * hd as usize + d];
                    }
                    sc.push(dota * scale);
                }
                let m = sc.iter().copied().fold(-f32::INFINITY, f32::max);
                let l: f32 = sc.iter().map(|x| (x - m).exp()).sum();
                for d in 0..hd as usize {
                    let mut acc = 0f32;
                    for (i, j) in (start..seq).enumerate() {
                        acc += ((sc[i] - m).exp() / l)
                            * vf[(kv_head * cap as usize + j) * hd as usize + d];
                    }
                    let g = got[(t * heads as usize + h) * hd as usize + d];
                    worst = worst.max((g - acc).abs());
                }
            }
        }
        worst_all = worst_all.max(worst);
        eprintln!(
            "split-K ({heads},{kv_heads},{hd},{cap},{pos},{n},w{window}): worst |diff| {worst:.3e}"
        );
        // Registered: 1e-6 x max|out| (measured worst across this set is
        // printed at the end; pinned from the run).
        assert!(
            worst <= 1e-6,
            "({heads},{kv_heads},{hd},{cap},{pos},{n},w{window}): worst {worst:.3e}"
        );
    }
    eprintln!("split-K set worst: {worst_all:.3e}");

    // Cross-kernel drift: the same absolute position served by the
    // decode path (n = 1, split-K) vs the chunk path (n = 5, the
    // one-group-per-row kernel). The two reduction orders differ by
    // design (decode parallelism vs chunk cost); this measures the
    // logits-level drift that the engine's cross-mode gates register.
    let (heads, kv_heads, hd, cap, window) = (4u32, 2u32, 16u32, 144u32, 8u32);
    let group = heads / kv_heads;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let q: Vec<f32> = (0..(8 * heads as usize * hd as usize))
        .map(|_| ((rng.next() % 2001) as f32 - 1000.0) / 1000.0)
        .collect();
    let k: Vec<f32> = (0..(cap as usize) * (kv_heads as usize) * (hd as usize))
        .map(|_| ((rng.next() % 2001) as f32 - 1000.0) / 1000.0)
        .collect();
    let v: Vec<f32> = (0..(cap as usize) * (kv_heads as usize) * (hd as usize))
        .map(|_| ((rng.next() % 2001) as f32 - 1000.0) / 1000.0)
        .collect();
    let f16 = |x: &[f32]| -> Vec<u8> {
        x.iter()
            .flat_map(|f| f32_to_f16_bits(*f).to_le_bytes())
            .collect()
    };
    let qb = ctx.buffer(q.len() * 4, true).unwrap();
    qb.upload(0, &f32_bytes(&q)).unwrap();
    let kc = ctx.buffer(k.len() * 2, true).unwrap();
    kc.upload(0, &f16(&k)).unwrap();
    let vc = ctx.buffer(v.len() * 2, true).unwrap();
    vc.upload(0, &f16(&v)).unwrap();
    let out_chunk = ctx.buffer(q.len() * 4, true).unwrap();
    let out_rows = ctx.buffer(heads as usize * hd as usize * 4, true).unwrap();
    let scratch = ctx
        .buffer(
            8 * heads as usize * ATTENTION_SK_SLICES as usize * (hd as usize + 2) * 4,
            true,
        )
        .unwrap();
    ctx.qwen35_attention_chunk(
        &qb, &kc, &vc, &out_chunk, &scratch, heads, 20, cap, hd, group, scale, 5, window, false,
    )
    .unwrap();
    ctx.fence().unwrap();
    let mut chunk_bytes = vec![0u8; q.len() * 4];
    out_chunk.download(0, &mut chunk_bytes).unwrap();
    let qrows = ctx.buffer(heads as usize * hd as usize * 4, true).unwrap();
    let mut worst_drift = 0f32;
    let mut top1_mismatch = 0u32;
    for (i, tpos) in (20u32..25u32).enumerate() {
        qrows
            .upload(0, &f32_bytes(&q[i * 4 * 16..(i + 1) * 4 * 16]))
            .unwrap();
        ctx.qwen35_attention_chunk(
            &qrows, &kc, &vc, &out_rows, &scratch, heads, tpos, cap, hd, group, scale, 1, window,
            false,
        )
        .unwrap();
        ctx.fence().unwrap();
        let mut row_bytes = vec![0u8; heads as usize * hd as usize * 4];
        out_rows.download(0, &mut row_bytes).unwrap();
        let chunk_row = &chunk_bytes
            [i * heads as usize * hd as usize * 4..(i + 1) * heads as usize * hd as usize * 4];
        let row_f: Vec<f32> = row_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let chunk_f: Vec<f32> = chunk_row
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let d = row_f
            .iter()
            .zip(&chunk_f)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        worst_drift = worst_drift.max(d);
        let top1 = |x: &[f32]| {
            x.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        if top1(&row_f) != top1(&chunk_f) {
            top1_mismatch += 1;
        }
    }
    eprintln!(
        "cross-kernel drift (chunk vs decode, 5 positions): max|diff| {worst_drift:.3e}, top-1 mismatches {top1_mismatch}"
    );
    assert!(
        worst_drift <= 1e-5 && top1_mismatch == 0,
        "cross-kernel drift {worst_drift:.3e}, top-1 mismatches {top1_mismatch}"
    );
}

#[test]
fn attention_decode_timing_when_requested() {
    if std::env::var("TUCKMILL_PROFILE").ok().as_deref() != Some("1") {
        eprintln!("skipping: TUCKMILL_PROFILE=1 not set (measurement only)");
        return;
    }
    let Some(ctx) = metal_context() else {
        return;
    };
    // Gemma 3 1B decode geometry: 4 heads, 1 KV head, head_dim 256,
    // f16 KV, cap 1024. One decode query at pos 530 with the sliding
    // window (512) and full causal, plus the prefill-chunk call for
    // reference.
    let heads = 4u32;
    let kv_heads = 1u32;
    let hd = 256u32;
    let cap = 1024u32;
    let group = heads / kv_heads;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let qb = (heads * hd * 4) as usize;
    let plane = (kv_heads * cap * hd * 2) as usize;
    let q = ctx.buffer(qb, true).unwrap();
    let kc = ctx.buffer(plane, true).unwrap();
    let vc = ctx.buffer(plane, true).unwrap();
    let out = ctx.buffer(qb, true).unwrap();
    // Split-K partials: rows x heads x slices x (head_dim + 2) f32, sized
    // for the n = 128 prefill-chunk case.
    let scratch = ctx
        .buffer(
            (128usize * heads as usize * ATTENTION_SK_SLICES as usize * (hd as usize + 2)) * 4,
            true,
        )
        .unwrap();
    let data = f32_bytes(&vec![0.25f32; (heads * hd) as usize]);
    q.upload(0, &data).unwrap();
    let kv16 = vec![0u8; plane];
    kc.upload(0, &kv16).unwrap();
    vc.upload(0, &kv16).unwrap();
    for (pos, window, n, label) in [
        (530u32, 512u32, 1u32, "decode windowed"),
        (530, 0, 1, "decode full   "),
        (530, 512, 128, "chunk n=128  "),
    ] {
        let group_fn = || {
            ctx.begin_group().unwrap();
            for _ in 0..16 {
                ctx.qwen35_attention_chunk(
                    &q, &kc, &vc, &out, &scratch, heads, pos, cap, hd, group, scale, n, window,
                    false,
                )
                .unwrap();
            }
            ctx.end_group().unwrap();
            ctx.fence().unwrap();
        };
        for _ in 0..3 {
            group_fn();
        }
        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            group_fn();
        }
        let per = t0.elapsed().as_secs_f64() / (iters * 16) as f64;
        eprintln!(
            "attention {label} (pos {pos}, w {window}): {us:8.1} us/call",
            us = per * 1e6
        );
    }
}

#[test]
fn q8_0_gemm_mma_matches_the_cpu_reference_within_tolerance() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // Shapes cover row tails (rows not a multiple of the 32-row tile),
    // partial token tiles, offsets, and the real 0.8B projection shapes.
    for (rows, cols, batch) in [
        (16usize, 1024usize, 16usize),
        (40, 64, 3),
        (3584, 1024, 16),
        (3584, 1024, 32),
        (40, 64, 27),
        (1024, 3584, 64),
        (100, 96, 77),
        (40, 64, 512),
        (1024, 3584, 7),
        (6144, 1024, 1),
    ] {
        let w = boundary_weights(rows, cols, 11);
        let x = boundary_input(cols * batch, 13);
        let wb = ctx.buffer(w.len(), true).unwrap();
        wb.upload(0, &w).unwrap();
        let x_off = 32usize;
        let xb = ctx.buffer((x_off + cols * batch) * 4, true).unwrap();
        let mut xs = vec![0f32; x_off];
        xs.extend_from_slice(&x);
        xb.upload(0, &f32_bytes(&xs)).unwrap();
        let y_off = 8usize;
        let yb = ctx.buffer((y_off + rows * batch) * 4, true).unwrap();
        ctx.q8_0_gemm_mma(
            &wb,
            &xb,
            &yb,
            rows as u32,
            cols as u32,
            batch as u32,
            x_off as u32,
            y_off as u32,
        )
        .unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; (y_off + rows * batch) * 4];
        yb.download(0, &mut out).unwrap();
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(
            got[..y_off].iter().all(|v| *v == 0.0),
            "nothing before y_off"
        );
        for b in 0..batch {
            let cpu = cpu_q8_0_gemv(&w, &x[b * cols..(b + 1) * cols], rows, cols);
            let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            for r in 0..rows {
                let g = got[y_off + b * rows + r];
                assert!(
                    (g - cpu[r]).abs() <= 1e-4 * scale,
                    "({rows},{cols},{batch}) b {b} r {r}: {g} vs {}",
                    cpu[r]
                );
            }
        }
    }
    assert!(
        ctx.q8_0_gemm_mma(
            &ctx.buffer(64, true).unwrap(),
            &ctx.buffer(64, true).unwrap(),
            &ctx.buffer(64, true).unwrap(),
            1,
            32,
            GEMM_MMA_MAX_BATCH + 1,
            0,
            0
        )
        .is_err(),
        "batch above the MMA maximum refused"
    );
}

#[test]
fn q8_0_gemm_mma_f16op_matches_the_f32_reference_within_tolerance() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // The metal-f16-act profile (ADR-010): operands round to half when
    // staged, accumulation stays f32. Data is the half-domain boundary
    // set (see half_domain_weights). Shapes cover row tails, partial
    // token tiles, offsets and the real 2B / Gemma 3 projection shapes.
    // Registered bound 1e-4 x max|reference| (observed 2.9e-6 across these
    // shapes, 2026-11-21, M3); whole-model envelopes qualify the profile
    // on real weights.
    let mut worst_rel = 0f32;
    let mut worst_at = (0usize, 0usize, 0usize, 0usize);
    for (rows, cols, batch) in [
        (16usize, 1024usize, 16usize),
        (40, 64, 3),
        (6144, 2048, 32),
        (2048, 2048, 27),
        (2048, 6144, 64),
        (6912, 1152, 128),
        (1152, 6912, 77),
        (1024, 1152, 512),
        (1152, 1024, 7),
        (256, 1152, 1),
    ] {
        let w = half_domain_weights(rows, cols, 11);
        let x = boundary_input(cols * batch, 13);
        let wb = ctx.buffer(w.len(), true).unwrap();
        wb.upload(0, &w).unwrap();
        let x_off = 32usize;
        let xb = ctx.buffer((x_off + cols * batch) * 4, true).unwrap();
        let mut xs = vec![0f32; x_off];
        xs.extend_from_slice(&x);
        xb.upload(0, &f32_bytes(&xs)).unwrap();
        let y_off = 8usize;
        let yb = ctx.buffer((y_off + rows * batch) * 4, true).unwrap();
        ctx.q8_0_gemm_mma_f16op(
            &wb,
            &xb,
            &yb,
            rows as u32,
            cols as u32,
            batch as u32,
            x_off as u32,
            y_off as u32,
        )
        .unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; (y_off + rows * batch) * 4];
        yb.download(0, &mut out).unwrap();
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(
            got[..y_off].iter().all(|v| *v == 0.0),
            "nothing before y_off"
        );
        for b in 0..batch {
            let cpu = cpu_q8_0_gemv(&w, &x[b * cols..(b + 1) * cols], rows, cols);
            let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            for r in 0..rows {
                let g = got[y_off + b * rows + r];
                let rel = (g - cpu[r]).abs() / scale;
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_at = (rows, cols, b, r);
                }
            }
        }
    }
    eprintln!(
        "f16op worst rel diff vs f32 reference: {worst_rel:.3e} at (rows, batch, row) {worst_at:?}"
    );
    assert!(
        worst_rel <= 1e-4,
        "metal-f16-act drift beyond the registered bound: {worst_rel:.3e}"
    );
    assert!(
        ctx.q8_0_gemm_mma_f16op(
            &ctx.buffer(64, true).unwrap(),
            &ctx.buffer(64, true).unwrap(),
            &ctx.buffer(64, true).unwrap(),
            1,
            32,
            GEMM_MMA_MAX_BATCH + 1,
            0,
            0
        )
        .is_err(),
        "batch above the MMA maximum refused"
    );
}

#[test]
fn argmax_matches_the_cpu_greedy_rule_exactly() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // The CPU rule (tuckmill_backend::sampling::argmax), restated here so
    // the metal crate keeps no backend dependency: NaN skipped, strict >,
    // so ties go to the lowest index.
    fn cpu(x: &[f32]) -> u32 {
        let mut best: Option<(usize, f32)> = None;
        for (i, v) in x.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            match best {
                None => best = Some((i, *v)),
                Some((_, bv)) if *v > bv => best = Some((i, *v)),
                _ => {}
            }
        }
        best.map(|(i, _)| i as u32).unwrap_or(ARGMAX_NONE)
    }
    let mut rng = Rng(0xA11CE);
    let mut cases: Vec<Vec<f32>> = vec![
        vec![1.0, 3.0, 2.0],
        vec![3.0, 3.0],
        vec![f32::NAN, 1.0],
        vec![f32::NAN, f32::INFINITY],
        vec![f32::NAN, f32::NAN, f32::NAN],
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
        vec![-0.0, 0.0],
        vec![0.0, -0.0],
        vec![7.5],
    ];
    // Vocabulary-sized random logits with the maximum planted several
    // times across different threadgroup slices (ties resolve low).
    let n = 248_320usize;
    let mut big: Vec<f32> = (0..n)
        .map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0)
        .collect();
    for k in [200_000usize, 77_777, 131_071, 77_776] {
        big[k] = 42.0;
    }
    big[5] = f32::NAN;
    cases.push(big);
    // Sizes around the partial/thread geometry, ties at the far end.
    for len in [255usize, 256, 16_384, 16_385, 70_001] {
        let mut v = vec![1.0f32; len];
        v[len - 1] = 2.0;
        v[len / 2] = 2.0;
        cases.push(v);
    }
    let partial = ctx.buffer(ARGMAX_PARTIAL_BYTES, true).unwrap();
    let out = ctx.buffer(4, true).unwrap();
    for (c, x) in cases.iter().enumerate() {
        let xb = ctx.buffer(x.len() * 4, true).unwrap();
        xb.upload(0, &f32_bytes(x)).unwrap();
        ctx.argmax_f32(&xb, x.len() as u32, &partial, &out, 0)
            .unwrap();
        ctx.fence().unwrap();
        let mut b = [0u8; 4];
        out.download(0, &mut b).unwrap();
        assert_eq!(u32::from_le_bytes(b), cpu(x), "case {c} (len {})", x.len());
    }
    let empty = ctx.buffer(4, true).unwrap();
    assert!(
        ctx.argmax_f32(&empty, 0, &partial, &out, 0).is_err(),
        "empty input refused"
    );
    assert!(
        ctx.argmax_f32(&empty, 1, &partial, &out, 1).is_err(),
        "out_index past the buffer refused"
    );

    // The indirect embedding lookup reads the row id the argmax wrote.
    let (rows, cols) = (7usize, 64usize);
    let w = boundary_weights(rows, cols, 21);
    let wb = ctx.buffer(w.len(), true).unwrap();
    wb.upload(0, &w).unwrap();
    let ids = ctx.buffer(8, true).unwrap();
    let row_out = ctx.buffer(cols * 4, true).unwrap();
    for (slot, id) in [(0u32, 3u32), (1, 6), (1, ARGMAX_NONE)] {
        ids.upload(slot as usize * 4, &id.to_le_bytes()).unwrap();
        ctx.row_indirect_f32(&wb, 8, rows as u32, cols as u32, &ids, slot, &row_out)
            .unwrap();
        ctx.fence().unwrap();
        let mut b = vec![0u8; cols * 4];
        row_out.download(0, &mut b).unwrap();
        let got: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let want: Vec<f32> = if id == ARGMAX_NONE {
            vec![0.0; cols]
        } else {
            let blocks = cols / 32;
            (0..cols)
                .map(|c| {
                    let blk = &w[(id as usize * blocks + c / 32) * 34..];
                    let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    d * blk[2 + c % 32] as i8 as f32
                })
                .collect()
        };
        assert_eq!(got, want, "slot {slot} id {id}");
    }
}

/// Measurement only (`TUCKMILL_PROFILE=1`): the matrix-unit GEMM timed the
/// way ggml's graph runs it, 16 dispatches in one command buffer, as wall
/// time per dispatch.
#[test]
fn q8_0_gemm_mma_grouped_wall_timing_when_requested() {
    if std::env::var("TUCKMILL_PROFILE").ok().as_deref() != Some("1") {
        eprintln!("skipping: TUCKMILL_PROFILE=1 not set (measurement only)");
        return;
    }
    let Some(ctx) = metal_context() else {
        return;
    };
    // `TUCKMILL_GEMM_SHAPES=rowsxcols,...` overrides the 0.8B default list
    // (measurement only; ADR-011 engine notes record the runs).
    let shapes = std::env::var("TUCKMILL_GEMM_SHAPES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|p| {
                    let (r, c) = p.split_once('x').expect("rowsxcols");
                    (
                        r.trim().parse::<usize>().expect("rows"),
                        c.trim().parse::<usize>().expect("cols"),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![(3584usize, 1024usize), (6144, 1024), (1024, 3584)]);
    for (rows, cols) in shapes {
        let w = boundary_weights(rows, cols, 3);
        let wb = ctx.buffer(w.len(), true).unwrap();
        wb.upload(0, &w).unwrap();
        for batch in [32usize, 128, 512] {
            let x = boundary_input(cols * batch, 5);
            let xb = ctx.buffer(cols * batch * 4, true).unwrap();
            xb.upload(0, &f32_bytes(&x)).unwrap();
            let yb = ctx.buffer(rows * batch * 4, true).unwrap();
            let group = |f16op: bool| {
                ctx.begin_group().unwrap();
                for _ in 0..16 {
                    if f16op {
                        ctx.q8_0_gemm_mma_f16op(
                            &wb,
                            &xb,
                            &yb,
                            rows as u32,
                            cols as u32,
                            batch as u32,
                            0,
                            0,
                        )
                        .unwrap();
                    } else {
                        ctx.q8_0_gemm_mma(
                            &wb,
                            &xb,
                            &yb,
                            rows as u32,
                            cols as u32,
                            batch as u32,
                            0,
                            0,
                        )
                        .unwrap();
                    }
                }
                ctx.end_group().unwrap();
                ctx.fence().unwrap();
            };
            for (f16op, label) in [(false, "f32  "), (true, "f16op")] {
                for _ in 0..3 {
                    group(f16op);
                }
                let iters = 20;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    group(f16op);
                }
                let per = t0.elapsed().as_secs_f64() / (iters * 16) as f64;
                eprintln!(
                    "ours mul_mat {label} {rows:>5}x{cols:<5} tokens {batch:>3}: {:>8.1} us  {:>6.0} GFLOP/s",
                    per * 1e6,
                    2.0 * (rows * cols * batch) as f64 / per / 1e9
                );
            }
        }
    }
}
