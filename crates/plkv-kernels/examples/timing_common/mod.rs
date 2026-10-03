//! Shared timing helpers for GPU examples: CUDA events on an explicit stream and an in-process
//! NVML reader (dlopen; no link-time dependency). Same rules as the fixed C1 harness: missing
//! NVML symbols are recorded as unavailable, never fabricated.
#![allow(dead_code)]

use cutile::cuda_core::{Stream, sys};
use std::sync::Arc;

pub const ABORT_TEMPERATURE_C: u32 = 91;

pub fn check_cuda(result: sys::CUresult, operation: &str) {
    assert_eq!(
        result,
        sys::cudaError_enum_CUDA_SUCCESS,
        "{operation} failed with CUDA result {result}"
    );
}

pub struct Events {
    start: sys::CUevent,
    stop: sys::CUevent,
}

impl Events {
    pub fn new() -> Self {
        let (mut s, mut e) = (std::ptr::null_mut(), std::ptr::null_mut());
        unsafe {
            check_cuda(sys::cuEventCreate(&mut s, 0), "event");
            check_cuda(sys::cuEventCreate(&mut e, 0), "event");
        }
        Self { start: s, stop: e }
    }

    /// Synchronizes, records start, runs `f` (which enqueues work), records stop, waits, and
    /// returns the elapsed milliseconds between the two events.
    pub fn measure<F: FnOnce()>(&self, st: &Arc<Stream>, f: F) -> f64 {
        unsafe {
            st.synchronize().unwrap();
            check_cuda(sys::cuEventRecord(self.start, st.cu_stream()), "record");
            f();
            check_cuda(sys::cuEventRecord(self.stop, st.cu_stream()), "record");
            check_cuda(sys::cuEventSynchronize(self.stop), "sync");
            let mut ms = 0.;
            check_cuda(
                sys::cuEventElapsedTime_v2(&mut ms, self.start, self.stop),
                "elapsed",
            );
            f64::from(ms)
        }
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuEventDestroy_v2(self.start);
            let _ = sys::cuEventDestroy_v2(self.stop);
        }
    }
}

pub struct Nvml {
    _lib: libloading::Library,
    device: *mut std::ffi::c_void,
    clock: unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32,
    temperature: unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32,
    reasons: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64) -> i32>,
}

impl Nvml {
    pub fn open() -> Option<Self> {
        unsafe {
            let lib = libloading::Library::new("libnvidia-ml.so.1").ok()?;
            let init: libloading::Symbol<unsafe extern "C" fn() -> i32> =
                lib.get(b"nvmlInit_v2\0").ok()?;
            if init() != 0 {
                return None;
            }
            let handle: libloading::Symbol<
                unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32,
            > = lib.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
            let mut device = std::ptr::null_mut();
            if handle(0, &mut device) != 0 {
                return None;
            }
            let clock = *lib.get(b"nvmlDeviceGetClockInfo\0").ok()?;
            let temperature = *lib.get(b"nvmlDeviceGetTemperature\0").ok()?;
            let reasons = lib
                .get(b"nvmlDeviceGetCurrentClocksEventReasons\0")
                .or_else(|_| lib.get(b"nvmlDeviceGetCurrentClocksThrottleReasons\0"))
                .ok()
                .map(|f| *f);
            Some(Self {
                _lib: lib,
                device,
                clock,
                temperature,
                reasons,
            })
        }
    }

    pub fn sm_clock(&self) -> Option<u32> {
        let mut v = 0;
        // NVML_CLOCK_SM = 1
        (unsafe { (self.clock)(self.device, 1, &mut v) } == 0).then_some(v)
    }

    pub fn mem_clock(&self) -> Option<u32> {
        let mut v = 0;
        // NVML_CLOCK_MEM = 2
        (unsafe { (self.clock)(self.device, 2, &mut v) } == 0).then_some(v)
    }

    pub fn temperature(&self) -> Option<u32> {
        let mut v = 0;
        (unsafe { (self.temperature)(self.device, 0, &mut v) } == 0).then_some(v)
    }

    pub fn reasons(&self) -> Option<u64> {
        let mut v = 0;
        self.reasons
            .and_then(|f| (unsafe { f(self.device, &mut v) } == 0).then_some(v))
    }
}

/// Clock telemetry captured around one timed sample.
pub fn clock_fields(nvml: Option<&Nvml>, pre: Option<u32>) -> serde_json::Value {
    serde_json::json!({
        "sm_clock_pre_mhz": pre,
        "sm_clock_post_mhz": nvml.and_then(Nvml::sm_clock),
        "mem_clock_post_mhz": nvml.and_then(Nvml::mem_clock),
        "temperature_c": nvml.and_then(Nvml::temperature),
        "clock_event_reasons": nvml.and_then(Nvml::reasons),
    })
}

/// Back-to-back untimed work for `warm_ms`, then the reference clock = mode of the second half
/// of the NVML reads taken after each synchronized burst.
pub fn clock_warmup<F: FnMut()>(
    nvml: Option<&Nvml>,
    st: &Arc<Stream>,
    warm_ms: u64,
    mut burst: F,
) -> (Option<u32>, Vec<u32>) {
    let mut reads = Vec::new();
    let start = std::time::Instant::now();
    while start.elapsed().as_millis() < u128::from(warm_ms) {
        burst();
        unsafe {
            st.synchronize().unwrap();
        }
        if let Some(c) = nvml.and_then(Nvml::sm_clock) {
            reads.push(c);
        }
    }
    let tail = &reads[reads.len() / 2..];
    let mut counts = std::collections::BTreeMap::new();
    for c in tail {
        *counts.entry(*c).or_insert(0usize) += 1;
    }
    (
        counts.into_iter().max_by_key(|(_, k)| *k).map(|(c, _)| c),
        reads,
    )
}

/// Writes 0 to `count` 32-bit words at `dptr` on `st` (L2 flush helper; synchronizes).
pub fn memset_d32(dptr: sys::CUdeviceptr, count: usize, st: &Arc<Stream>) {
    unsafe {
        check_cuda(
            sys::cuMemsetD32Async(dptr, 0, count, st.cu_stream()),
            "memset",
        );
        st.synchronize().unwrap();
    }
}
