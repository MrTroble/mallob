# GPAP on MGI (initial version)

[GPAP](https://github.com/yogevshalmon/gpap) is a GPU library for mining entailed clauses: it
probes many sets of literals at once with unit propagation on the GPU and reports the ones that
conflict. This branch runs GPAP's engine on mallob's GPU layer, MGI, so that GPAP's GPU work goes
through the same `mgi::DeferredAPI` as the rest of mallob.

**Status: initial version.** GPAP's engine runs on MGI. On every formula we compared (7 instances
and a small known-answer formula, probing single literals and pairs), it finds exactly the clauses
it finds on its own OpenCL plug. No mallob job uses it yet; that is the next step to discuss
(below).

## How it fits

```
mallob process
 └─ mgi::DeferredAPI (OpenCL)
     ├─ mallob's own MGI kernels
     └─ MgiGpu  ←  GPAP engine (probing, clause mining)
```

GPAP reaches the GPU through one small interface, `gpap::Gpu` (GPAP's `<gpap/gpu_api.hpp>`):
allocate, copy, read back, run a kernel. GPAP ships a CUDA and an OpenCL implementation of it.
`MgiGpu` ([src/app/sat/gpap/mgi_gpu.hpp](../src/app/sat/gpap/mgi_gpu.hpp)) is a third one, on
MGI: it loads GPAP's OpenCL kernels (`bcp.cl`) with MGI's kernel loader and runs them as MGI
tasks, by GPAP's launch rules (`<gpap/opencl_launch.hpp>`). A host hands its plug to
`gpap::engine_create_on`; GPAP needs nothing else from mallob.

## Build and run

GPAP comes in with CMake `FetchContent` (CMake 3.18 or later), built without CUDA as a static
library. Until the GPAP side is merged into GPAP's `main`, the build pins a commit of GPAP's
`mallob-mgi` branch; `-DFETCHCONTENT_SOURCE_DIR_GPAP=<GPAP checkout>` builds against a local GPAP
instead.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DMALLOB_USE_GPU=1 -DMALLOB_USE_GPAP=1 -DMALLOB_ASSERT=1 \
      -DMALLOB_SUBPROC_DISPATCH_PATH=\"$PWD/build/\"
cmake --build build -j --target test_gpap_mgi
build/test_gpap_mgi            # from the repository root; --compare also times both plugs
```

`MALLOB_ASSERT=1` is required (see the caveats). `MALLOB_SUBPROC_DISPATCH_PATH` must be the build
directory: CMake copies MGI's and GPAP's kernels to `<build>/mgi_kernel` and `<build>/gpap_kernel`,
and MGI loads them from `<dispatch path>/...` at run time.

On WSL2, where NVIDIA ships no OpenCL, we ran on PoCL's CUDA driver, installed with GPAP's own
script: in a GPAP clone (for example `build/_deps/gpap-src`), run `scripts/setup_opencl.sh cuda`
once, then `source <(scripts/setup_opencl.sh env)` in the shell that runs the test.

[src/test/test_gpap_mgi.cpp](../src/test/test_gpap_mgi.cpp) checks `MgiGpu` on its own
(transfers larger than its bounce buffer, offsets, kernel launches), known answers, and then GPAP
on `MgiGpu` against GPAP on its own OpenCL plug, on 7 instances from `instances/`: the same unit
and binary clauses (689, 567 and 5451 binary clauses, 24 units on the structured ones), on both
of GPAP's propagation kernels.

## Where it ran

| Machine | OpenCL | `test_gpap_mgi` |
|---|---|---|
| L4 (Google Colab) | NVIDIA, driver 580.82 | passed (2026-11-23) |
| RTX 4080 Laptop (WSL2) | PoCL 7.2 over CUDA | passed (2026-11-23) |
| A100-SXM4-40GB (Technion cluster) | NVIDIA, driver 595.71 | passed (2026-11-20, an earlier revision) |

On the L4, the same probe sweeps through both plugs (`--compare`: same tuples, same batch, best
of 5 runs):

| Formula | Clauses | MGI time / GPAP's own OpenCL time |
|---|---|---|
| entertainment08-1, towers05-10, roverg10-1 | 0.7k–2.2k | 1.06 |
| roverg10-2 | 45k | 1.26 |
| r3unknown_10k | 43k | 1.41 |

MGI compiled `bcp.cl` with its debug flags (`-cl-nv-opt-level=0`, see the caveats), GPAP's own plug
with the driver's default optimisation. So the gap on the large formulas mixes `MgiGpu`'s
per-launch synchronisation and copies with unoptimised kernels; we have not measured which is
which. Creating an engine on `MgiGpu` takes about 0.5 s longer, because MGI compiles `bcp.cl` for
each engine.

## Caveats

- **Assert builds only.** `MGI_DB_CHECK` compiles MGI's OpenCL calls out unless `DEBUG` is defined,
  and only `MALLOB_ASSERT` defines it. `MALLOB_USE_GPAP` refuses to configure without it, and
  `mgi_gpu.cpp` refuses to compile. With `DEBUG`, MGI also compiles kernels on NVIDIA with
  `-cl-nv-opt-level=0`.
- **`MgiGpu` works around MGI's behaviour instead of changing it:**
  - MGI's queue is out-of-order and its transfers do not wait for queued tasks, so every launch
    waits for the previous one and every transfer waits for the last launch;
  - MGI's reads and writes of device-local memory leave a staging buffer behind each time, so host
    transfers go through one persistent 8 MiB host-visible buffer;
  - MGI aborts the process on a failed allocation, so `MgiGpu` refuses buffers past its own limit
    (its memory budget, at most 1 GiB) with an error GPAP can report.
- **Leaks.** `freeObj` retains instead of releasing (`clRetain*`), so buffers, events and cloned
  kernels are never freed. Every buffer an engine frees stays allocated: with the test's 1 GiB
  budget, up to about 220 MB of row buffers per engine, plus its parameter blocks and argument
  copies. That is fine for the test, but a long-running worker would run the GPU out of memory,
  and MGI aborts on a failed allocation. Not fixed here.
- **Memory budget.** `MgiGpu` reports a fixed budget, and GPAP's batch sizing still subtracts a
  device-wide reserve from it: [GPAP issue 72](https://github.com/yogevshalmon/gpap/issues/72).
- **Not covered:** mallob's own MGI resolution worker. Its part of `test_mgi_base` hangs on PoCL
  (the task never leaves `CL_SUBMITTED`); it passes the API part.

## Changes to MGI

Four, only so that the OpenCL backend builds and runs here. On NVIDIA, MGI behaves as before.

1. **The OpenCL backend compiles again.** `DeferredAPIOCL.hpp` used `KernelLoaderOCL` (since
   renamed) and the one-argument `slabs()`/`regions()` (now two arguments).
2. **The kernel files reach the build directory.** Without `MALLOB_MGI_TEST_COMPILE`, the loader
   still reads `src/mgi_kernel/*` (the kernels and `MGIKernelDefs.hpp`, `MGIShared.hpp`) from
   `<dispatch path>/mgi_kernel` at run time, so CMake now copies them there.
3. **Compiler options per device** (`KernelLoader.cpp`). NVIDIA keeps MGI's options. Elsewhere the
   NVIDIA-only flags are dropped, and `-cl-std` names an OpenCL C version the device supports
   (PoCL's CUDA driver has 1.2 and 3.0, not 2.0).
4. **In-order queue fallback** (`DeferredAPIOCL.hpp`). A device that refuses out-of-order queues
   (PoCL) gets an in-order one instead of failing.

## Next steps to discuss

1. **Where GPAP runs in mallob.** Our plan: a worker in a SAT job that uses the job's MGI instance,
   mines clauses with GPAP, and hands them to mallob's clause sharing.
2. **MGI items that limit a heavy user like GPAP:** release builds without `MGI_DB_CHECK`
   compiling calls out, ordering of transfers against queued tasks, the leaks, returning errors
   instead of aborting, compiling a kernel file once per process, and device queries (largest work
   group, memory) for the launch rules. When `freeObj` releases, it must also erase the handle
   from MGI's per-memory maps: OpenCL can hand the same handle out again, and `ProtectedMap::insert`
   asserts on a duplicate in debug builds.
3. **Upstreaming.** The four MGI changes into `feat/mgi`; the GPAP side into GPAP's `main`, then
   pin it here.
4. **Memory.** How much of the GPU mallob gives GPAP, and how GPAP should honour it (GPAP issue 72).
