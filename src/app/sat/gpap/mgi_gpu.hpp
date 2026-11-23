#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gpap/gpu_api.hpp>

#include "mgi/DeferredAPI.hpp"

// GPAP's GPU socket (gpap::Gpu) over MGI, the Mallob GPU Interface: GPAP's engine runs its OpenCL
// kernels (GPAP's cpp/kernels/bcp.cl) through MGI's kernel loader and task queue, by GPAP's own
// launch rules (gpap/opencl_launch.hpp).
//
// MGI's defects are worked around here, not fixed:
//  * MGI's queue may run out of order, and its transfers do not wait for queued kernels. So every
//    launch waits for the previous one, and every transfer first waits for the last launch.
//  * MGI's reads and writes of device-only memory leak a staging buffer each. So host transfers
//    go through one persistent host-visible bounce buffer, copied to and from on the device.
//
// Not thread-safe, like the engine that owns it. `api` must outlive it.
class MgiGpu final : public gpap::Gpu {
public:
    static constexpr size_t kBounceBytes = size_t(8) << 20;
    static constexpr int64_t kMaxBufferBytes = int64_t(1) << 30;

    // Loads `kernelFile` (a path relative to MALLOB_SUBPROC_DISPATCH_PATH) and allocates the bounce
    // buffer and the null buffer. `memBytes` is the memory budget mem_info reports.
    // On failure *out stays empty: GPAP_ERR_KERNEL if the kernels do not load.
    static gpap_status open(mgi::DeferredAPI &api, size_t memBytes, std::unique_ptr<MgiGpu> *out,
                            const std::string &kernelFile = "gpap_kernel/bcp.cl");
    ~MgiGpu() override;

    const char *backend() const override { return "mgi"; }
    const char *device_name() const override { return "MGI"; }
    gpap_status make_current() override { return GPAP_OK; }
    gpap_status mem_info(int64_t *freeBytes, int64_t *totalBytes) override;
    bool free_is_measured() const override { return false; }
    int64_t max_buffer_bytes() const override;

    gpap_status alloc(size_t bytes, gpap::GpuMem *out) override;
    void free(gpap::GpuMem m) override;
    gpap_status zero(gpap::GpuMem m, size_t bytes) override;
    gpap_status upload(gpap::GpuMem dst, const void *src, size_t bytes) override;
    gpap_status download(void *dst, gpap::GpuMem src, size_t offset, size_t bytes) override;
    gpap_status copy(gpap::GpuMem dst, gpap::GpuMem src, size_t bytes) override;
    gpap_status adopt(uintptr_t foreign, gpap::GpuMem *out) override;

    bool has_kernel(const char *key) const override;
    gpap_status run(const char *key, int64_t n0, int64_t n1,
                    std::initializer_list<gpap::GpuArg> args) override;

private:
    struct ParamBlock {
        std::vector<int64_t> values;
        gpap::GpuMem mem;
    };

    MgiGpu(mgi::DeferredAPI &api, size_t memBytes, mgi::Kernel kernel, mgi::Memory bounce,
           mgi::Memory nullBuffer);

    // Waits for the last launch, then frees the offset views of finished launches.
    void sync();
    // The device copy of a parameter block, kept for reuse (at most 8 per kernel).
    gpap_status paramBlock(const std::string &key, const std::vector<int64_t> &values,
                           gpap::GpuMem *out);
    // A new buffer holding `src` from `offset` on, for a kernel's offset argument.
    gpap_status view(gpap::GpuMem src, size_t offset, gpap::GpuMem *out);

    mgi::DeferredAPI &api_;
    const size_t memBytes_;
    const mgi::Kernel kernel_;
    const mgi::Memory bounce_;   // Global (host-visible), kBounceBytes
    const mgi::Memory null_;     // 16 bytes: what a kernel gets for a "no buffer" argument
    mgi::Task last_;             // the last launch, if any
    std::vector<gpap::GpuMem> views_;                  // offset views, freed at the next sync()
    std::unordered_map<gpap::GpuMem, size_t> sizes_;   // bytes of every buffer alloc() made
    std::unordered_map<std::string, std::vector<ParamBlock>> params_;   // per kernel, oldest first
};
