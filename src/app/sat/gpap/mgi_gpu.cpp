#include "app/sat/gpap/mgi_gpu.hpp"

// MGI's GPU calls exist only where util/assert.hpp defines DEBUG (an assert build); elsewhere
// MGI_DB_CHECK compiles them out, and every transfer and launch here would silently do nothing.
#ifndef DEBUG
#error "MgiGpu needs MALLOB_ASSERT: MGI compiles its OpenCL calls out without it"
#endif

#include <algorithm>
#include <cstring>

#include <gpap/opencl_launch.hpp>

#include "util/logger.hpp"

namespace {

// GPAP's launch rules for bcp.cl (gpap/opencl_launch.hpp) take the device's largest work group.
// MGI cannot query it; every NVIDIA GPU allows 256, the largest group the rules ask for.
constexpr size_t kMaxGroup = 256;

// Parameter blocks kept per kernel: a sweep launches the same kernel with the same few blocks.
constexpr size_t kParamBlocksPerKernel = 8;

mgi::Memory asMemory(gpap::GpuMem m) {
    mgi::Memory memory;
    memory.internal = (size_t)m;
    return memory;
}

template <typename T>
mgi::span<const T> one(const T &value) {
    return mgi::span<const T>(&value, 1);
}

}  // namespace

gpap_status MgiGpu::open(mgi::DeferredAPI &api, size_t memBytes, std::unique_ptr<MgiGpu> *out,
                         const std::string &kernelFile) {
    out->reset();
    const mgi::Kernel kernel = api.loadKernel(kernelFile);
    if (!kernel) {
        LOG(V1_WARN, "[MgiGpu] cannot load the GPAP kernels from %s\n", kernelFile.c_str());
        gpap::set_error("MGI cannot load the GPAP kernels from %s", kernelFile.c_str());
        return GPAP_ERR_KERNEL;
    }
    const std::vector<mgi::AllocationInfo> infos = {
        mgi::AllocationInfo::from(mgi::MemoryType::Global, kBounceBytes),
        mgi::AllocationInfo::from(mgi::MemoryType::DeviceLocal, 16)};
    const auto memories = api.allocate(infos);
    if (memories.size() != 2 || !memories[0] || !memories[1]) {
        LOG(V1_WARN, "[MgiGpu] cannot allocate the bounce buffer\n");
        gpap::set_error("MGI cannot allocate the bounce buffer");
        return GPAP_ERR_MEMORY;
    }
    out->reset(new MgiGpu(api, memBytes, kernel, memories[0], memories[1]));
    return GPAP_OK;
}

MgiGpu::MgiGpu(mgi::DeferredAPI &api, size_t memBytes, mgi::Kernel kernel, mgi::Memory bounce,
               mgi::Memory nullBuffer)
    : api_(api), memBytes_(memBytes), kernel_(kernel), bounce_(bounce), null_(nullBuffer) {}

MgiGpu::~MgiGpu() {
    sync();
    for (auto &kv : params_)
        for (const ParamBlock &p : kv.second) free(p.mem);
    api_.freeObj(bounce_);
    api_.freeObj(null_);
}

gpap_status MgiGpu::mem_info(int64_t *freeBytes, int64_t *totalBytes) {
    *freeBytes = (int64_t)memBytes_;   // MGI cannot measure memory: the budget is all there is
    *totalBytes = (int64_t)memBytes_;
    return GPAP_OK;
}

int64_t MgiGpu::max_buffer_bytes() const {
    // MGI aborts on a failed allocation, so stay inside what an OpenCL GPU allows in one buffer:
    // OpenCL guarantees a quarter of the device's memory, so 1 GiB on any GPU with 4 GiB or more.
    return std::min((int64_t)memBytes_, kMaxBufferBytes);
}

gpap_status MgiGpu::alloc(size_t bytes, gpap::GpuMem *out) {
    // GPAP bounds its per-row buffers by max_buffer_bytes() but not the formula's; MGI aborts the
    // process on a failed allocation, so refuse here instead.
    if ((int64_t)bytes > max_buffer_bytes()) {
        gpap::set_error("MGI: a buffer of %zu bytes is past the MGI plug's %lld-byte limit", bytes,
                        (long long)max_buffer_bytes());
        return GPAP_ERR_MEMORY;
    }
    const size_t size = std::max<size_t>(bytes, 16);   // MGI asserts on a zero-size buffer
    const auto memories = api_.allocate(one(mgi::AllocationInfo::from(mgi::MemoryType::DeviceLocal, size)));
    if (memories.size() != 1 || !memories[0]) {
        LOG(V1_WARN, "[MgiGpu] allocating %zu bytes failed\n", bytes);
        gpap::set_error("MGI: allocating %zu bytes failed", bytes);
        return GPAP_ERR_MEMORY;
    }
    *out = (gpap::GpuMem)memories[0].internal;
    sizes_[*out] = size;
    return GPAP_OK;
}

void MgiGpu::free(gpap::GpuMem m) {
    if (!m) return;
    sizes_.erase(m);
    api_.freeObj(asMemory(m));   // a queued launch keeps its own reference (OpenCL)
}

void MgiGpu::sync() {
    if (last_) {
        api_.waitTasks(one(last_));
        api_.freeObj(last_);
        last_ = mgi::Task{};
    }
    for (gpap::GpuMem v : views_) free(v);
    views_.clear();
}

gpap_status MgiGpu::zero(gpap::GpuMem m, size_t bytes) {
    if (!m || bytes == 0) return GPAP_OK;
    return run("fill_i8", (int64_t)bytes, 1, {gpap::buf(m), gpap::i32(0), gpap::i64((int64_t)bytes)});
}

gpap_status MgiGpu::upload(gpap::GpuMem dst, const void *src, size_t bytes) {
    if (bytes == 0) return GPAP_OK;
    sync();
    const auto *bytesIn = static_cast<const uint8_t *>(src);
    for (size_t done = 0; done < bytes; done += kBounceBytes) {
        const size_t n = std::min(kBounceBytes, bytes - done);
        const mgi::BufferUpdateInfo update{(void *)(bytesIn + done), n, 0};
        api_.writeMemory(bounce_, one(update));
        api_.copyMemoryWait(bounce_, asMemory(dst), one(mgi::MemoryCopyInfo{n, 0, done}));
    }
    return GPAP_OK;
}

gpap_status MgiGpu::download(void *dst, gpap::GpuMem src, size_t offset, size_t bytes) {
    if (bytes == 0) return GPAP_OK;
    sync();
    auto *bytesOut = static_cast<uint8_t *>(dst);
    for (size_t done = 0; done < bytes; done += kBounceBytes) {
        const size_t n = std::min(kBounceBytes, bytes - done);
        api_.copyMemoryWait(asMemory(src), bounce_, one(mgi::MemoryCopyInfo{n, offset + done, 0}));
        // MGI locks each buffer while a ReadLock lives: it must be gone before the next write.
        const auto lock = api_.readMemory(bounce_, one(mgi::ReadInfo{n, 0}));
        std::memcpy(bytesOut + done, lock.ptr[0], n);
    }
    return GPAP_OK;
}

gpap_status MgiGpu::copy(gpap::GpuMem dst, gpap::GpuMem src, size_t bytes) {
    if (bytes == 0) return GPAP_OK;
    sync();
    api_.copyMemoryWait(asMemory(src), asMemory(dst), one(mgi::MemoryCopyInfo{bytes, 0, 0}));
    return GPAP_OK;
}

gpap_status MgiGpu::adopt(uintptr_t, gpap::GpuMem *) {
    gpap::set_error("the MGI plug cannot adopt a foreign device pointer (the `_dev` entry points)");
    return GPAP_ERR_ARG;
}

bool MgiGpu::has_kernel(const char *key) const {
    const gpap::opencl::Spec *spec = gpap::opencl::find_spec(key);
    return spec && gpap::opencl::fits(*spec, kMaxGroup);
}

gpap_status MgiGpu::paramBlock(const std::string &key, const std::vector<int64_t> &values,
                               gpap::GpuMem *out) {
    std::vector<ParamBlock> &blocks = params_[key];
    for (const ParamBlock &b : blocks)
        if (b.values == values) {
            *out = b.mem;
            return GPAP_OK;
        }
    gpap::GpuMem m = 0;
    gpap_status st = alloc(values.size() * sizeof(int64_t), &m);
    if (st == GPAP_OK) st = upload(m, values.data(), values.size() * sizeof(int64_t));
    if (st != GPAP_OK) {
        free(m);
        return st;
    }
    // The upload synced, so no queued launch still reads the oldest block.
    if (blocks.size() >= kParamBlocksPerKernel) {
        free(blocks.front().mem);
        blocks.erase(blocks.begin());
    }
    blocks.push_back({values, m});
    *out = m;
    return GPAP_OK;
}

gpap_status MgiGpu::view(gpap::GpuMem src, size_t offset, gpap::GpuMem *out) {
    const auto it = sizes_.find(src);
    if (it == sizes_.end() || offset >= it->second) {
        gpap::set_error("MGI plug: an argument offset of %zu is past its buffer", offset);
        return GPAP_ERR_ARG;
    }
    const size_t bytes = it->second - offset;
    gpap::GpuMem v = 0;
    const gpap_status st = alloc(bytes, &v);
    if (st != GPAP_OK) return st;
    sync();   // the copy must see every earlier launch
    api_.copyMemoryWait(asMemory(src), asMemory(v), one(mgi::MemoryCopyInfo{bytes, offset, 0}));
    *out = v;
    return GPAP_OK;
}

gpap_status MgiGpu::run(const char *key, int64_t n0, int64_t n1,
                        std::initializer_list<gpap::GpuArg> args) {
    if (n0 <= 0 || n1 <= 0) return GPAP_OK;   // an empty extent: nothing to do
    const gpap::opencl::Spec *spec = gpap::opencl::find_spec(key);
    if (!spec || !gpap::opencl::fits(*spec, kMaxGroup)) {   // MGI asserts on a missing function
        gpap::set_error("kernel '%s' is not part of the mgi backend (bcp.cl)", key);
        return GPAP_ERR_KERNEL;
    }

    // Argument 0, the parameter block, then the buffers in order (gpap/opencl_launch.hpp).
    gpap::GpuMem prm = 0;
    gpap_status st = paramBlock(key, gpap::opencl::param_block(*spec, n0, n1, args), &prm);
    if (st != GPAP_OK) return st;

    mgi::TaskInfo task;
    task.descriptor.memory.push_back(asMemory(prm));
    std::vector<gpap::GpuMem> launchViews;   // joins views_ only once the launch is queued
    for (const gpap::GpuArg &a : args) {
        if (a.kind != gpap::GpuArg::BUF) continue;
        gpap::GpuMem m = a.mem;
        if (!m) {
            task.descriptor.memory.push_back(null_);
            continue;
        }
        if (a.offset) {
            st = view(a.mem, a.offset, &m);
            if (st != GPAP_OK) {
                for (gpap::GpuMem v : launchViews) free(v);
                return st;
            }
            launchViews.push_back(m);
        }
        task.descriptor.memory.push_back(asMemory(m));
    }

    const gpap::opencl::Launch launch = gpap::opencl::launch(*spec, n0, n1, kMaxGroup);
    task.type = mgi::TaskType::Burst;
    task.range[0] = launch.global;
    task.range[1] = 1;
    task.range[2] = 1;
    task.groupSizes[0] = launch.local;
    task.groupSizes[1] = 1;
    task.groupSizes[2] = 1;
    task.kernel = kernel_;
    task.function = gpap::opencl::function_name(key);
    if (last_) task.waitForTasks.push_back(last_);   // MGI's queue may run out of order

    const auto tasks = api_.queueTasks(one(task));
    if (last_) api_.freeObj(last_);
    last_ = tasks.at(0);
    views_.insert(views_.end(), launchViews.begin(), launchViews.end());
    return GPAP_OK;
}
