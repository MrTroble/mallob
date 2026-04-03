#pragma once

#include <algorithm>

#include "KernelLoader.hpp"
#ifdef MGI_API_OCL
#include <CL/opencl.hpp>
#endif

namespace mgi
{

    struct InitInfo
    {
        std::vector<float> queuePriorities{1.0f};
    };

    struct OCLSetup
    {
        cl::Context context;
        cl::Platform platform;
        std::vector<cl::Device> devicesUsed;
        std::vector<std::vector<cl::CommandQueue>> queues;
    };

    struct KernelCache
    {
    };

    struct Memory : TypeHandle
    {
    };

    enum class MemoryType
    {
        Global,      // All read/write
        Uniform,     // Host read/write, Kernel read only
        DeviceLocal, // Kernel read/write only
        Constant     // Kernel read only
    };

    inline bool isWritable(MemoryType type) {
        switch (type)
        {
        case MemoryType::Uniform:
        case MemoryType::Global:
            return true;
        default:
            return false;
        }
    }

    struct AllocationInfo
    {
        Extension extensions;
        MemoryType type;
        size_t size;
        const void *initialMemory = nullptr;
        size_t initialSize = 0;
    };

    struct AllocationSlab
    {
        size_t size;
        MemoryType type;
    };

    struct AllocationRegions
    {
        size_t offset;
        size_t size;
        size_t index;
    };

    struct AllocationStrategy
    {

        virtual ~AllocationStrategy() {}

        virtual std::vector<AllocationSlab> slabs(span<const AllocationInfo> infos) const;

        virtual bool needsSubBuffers(span<const AllocationInfo> infos) const;

        virtual std::vector<AllocationRegions> regions(span<const AllocationInfo> infos) const;
    };

#ifdef MGI_API_OCL

    inline cl_mem_flags toOCLMemoryType(MemoryType type)
    {
        switch (type)
        {
        case MemoryType::Constant:
            return CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS;
        case MemoryType::DeviceLocal:
            return CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;
        case MemoryType::Uniform:
            return CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        case MemoryType::Global:
            return CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
        default:
            LOG(V0_CRIT, "The given type is not valid for buffer creation! Extension defined?\n");
            return {};
        }
    }

    class OCLDeferredAPI
    {
        OCLSetup init;
        KernelLoaderOCL loader;
        std::vector<cl::Program> programs;

        friend class KernelLoaderOCL;

    public:
        OCLDeferredAPI(OCLSetup &&init) : init(std::move(init)) {}

        Kernel loadKernel(const std::string &file, const KernelCache &cache = {})
        {
            // TODO Caching
            return loader.loadKernel(this, file);
        }

        std::vector<Memory> allocate(span<const AllocationInfo> infos, const AllocationStrategy &strategy = {})
        {               
            const auto slabsToAllocate = strategy.slabs(infos);
            std::vector<cl_mem> subBuffers;
            {
                std::vector<cl_mem> slabs;
                slabs.reserve(slabsToAllocate.size());
                for (auto &slab : slabsToAllocate)
                {
                    cl_mem_flags flags = toOCLMemoryType(slab.type);
                    cl_int error = 0;
                    cl_mem memory = clCreateBuffer(init.context.get(), flags, slab.size, nullptr, &error);
                    if (error != 0)
                    {
                        LOG(V0_CRIT, "Error: %u; Slab creation failed for buffer with type %u\n", error, (uint32_t)slab.type);
                        return {};
                    }
                    slabs.push_back(memory);
                }
                if (strategy.needsSubBuffers(infos))
                {
                    const auto regions = strategy.regions(infos);
                    subBuffers.resize(infos.size());
                    for (size_t i = 0; i < infos.size(); i++)
                    {
                        const auto &info = infos[i];
                        const auto &region = regions[i];
                        cl_mem_flags flags = toOCLMemoryType(info.type);
                        cl_buffer_region buff_region{region.offset, region.size};
                        cl_int error = 0;
                        cl_mem subbuffer = clCreateSubBuffer(slabs[region.index], flags, CL_BUFFER_CREATE_TYPE_REGION, &buff_region, &error);
                        if (error != 0)
                        {
                            LOG(V0_CRIT, "Error: %u; Subbuffer creation failed with type %u\n", error, (uint32_t)info.type);
                            return {};
                        }
                        subBuffers[i] = subbuffer;
                    }
                }
                else
                {
                    subBuffers = std::move(slabs);
                }
            }
            // TODO: Better queue and device selection
            cl_command_queue queue = init.queues.back().back().get();
            std::vector<cl_event> events;
            events.reserve(infos.size());
            for (size_t i = 0; i < infos.size(); i++)
            {
                const auto &info = infos[i];
                if(info.initialMemory == nullptr || info.initialSize == 0) continue;
                const auto buffer = subBuffers[i];
                if(isWritable(info.type)) { // TODO: This can be segregated earlier for more performance
                    cl_event event;
                    clEnqueueWriteBuffer(queue, buffer, false, 0, info.initialSize, info.initialMemory, 0, nullptr, &event);
                    events.push_back(event);
                } else {
                    // TODO? MAP?
                }             
            }
            clWaitForEvents(events.size(), events.data());
            std::vector<Memory> memories(subBuffers.size());
            std::transform(subBuffers.begin(), subBuffers.end(), memories.begin(), [](cl_mem mem) {return Memory{(size_t)mem};});
            return memories;
        }
    };

    inline OCLDeferredAPI initMGI(const InitInfo &info = {})
    {
        OCLSetup setup;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform usedPlatform;
        for (auto platform : platforms)
        {
            setup.devicesUsed.clear();
            const auto profile = platform.getInfo<CL_PLATFORM_PROFILE>();
            if (profile != "FULL_PROFILE")
                continue;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &setup.devicesUsed);
            if (setup.devicesUsed.empty())
            {
                LOG(V4_VVER, "Platform has no GPU devices!\n");
                continue;
            }
            usedPlatform = platform;
            break;
        }
        if (usedPlatform() == nullptr)
        {
            LOG(V0_CRIT, "Could not find full platform with devices!\n");
            throw std::runtime_error("Could not find full platform with devices!");
        }
#ifdef DEBUG
        const auto profile = usedPlatform.getInfo<CL_PLATFORM_PROFILE>();
        LOG(V5_DEBG, "Platform profile: %s\n", profile.c_str());
        const auto version = usedPlatform.getInfo<CL_PLATFORM_VERSION>();
        LOG(V5_DEBG, "Platform version: %s\n", version.c_str());
        const auto name = usedPlatform.getInfo<CL_PLATFORM_NAME>();
        LOG(V5_DEBG, "Platform: %s\n", name.c_str());
        const auto extNames = usedPlatform.getInfo<CL_PLATFORM_EXTENSIONS>();
        LOG(V5_DEBG, "Extensions: %s\n", extNames.c_str());
#endif

        setup.platform = usedPlatform;

        const cl_platform_id platformID = usedPlatform();
        const auto contextFlags = Build<cl_context_properties>().with(CL_CONTEXT_PLATFORM, platformID);
        setup.context = cl::Context(setup.devicesUsed, contextFlags.value);

        const cl_command_queue_properties queueFlags = CL_QUEUE_ON_DEVICE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        size_t deviceID = 0;
        setup.queues.resize(setup.devicesUsed.size());
        for (const auto device : setup.devicesUsed)
        {
            auto &deviceQueues = setup.queues[deviceID++];
#ifdef DEBUG
            const auto deviceName = device.getInfo<CL_DEVICE_NAME>();
            LOG(V5_DEBG, "Device %u: %s\n", deviceID, deviceName.c_str());
            cl_int cppVersion = 0;
#endif
            for (const auto priority : info.queuePriorities)
            {
                if (priority != 1.0f)
                    LOG(V1_WARN, "Currently priorities other then 1.0f are unsupported\n");
                deviceQueues.emplace_back(setup.context, device, queueFlags);
            }
        }
        return OCLDeferredAPI{std::move(setup)};
    }

    using DeferredAPI = OCLDeferredAPI;
#endif

}
