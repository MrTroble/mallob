#pragma once

#include <algorithm>
#include <mutex>
#include <shared_mutex>

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

    struct Memory : public TypeHandle
    {
    };
    MGI_DEFINE_TYPE_HASH(mgi::Memory);

    template<typename BaseMap>
    struct ProtectedMap {
        BaseMap map;
        std::shared_mutex mutex;

        void insert(typename BaseMap::value_type&& value) {
            std::lock_guard localGuard(mutex);
            map.insert(std::forward(value));
        }

        template<typename InputIter>
        void insert(InputIter first, InputIter last) {
            std::lock_guard localGuard(mutex);
            map.insert(first, last);
        }

        typename BaseMap::mapped_type& operator[](typename BaseMap::key_type&& key) {
            std::shared_lock localGuard(mutex);
            return map[key];
        }

        typename BaseMap::mapped_type& operator[](const typename BaseMap::key_type& key) {
            std::shared_lock localGuard(mutex);
            return map[key];
        }
    };

    enum class MemoryType
    {
        Global,      // All read/write
        Uniform,     // Host read/write, Kernel read only
        DeviceLocal, // Kernel read/write only
        Constant     // Kernel read only
    };

    inline bool isHostWritable(MemoryType type)
    {
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

    struct ReadInfo {
        size_t size = SIZE_MAX; // WholeSize 
        size_t offset = 0;
    };

    typedef void(*ReadLockReleaseFunc)(std::vector<void*>&, void*);

    static void __noop_func(std::vector<void*>&, void*) {}

    struct ReadLock {
        ReadLockReleaseFunc releaseFunction = &__noop_func;
        void* customData = nullptr;
        std::unique_lock<std::shared_mutex> lock;
        std::vector<void*> ptr;

        ReadLock() = default;
        ReadLock(ReadLock&&) = default;
        ReadLock(std::unique_lock<std::shared_mutex>&& lock) : lock(std::move(lock)) {}

        ~ReadLock() {
            releaseFunction(ptr, customData);
        }
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

    struct ClearReadDataGlobal {
        cl_command_queue queue;
        cl_mem memory;
    };

    class OCLDeferredAPI
    {
        OCLSetup init;
        KernelLoaderOCL loader;
        std::vector<cl::Program> programs;
        ProtectedMap<std::unordered_map<Memory, MemoryType>> typesCreated;
        ProtectedMap<std::unordered_map<Memory, std::shared_mutex*>> perMemoryMutex;

        friend class KernelLoaderOCL;

        cl_command_queue selectQueue() {
            // TODO make this device selection MPI dependent
            return init.queues.back().back().get();
        }

        static void clearGlobalReadLock(std::vector<void*>&ptr, void* queuePtr) {
            if(ptr.empty()) return;
            const auto& data = *((ClearReadDataGlobal*) queuePtr);
            
            std::vector<cl_event> events(ptr.size());
            mgi::OnExit raiiEventsHandle([&](){ for(auto event : events) clRetainEvent(event); });
            size_t index = 0;
            for(const auto mapped : ptr) {
                MGI_DB_CHECK(clEnqueueUnmapMemObject(data.queue, data.memory, mapped, 0, nullptr, events.data() + index++),
                             "Could not unmap mem object!");
            }
            MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Event wait failed!");
            free(queuePtr);
        }

        static void clearCopyReadLock(std::vector<void*>&ptr, void* queuePtr) {
            for(const auto alloc : ptr) free(alloc);
        }


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
                    MGI_ERROR_CHECK(error, "Slab creation failed for buffer with type %u", return {}, (uint32_t)slab.type);
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
                        MGI_ERROR_CHECK(error, "Subbuffer creation failed with type %u", return {}, (uint32_t)info.type)
                    }
                }
                else
                {
                    subBuffers = std::move(slabs);
                }
            }

            cl_command_queue queue = selectQueue();
            std::vector<cl_event> events;
            mgi::OnExit raiiEventsHandle([&](){ for(auto event : events) clRetainEvent(event); });
            events.reserve(infos.size());
            std::vector<std::tuple<cl_mem, uint8_t*, uint8_t*, size_t>> buffersToUnmap;
            buffersToUnmap.reserve(infos.size());
            std::vector<std::pair<Memory, MemoryType>> typesToInsert;
            for (size_t i = 0; i < infos.size(); i++)
            {
                const auto &info = infos[i];
                if (info.initialMemory == nullptr || info.initialSize == 0)
                    continue;
                const auto buffer = subBuffers[i];
                typesToInsert.emplace_back(Memory{(size_t)buffer}, info.type);
                if (isHostWritable(info.type))
                { // TODO: This can be segregated earlier for more performance
                    cl_event event{};
                    MGI_DB_CHECK(clEnqueueWriteBuffer(queue, buffer, false, 0, info.initialSize, info.initialMemory, 0, nullptr, &event), 
                                 "Could not enqueue write on allocate!");
                    events.push_back(event);
                }
                else
                {
                    cl_event event{};
                    cl_int error{};
                    auto hostBuffer = clEnqueueMapBuffer(queue, buffer, false, CL_MAP_WRITE, 0, info.initialSize, 0, nullptr, &event, &error);
                    MGI_DB_CHECK(error, "Map enqueue failed on allocate!");
                    buffersToUnmap.emplace_back(buffer, (uint8_t*)hostBuffer, (uint8_t*)info.initialMemory, info.initialSize);
                    events.push_back(event);
                }
            }
            MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Write/Map Events failed!");
            for(auto event : events) clRetainEvent(event); 
            events.clear();

            for (auto [buffer, ptr, from, amount] : buffersToUnmap)
            {
                std::copy(from, from + amount, ptr);
                cl_event event{};
                MGI_DB_CHECK(clEnqueueUnmapMemObject(queue, buffer, ptr, 0, nullptr, &event), "Could not unmap buffer!");
                events.push_back(event);
            }
            if(!events.empty())
                MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Unmap events failed!");

            this->typesCreated.insert(typesToInsert.begin(), typesToInsert.end());
            std::vector<std::pair<Memory, std::shared_mutex*>> mutexArray(subBuffers.size());
            size_t index = 0;
            for(auto &[mem, mutexPtr] : mutexArray) {
                mem = Memory{(size_t)subBuffers[index]};
                mutexPtr = new std::shared_mutex;
                index++;
            }
            this->perMemoryMutex.insert(mutexArray.begin(), mutexArray.end());

            std::vector<Memory> memories(subBuffers.size());
            std::transform(subBuffers.begin(), subBuffers.end(), memories.begin(), [](cl_mem mem)
                           { return Memory{(size_t)mem}; });
            return memories;
        }

        ReadLock readMemory(Memory memory, span<const ReadInfo> reads) {
            assert(!reads.empty());
            ReadLock readLock(std::unique_lock(*this->perMemoryMutex[memory]));
            readLock.ptr.reserve(reads.size());
            cl_int error{};
            const auto queue = selectQueue();
            std::vector<cl_event> events(reads.size());
            size_t index = 0;
            if(isHostWritable(typesCreated[memory])) {

                // This is shit, have locally cached versions
                auto clearData = (ClearReadDataGlobal*)malloc(sizeof(ClearReadDataGlobal));
                clearData->queue = queue;
                clearData->memory = (cl_mem)memory.internal;
                readLock.customData = clearData;
                readLock.releaseFunction = &clearGlobalReadLock;
                for (const auto& info : reads)
                {
                    cl_int error{};
                    const auto ptr = clEnqueueMapBuffer(queue, (cl_mem)memory.internal, true, CL_MAP_READ, info.offset, info.size, 
                        0, nullptr, events.data() + index++, &error);
                    MGI_ERROR_CHECK(error, "Could not map global at offset %u with size %u", return {},
                                    info.offset, info.size);
                    readLock.ptr.push_back(ptr);
                }
            } else {
                readLock.releaseFunction = &clearCopyReadLock;
                for (const auto& info : reads)
                {
                    void* ptr = malloc(info.size); // Well to bad use malloc here!
                    MGI_ERROR_CHECK(clEnqueueReadBuffer(queue, (cl_mem)memory.internal, true, info.offset, info.size, ptr, 0, nullptr, events.data() + index++),
                            "Could not read (copy to host) from GPU", return {});
                    readLock.ptr.push_back(ptr);
                }
            }
            MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Wait event failed!");
            return readLock;
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

        const cl_command_queue_properties queueFlags = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
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
                cl::CommandQueue queue(setup.context, device, queueFlags);
                deviceQueues.emplace_back(std::move(queue));
            }
        }
        return OCLDeferredAPI{std::move(setup)};
    }

    using DeferredAPI = OCLDeferredAPI;
#endif

}
