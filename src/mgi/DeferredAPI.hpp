#pragma once

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <numeric>
#include <vector>
#include "KernelLoader.hpp"
#ifdef MGI_API_OCL_HOST
#define CL_HPP_TARGET_OPENCL_VERSION 300
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

    template <typename BaseMap>
    struct ProtectedMap
    {
        BaseMap map;
        mutable std::shared_mutex mutex;

        void insert(typename BaseMap::value_type &&value)
        {
            std::lock_guard localGuard(mutex);
            const auto eval = map.insert(std::forward(value));
#ifdef DEBUG
            assert(eval.second);
#endif
        }

        template <typename InputIter>
        void insert(InputIter first, InputIter last)
        {
            std::lock_guard localGuard(mutex);
#ifdef DEBUG
            for (auto i = first; i != last; i++)
            {
                const auto eval = map.insert(*i);
                assert(eval.second);
            }
#else
            map.insert(first, last);
#endif
        }

        const typename BaseMap::mapped_type operator[](typename BaseMap::key_type &&key) const
        {
            std::shared_lock localGuard(mutex);
            const auto iter = map.find(key);
            assert(iter != std::end(map));
            return iter->second;
        }

        const typename BaseMap::mapped_type operator[](const typename BaseMap::key_type &key) const
        {
            std::shared_lock localGuard(mutex);
            const auto iter = map.find(key);
            assert(iter != std::end(map));
            return iter->second;
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

    struct MemoryCopyInfo {
        size_t size = 0;
        size_t srcOffset = 0;
        size_t destOffset = 0;
    };

    struct AllocationInfo
    {
        Extension extensions;
        MemoryType type;
        size_t size;
        const void *initialMemory = nullptr;
        size_t initialSize = 0;

        template <typename T>
        constexpr static AllocationInfo from(MemoryType type, span<const T> value)
        {
            return {{}, type, value.size_bytes(), value.data(), value.size_bytes()};
        }

        constexpr static AllocationInfo from(MemoryType type, size_t value)
        {
            return {{}, type, value};
        }
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

        virtual ~AllocationStrategy();

        virtual std::vector<AllocationSlab> slabs(span<const AllocationInfo> infos) const;

        virtual bool needsSubBuffers(span<const AllocationInfo> infos) const;

        virtual std::vector<AllocationRegions> regions(span<const AllocationInfo> infos) const;
    };

    struct AllocationCache
    {
        std::vector<std::pair<size_t, mgi::Memory>> memoryCache;
        size_t cacheLoadCount = 0;

        void retire(mgi::Memory memory, size_t size)
        {
            if (memoryCache.size() >= cacheLoadCount) [[likely]]
            {
                const auto iter = std::min_element(memoryCache.begin(), memoryCache.end(), [](auto c1, auto c2)
                                                   { return c1.first < c2.first; });
                if (iter == std::end(memoryCache))
                    return;
                std::swap(*iter, memoryCache.back());
                memoryCache.back() = std::make_pair(size, memory);
                return;
            }
            memoryCache.emplace_back(size, memory);
        }

        mgi::Memory tryGet(size_t size)
        {
            mgi::Memory memory;
            size_t lastSize = 0;
            for (auto &[currentSize, currentMemory] : memoryCache)
            {
                if (currentSize <= size && lastSize < currentSize)
                {
                    memory = currentMemory;
                    lastSize = currentSize;
                }
            }
            return memory;
        }
    };

    struct ReadInfo
    {
        size_t size = SIZE_MAX; // WholeSize
        size_t offset = 0;
    };

    typedef void (*ReadLockReleaseFunc)(std::vector<void *> &, void *);

    static void __noop_func(std::vector<void *> &, void *) {}

    struct ReadLock
    {
        ReadLockReleaseFunc releaseFunction = &__noop_func;
        void *customData = nullptr;
        std::shared_lock<std::shared_mutex> lock;
        std::vector<void *> ptr;

        ReadLock() = default;
        ReadLock(ReadLock &&) = default;
        ReadLock(std::shared_lock<std::shared_mutex> &&lock) : lock(std::move(lock)) {}

        ~ReadLock()
        {
            releaseFunction(ptr, customData);
        }
    };

    struct BufferUpdateInfo
    {
        void *data = nullptr;
        size_t size = SIZE_MAX;
        size_t destinationOffset = 0;

        template <typename T>
        constexpr static BufferUpdateInfo from(span<const T> value, size_t offset = 0)
        {
            return BufferUpdateInfo{(void *)value.data(), value.size_bytes(), offset};
        }

        template <typename T>
        constexpr static BufferUpdateInfo from(span<T> value, size_t offset = 0)
        {
            return BufferUpdateInfo{(void *)value.data(), value.size_bytes(), offset};
        }
    };

    struct MemoryDescriptor
    {
        std::vector<Memory> memory;
    };

    struct Task : public TypeHandle
    {
    };
    MGI_DEFINE_TYPE_HASH(mgi::Task);

    enum class TaskType : uint32_t
    {
        Burst,
        Long
    };

    enum class TaskStrategyType
    {
        OutOfOrder,
        Lockstep
    };

    struct TaskStrategy
    {
        TaskStrategyType type = TaskStrategyType::OutOfOrder;
    };

    struct TaskInfo
    {
        Extension extensions{};
        TaskType type = TaskType::Burst;
        size_t range[3];
        Kernel kernel;
        std::string function;
        MemoryDescriptor descriptor;
        size_t groupSizes[3] = {1, 1, 1};
        std::vector<Task> waitForTasks;
    };

    enum class TaskStatus
    {
        Queued,
        Submitted,
        Running,
        Complete,
        Error
    };

#ifdef MGI_API_OCL_HOST

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

    inline TaskStatus toTaskStatus(cl_int value)
    {
        switch (value)
        {
        case CL_QUEUED:
            return TaskStatus::Queued;
        case CL_SUBMITTED:
            return TaskStatus::Submitted;
        case CL_RUNNING:
            return TaskStatus::Running;
        case CL_COMPLETE:
            return TaskStatus::Complete;
        default:
            return TaskStatus::Error;
        }
    }

    struct ClearReadDataGlobal
    {
        cl_command_queue queue;
        cl_mem memory;
    };

    struct ClearReadDataCopy
    {
        cl_mem stagingBuffer;
    };

    class OCLDeferredAPI
    {
        OCLSetup init;
        KernelLoaderOCL loader;
        std::vector<cl::Program> programs;
        ProtectedMap<std::unordered_map<Memory, MemoryType>> typesCreated;
        ProtectedMap<std::unordered_map<Memory, std::shared_mutex *>> perMemoryMutex;
        ProtectedMap<std::unordered_map<std::string, std::vector<std::pair<Kernel, cl_kernel>>>> kernelCache;

        friend class KernelLoaderOCL;

        cl_command_queue selectQueue()
        {
            // TODO make this device selection MPI dependent
            return init.queues.back().back().get();
        }

        static void clearGlobalReadLock(std::vector<void *> &ptr, void *queuePtr)
        {
            if (queuePtr == nullptr)
                return;
            if (ptr.empty())
                return;
            const auto &data = *((ClearReadDataGlobal *)queuePtr);

            std::vector<cl_event> events(ptr.size());
            mgi::OnExit raiiEventsHandle([&]()
                                         { for(auto event : events) clRetainEvent(event); });
            size_t index = 0;
            for (const auto mapped : ptr)
            {
                MGI_DB_CHECK(clEnqueueUnmapMemObject(data.queue, data.memory, mapped, 0, nullptr, events.data() + index++),
                             "Could not unmap mem object!");
            }
            MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Event wait failed!");
            free(queuePtr);
        }

        static void clearCopyReadLock(std::vector<void *> &ptr, void *queuePtr)
        {
            if (queuePtr == nullptr || ptr.empty())
                return;
            const auto &data = *((ClearReadDataCopy *)queuePtr);
            clRetainMemObject(data.stagingBuffer);
            free(queuePtr);
        }

        cl_kernel selectKernel(const std::string &name, Kernel program)
        {
            {
                std::shared_lock sharedLock(kernelCache.mutex);
                const auto iter = kernelCache.map.find(name);
                if (iter != std::end(kernelCache.map))
                {
                    const auto &simpleMap = iter->second;
                    for (auto [p, kernel] : simpleMap)
                    {
                        if (p == program)
                            return kernel;
                    }
                }
            }
            std::lock_guard lock(kernelCache.mutex);
            auto &simpelMap = kernelCache.map[name];
            const auto compiled = programs[program.internal].get();
            cl_int error;
            const auto func = clCreateKernel(compiled, name.c_str(), &error);
            MGI_ERROR_CHECK(error, "Could not create kernel with name '%s' and id %u", return {}, name.c_str(), program.internal);
            simpelMap.emplace_back(program, func);
            return func;
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
            mgi::OnExit raiiEventsHandle([&]()
                                         { for(auto event : events) clRetainEvent(event); });
            events.reserve(infos.size());
            std::vector<cl_mem> buffersToRetain;
            buffersToRetain.reserve(infos.size());
            OnExit stagingBuffersRetain([&]()
                                        { for(const auto stager : buffersToRetain) clRetainMemObject(stager); });
            std::vector<std::pair<Memory, MemoryType>> typesToInsert;
            for (size_t i = 0; i < infos.size(); i++)
            {
                const auto &info = infos[i];
                const auto buffer = subBuffers[i];
                typesToInsert.emplace_back(Memory{(size_t)buffer}, info.type);
                if (info.initialMemory == nullptr || info.initialSize == 0)
                    continue;
                if (isHostWritable(info.type))
                { // TODO: This can be segregated earlier for more performance
                    cl_event event{};
                    MGI_DB_CHECK(clEnqueueWriteBuffer(queue, buffer, false, 0, info.initialSize, info.initialMemory, 0, nullptr, &event),
                                 "Could not enqueue write on allocate!");
                    events.push_back(event);
                }
                else
                {
                    // TODO Enable staging caches
                    cl_int error{};
                    const auto stagingBuffer = clCreateBuffer(init.context.get(), toOCLMemoryType(MemoryType::Global), info.initialSize, nullptr, &error);
                    MGI_DB_CHECK(error, "Failed to create staging buffer!");
                    buffersToRetain.push_back(stagingBuffer);

                    cl_event writeToStagingEvent{};
                    MGI_DB_CHECK(clEnqueueWriteBuffer(queue, stagingBuffer, false, 0, info.initialSize, info.initialMemory, 0, nullptr, &writeToStagingEvent),
                                 "Could not enqueue write on allocate!");

                    cl_event event{};
                    MGI_DB_CHECK(clEnqueueCopyBuffer(queue, stagingBuffer, buffer, 0, 0, info.size, 1, &writeToStagingEvent, &event),
                                 "Copy buffer in allocation failed!");
                    events.push_back(event);
                }
            }
            if(!events.empty()) {
                MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Write/Map Events failed!");
                for (auto event : events)
                    clRetainEvent(event);
                events.clear();
            }

            std::vector<std::pair<Memory, std::shared_mutex *>> mutexArray(subBuffers.size());
            size_t index = 0;
            for (auto &[mem, mutexPtr] : mutexArray)
            {
                mem = Memory{(size_t)subBuffers[index]};
                mutexPtr = new std::shared_mutex;
                index++;
            }
            this->perMemoryMutex.insert(mutexArray.begin(), mutexArray.end());
            this->typesCreated.insert(typesToInsert.begin(), typesToInsert.end());

            std::vector<Memory> memories(subBuffers.size());
            std::transform(subBuffers.begin(), subBuffers.end(), memories.begin(), [](cl_mem mem)
                           { return Memory{(size_t)mem}; });
            return memories;
        }

        ReadLock readMemory(Memory memory, span<const ReadInfo> reads)
        {
            assert(!reads.empty());
            ReadLock readLock(std::shared_lock(*this->perMemoryMutex[memory]));
            readLock.ptr.reserve(reads.size());
            cl_int error{};
            const auto queue = selectQueue();
            std::vector<cl_event> events(reads.size());
            mgi::OnExit raiiEventsHandle([&]()
                                         { for(auto event : events) clRetainEvent(event); });
            if (isHostWritable(typesCreated[memory]))
            {
                size_t index = 0;
                // This is shit, have locally cached versions
                auto clearData = (ClearReadDataGlobal *)malloc(sizeof(ClearReadDataGlobal));
                clearData->queue = queue;
                clearData->memory = (cl_mem)memory.internal;
                readLock.customData = clearData;
                readLock.releaseFunction = &clearGlobalReadLock;
                for (const auto &info : reads)
                {
                    cl_int error{};
                    const auto ptr = clEnqueueMapBuffer(queue, (cl_mem)memory.internal, false, CL_MAP_READ, info.offset, info.size,
                                                        0, nullptr, events.data() + index++, &error);
                    MGI_ERROR_CHECK(error, "Could not map global at offset %u with size %u",
                                    return {},
                                    info.offset, info.size);
                    readLock.ptr.push_back(ptr);
                }
            }
            else
            {
                readLock.releaseFunction = &clearCopyReadLock;
                std::vector<size_t> beginOffset(reads.size());
                size_t slabAllocated = 0;
                static constexpr size_t ALIGNMENT = 64; // TODO Revisit
                for (size_t i = 0; i < reads.size(); i++)
                {
                    size_t oldSize = reads[i].size;
                    // Alignment
                    oldSize += ALIGNMENT - (oldSize % ALIGNMENT);
                    beginOffset[i] = slabAllocated;
                    slabAllocated += oldSize;
                }

                cl_int error{}; // TODO Use cached staging if possible or requested
                const auto stagingBuffer = clCreateBuffer(init.context.get(), toOCLMemoryType(MemoryType::Global), slabAllocated, nullptr, &error);
                MGI_ERROR_CHECK(error, "Could not create staging buffer!", return {});

                auto clearData = (ClearReadDataCopy *)malloc(sizeof(ClearReadDataCopy));
                clearData->stagingBuffer = stagingBuffer;
                readLock.customData = clearData;

                for (size_t i = 0; i < reads.size(); i++)
                {
                    const auto &info = reads[i];
                    const auto offset = beginOffset[i];
                    MGI_ERROR_CHECK(clEnqueueCopyBuffer(queue, (cl_mem)memory.internal, stagingBuffer, info.offset, offset, info.size, 0, nullptr, events.data() + i),
                                    "Copy to staging for read failed!", {free(clearData); return {}; });
                }
                cl_event mapEvent{};
                const auto ptr = clEnqueueMapBuffer(queue, stagingBuffer, true, CL_MAP_READ, 0, slabAllocated, events.size(), events.data(),
                                                    &mapEvent, &error);
                MGI_DB_CHECK(error, "Could not map staging buffer for read!");
                for (size_t i = 0; i < reads.size(); i++)
                {
                    readLock.ptr.push_back((uint8_t *)ptr + beginOffset[i]);
                }
                MGI_DB_CHECK(clWaitForEvents(1, &mapEvent), "Wait event failed!");
            }
            MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Wait event failed!");
            return readLock;
        }

        inline void writeMemory(Memory memory, span<const BufferUpdateInfo> updates)
        {
            if (updates.empty())
                return;
            const auto queue = selectQueue();
            const auto buffer = (cl_mem)memory.internal;
            const auto type = typesCreated[memory];
            if (isHostWritable(type))
            {
                std::vector<cl_event> events(updates.size());
                mgi::OnExit raiiEventsHandle([&]()
                                             { for(auto event : events) clRetainEvent(event); });
                size_t idx = 0;
                std::lock_guard lockWrite(*perMemoryMutex[memory]);
                for (const auto &info : updates)
                {
                    MGI_DB_CHECK(clEnqueueWriteBuffer(queue, buffer, false, info.destinationOffset, info.size, info.data, 0, nullptr, &events[idx++]),
                                 "Could not enqueue write on allocate!");
                }
                MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Wait for events failed!");
            }
            else
            {
                // TODO Enable staging caches
                // TODO Yeah alignment please!
                const auto wholeSize = std::accumulate(updates.begin(), updates.end(), size_t{0}, [](auto first, auto &t)
                                                       { return t.size + first; });

                cl_int error{};
                const auto stagingBuffer = clCreateBuffer(init.context.get(), toOCLMemoryType(MemoryType::Global), wholeSize, nullptr, &error);
                OnExit stagingDestroy([&]()
                                      { clRetainMemObject(stagingBuffer); });
                MGI_DB_CHECK(error, "Failed to create staging buffer!");

                std::vector<cl_event> events(updates.size());
                mgi::OnExit raiiEventsHandle([&]()
                                             { for(auto event : events) clRetainEvent(event); });
                size_t offset = 0;
                size_t idx = 0;
                for (const auto &info : updates)
                {
                    MGI_DB_CHECK(clEnqueueWriteBuffer(queue, stagingBuffer, false, offset, info.size, info.data, 0, nullptr, &events[idx++]),
                                 "Could not enqueue write on allocate!");
                    offset += info.size;
                }
                MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Wait for events failed!");
                for (auto event : events)
                    clRetainEvent(event);
                // TODO Check if this locking is smart!
                std::lock_guard lockWrite(*perMemoryMutex[memory]);
                offset = 0;
                idx = 0;
                for (const auto &info : updates)
                {
                    MGI_DB_CHECK(clEnqueueCopyBuffer(queue, stagingBuffer, buffer, offset, info.destinationOffset, info.size, 0, nullptr, &events[idx++]),
                                 "Copy buffer in allocation failed!");
                    offset += info.size;
                }
                MGI_DB_CHECK(clWaitForEvents(events.size(), events.data()), "Wait for events failed!");
            }
        }

        std::vector<Task> queueTasks(span<const TaskInfo> tasks, const TaskStrategy &strategy = {})
        {
            const auto queue = selectQueue();
            cl_int error{};
            std::vector<Task> returnTasks(tasks.size());
            uint32_t index{};
            uint32_t dependencyAmount = (strategy.type == TaskStrategyType::OutOfOrder ? 0 : 1);
            std::vector<cl_event> dependencyNext;
            for (const auto &task : tasks)
            {
                // Cache is reused! To bad!
                const auto mainKernel = selectKernel(task.function, task.kernel);
                const auto kernel = clCloneKernel(mainKernel, &error);
                MGI_DB_CHECK(error, "Could not clone Kernel!"); // To bad clone needed ... parameter caching needed!
                uint32_t argID = 0;
                for (auto desc : task.descriptor.memory)
                {
                    MGI_DB_CHECK(!(bool)desc, "Descriptor memory is not valid! ID: %u", argID);
                    MGI_DB_CHECK(clSetKernelArg(kernel, argID++, sizeof(cl_mem), &desc),
                                 "Could not set kernel %s arguments for descriptors!", task.function.c_str());
                }
                dependencyNext.clear();
                if (strategy.type == TaskStrategyType::OutOfOrder && index != 0)
                {
                    dependencyNext.push_back(((cl_event *)returnTasks.data())[index - 1]);
                }
                if (!task.waitForTasks.empty())
                {
                    const auto last = dependencyNext.size();
                    dependencyNext.resize(last + task.waitForTasks.size());
                    std::copy((cl_event *)task.waitForTasks.data(),
                              (cl_event *)task.waitForTasks.data() + task.waitForTasks.size(), dependencyNext.begin() + last);
                }
                // Must be nullptr ... to bad that the api is shit!
                cl_event *dependencyChain = dependencyNext.empty() ? nullptr : dependencyNext.data();
                MGI_DB_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, task.range, task.groupSizes, dependencyNext.size(),
                                                    dependencyChain, (cl_event *)(returnTasks.data() + index)),
                             "Could not enqueue kernel %s", task.function.c_str());
                index++;
            }
            MGI_DB_CHECK(clFlush(queue), "Could not flush!");
            return returnTasks;
        }

        std::vector<TaskStatus> queueWaitTasks(span<const TaskInfo> taskInfos, const TaskStrategy &strategy = {})
        {
            const auto tasks = queueTasks(taskInfos, strategy);
            MGI_DB_CHECK(clWaitForEvents(tasks.size(), (cl_event *)tasks.data()), "Wait Tasks failed!");
            for(const auto t : tasks) freeObj(t);
            return std::vector(tasks.size(), TaskStatus::Complete);
        }

        void waitTasks(span<const Task> tasks)
        {
            MGI_DB_CHECK(clWaitForEvents(tasks.size(), (cl_event *)tasks.data()), "Wait Tasks failed!");
        }

        std::vector<TaskStatus> getStatus(span<const Task> tasks)
        {
            cl_int value = 0;
            std::vector<TaskStatus> status(tasks.size());
            cl_int index = 0;
            for (const auto t : tasks)
            {
                const auto error = clGetEventInfo((cl_event)t.internal, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(value), &value, nullptr);
                MGI_DB_CHECK(error, "Could not get event info!");
                status[index++] = toTaskStatus(value);
            }
            return status;
        }
    
        Task copyMemory(Memory source, Memory destination, span<const MemoryCopyInfo> copyInfos) {
            
            const auto queue = selectQueue();
            std::vector<cl_event> events(copyInfos.size());
            mgi::OnExit raiiEventsHandle([&]()
                                         { for(auto event : events) clRetainEvent(event); });
            size_t index = 0;
            for (const auto &info : copyInfos)
            {
                assert(source && destination && "Source or destination memory is not valid!");
                MGI_DB_CHECK(clEnqueueCopyBuffer(queue, (cl_mem)source.internal, (cl_mem)destination.internal, info.srcOffset, info.destOffset, info.size, 0, nullptr, &events[index++]),
                             "Could not enqueue copy buffer!");
            }
            cl_event userEvent;
            const auto error = clEnqueueMarkerWithWaitList(queue, events.size(), events.data(), &userEvent);
            MGI_DB_CHECK(error, "Could not create user event!");
            return Task{(size_t)userEvent};
        }

        void copyMemoryWait(Memory source, Memory destination, span<const MemoryCopyInfo> copyInfos) {
            const auto task = copyMemory(source, destination, copyInfos);
            MGI_DB_CHECK(clWaitForEvents(1, (cl_event *)&task.internal), "Wait for copy memory failed!");
            freeObj(task);
        }

        inline void freeObj(Memory memory) {
            MGI_DB_CHECK(clRetainMemObject((cl_mem)memory.internal), "Free failed!");
        }

        inline void freeObj(Task task) {
            MGI_DB_CHECK(clRetainEvent((cl_event)task.internal), "Free failed!");
        }
    };

    inline OCLSetup initMGI(const InitInfo &info = {})
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
        for (const auto &device : setup.devicesUsed)
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
        return setup;
    }

    using DeferredAPI = OCLDeferredAPI;
#endif
}

#ifndef MALLOB_USE_GPU // Dummy
#ifdef MGI_API_OCL_HOST
#error "OCL is active even tho GPU support is disabled!"
#endif
namespace mgi
{
    inline int initMGI(const InitInfo &info = {}) { return 0; }

    struct DeferredAPI
    {
        int t;
    };
}
#endif
