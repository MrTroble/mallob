#pragma once

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <numeric>
#include <chrono>
#include <thread>
#include <vector>
#include "comm/sysstate.hpp"
#include <signal.h>
#include "KernelLoader.hpp"

extern volatile int gdb_attached;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static void __implWaitForDebugger() {
    SysState_disableUnresponsiveNodeCrashing(); \
    LOG(V1_WARN, "%s, %d\n", SysState_isUnresponsiveNodeCrashingEnabled() ? "true":"false", MyMpi::rank(MPI_COMM_WORLD));\
    LOG(V1_WARN, "DEBUGGER MODE: Disabled unresponsivnes check\n"); \
    while(gdb_attached == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1)); 
    }
}
#pragma GCC diagnostic pop

#ifdef DEBUG
#define WAIT_FOR_DEBUGGER  __implWaitForDebugger();
#else
#define WAIT_FOR_DEBUGGER
#endif

namespace mgi
{
    struct InitInfo
    {
        std::vector<float> queuePriorities{1.0f};
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

    struct BufferRange {
        size_t offset = 0;
        size_t size = SIZE_MAX;
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

}