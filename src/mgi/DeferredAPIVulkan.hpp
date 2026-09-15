#pragma once

#include "DeferredAPICommon.hpp"

#include <vulkan/vulkan.hpp>

namespace mgi
{   
    struct VulkanSetup
    {
        vk::Instance instance;
        vk::PhysicalDevice physicalDevice;
        vk::Device device;
        std::vector<vk::Queue> queues;
    }; 

    class VulkanDeferredAPI
    {

        VulkanSetup setup;

        vk::Queue selectQueue()
        {
            // TODO make this device selection MPI dependent
            return {};
        }

    public:
        VulkanDeferredAPI(VulkanSetup&& setup) : setup(std::move(setup)) {}

        Kernel loadKernel(const std::string &file, const KernelCache &cache = {})
        {
            // TODO Caching
            return {};
        }

        std::vector<Memory> allocate(span<const AllocationInfo> infos, const AllocationStrategy &strategy = {})
        {
            return {};
        }

        ReadLock readMemory(Memory memory, span<const ReadInfo> reads)
        {
            assert(!reads.empty());
            return {};
        }

        inline void writeMemory(Memory memory, span<const BufferUpdateInfo> updates)
        {
            if (updates.empty())
                return;
            
        }

        std::vector<Task> queueTasks(span<const TaskInfo> tasks, const TaskStrategy &strategy = {})
        {
            return {};
        }
        
        void waitTasks(span<const Task> tasks)
        {
        }

        std::vector<TaskStatus> queueWaitTasks(span<const TaskInfo> taskInfos, const TaskStrategy &strategy = {})
        {
            return {};
        }

        std::vector<TaskStatus> getStatus(span<const Task> tasks)
        {
            return {};
        }
    
        Task copyMemory(Memory source, Memory destination, span<const MemoryCopyInfo> copyInfos) {            
            return {};
        }

        void copyMemoryWait(Memory source, Memory destination, span<const MemoryCopyInfo> copyInfos) {
        }

        inline void freeObj(Memory memory) {
        }

        inline void freeObj(Task task) {
        }
    };

    inline VulkanSetup initMGI(const InitInfo &info = {})
    {
        return {};
    }
}
