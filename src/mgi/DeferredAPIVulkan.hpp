#pragma once

#include "DeferredAPICommon.hpp"

#include <vulkan/vulkan.hpp>

#define CHECK_RESULT(result) if(result != vk::Result::eSuccess) {\
                LOG(V0_CRIT, "VkResult is %s\n", vk::to_string(result).c_str());\
                LOG(V0_CRIT, "In %s:%d\n", __FILE__, __LINE__);\
            }

namespace mgi
{
    using MemTypeTable = std::array<uint32_t, (uint32_t)MemoryType::Constant + 1>;

    struct VulkanSetup
    {
        vk::Instance instance;
        std::vector<vk::PhysicalDevice> physicalDevices;
        std::vector<uint32_t> queueFamilies;
        std::vector<vk::Device> devices;
        std::vector<vk::Queue> queues;
        std::vector<vk::CommandPool> commandPools;
        std::vector<MemTypeTable> memoryTypeIndices;
        std::vector<std::mutex *> queuesProtects;
    };

    struct PackedBuffer
    {
        vk::Buffer buffer;
        vk::DeviceMemory memory;

        inline void destroy(vk::Device device)
        {
            device.destroy(buffer);
            device.free(memory);
        }
    };

    struct VulkanMemoryRegion
    {
        vk::DeviceMemory memory;
        size_t offset;
    };

    class VulkanDeferredAPI
    {

        VulkanSetup setup;
        std::vector<vk::ShaderModule> shaderModules;
        std::shared_mutex shaderModuleLock;
        KernelLoader loader;
        ProtectedMap<std::unordered_map<size_t, VulkanMemoryRegion>> bufferToMemLookup;
        ProtectedMap<std::unordered_map<size_t, size_t>> memoryCounter;

        friend class KernelLoader;

        uint32_t selectDevice()
        {
            // TODO make this device selection MPI dependent
            return 0;
        }

        vk::Queue selectQueue()
        {
            // TODO make this device selection MPI dependent
            return setup.queues[0];
        }

        inline PackedBuffer allocatePackedBuffer(size_t size, vk::Device device, size_t memType, vk::BufferUsageFlags flags)
        {
            vk::BufferCreateInfo bufferInfo({}, size, flags);
            const auto buffer = device.createBuffer(bufferInfo);
            const auto memRequ = device.getBufferMemoryRequirements(buffer);

            vk::MemoryAllocateInfo allocInfo(memRequ.size, memType);
            const auto mem = device.allocateMemory(allocInfo);
            device.bindBufferMemory(buffer, mem, 0);
            return {buffer, mem};
        }

        template <class T>
        inline void upload(vk::Device device, vk::DeviceMemory memory, span<const T> values, size_t offset = 0)
        {
            T *ptr = (T *)device.mapMemory(memory, offset, values.size_bytes());
            std::copy(values.begin(), values.end(), ptr);
            device.unmapMemory(memory);
        }

    public:
        VulkanDeferredAPI(VulkanSetup &&setup) : setup(std::move(setup)) {}

        Kernel loadKernel(const std::string &file, const KernelCache &cache = {})
        {
            return loader.loadKernel(this, file);
        }

        std::vector<Memory> allocate(span<const AllocationInfo> infos, const AllocationStrategy &strategy = {})
        {
            std::vector<Memory> allocated;
            const auto deviceID = selectDevice();
            const auto &memTypeIndices = setup.memoryTypeIndices[deviceID];
            const auto &device = setup.devices[deviceID];

            std::vector<MemoryRequirements> requirements;
            for (const auto &region : infos)
            {
                auto bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
                vk::BufferCreateInfo bufferInfo({}, region.size, bufferUsage);
                const auto buffer = device.createBuffer(bufferInfo);
                const auto memRequ = device.getBufferMemoryRequirements(buffer);
                requirements.push_back({memRequ.size, memRequ.alignment});
                Memory m;
                m.internal = (size_t)(VkBuffer)buffer;
                allocated.push_back(m);
            }

            const auto slabsToAlloc = strategy.slabs(infos, requirements);
            std::vector<vk::DeviceMemory> allocatedMem;
            for (const auto &slab : slabsToAlloc)
            {
                const auto memTypeIndex = memTypeIndices[(uint32_t)slab.type];
                vk::MemoryAllocateInfo allocInfo(slab.size, memTypeIndex);
                allocatedMem.push_back(device.allocateMemory(allocInfo));
            }

            const auto cmdPool = setup.commandPools[deviceID];
            const auto cmd = device.allocateCommandBuffers({cmdPool, vk::CommandBufferLevel::ePrimary, 1}).back();
            const auto regionsToAlloc = strategy.regions(infos, requirements);
            std::vector<PackedBuffer> packedBuffer;
            vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            cmd.begin(beginInfo);
            for (size_t i = 0; i < regionsToAlloc.size(); i++)
            {
                const auto &region = regionsToAlloc[i];
                const vk::Buffer buffer = *((VkBuffer *)&allocated[i].internal);
                const auto mem = allocatedMem[region.index];
                device.bindBufferMemory(buffer, mem, region.offset);

                const auto &info = infos[i];
                if (info.initialSize != 0 && info.initialMemory)
                {
                    if (isHostWritable(info.type))
                    {
                        upload<uint8_t>(device, mem, span<uint8_t>{(uint8_t *)info.initialMemory, info.initialSize}, region.offset);
                    }
                    else
                    {
                        const auto packBuffer = allocatePackedBuffer(info.initialSize, device,
                                                                     memTypeIndices[(uint32_t)MemoryType::Global], vk::BufferUsageFlagBits::eTransferSrc);
                        upload<uint8_t>(device, packBuffer.memory, span<uint8_t>{(uint8_t *)info.initialMemory, info.initialSize});
                        packedBuffer.push_back(packBuffer);
                        vk::BufferCopy copy(0, 0, info.initialSize);
                        cmd.copyBuffer(packBuffer.buffer, buffer, copy);
                    }
                }
            }
            cmd.end();
            const auto queue = selectQueue();
            vk::SubmitInfo submitInfo = {{}, {}, cmd};
            const auto fence = device.createFence({});
            queue.submit(submitInfo, fence);
            const auto result = device.waitForFences(fence, true, UINT64_MAX);
            CHECK_RESULT(result);
            return allocated;
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

        Task copyMemory(Memory source, Memory destination, span<const MemoryCopyInfo> copyInfos)
        {
            return {};
        }

        void copyMemoryWait(Memory source, Memory destination, span<const MemoryCopyInfo> copyInfos)
        {
        }

        inline void freeObj(Memory memory)
        {
        }

        inline void freeObj(Task task)
        {
        }
    };

    inline VulkanSetup initMGI(const InitInfo &info = {})
    {
        VulkanSetup setup;

        std::vector<const char *> extensions;
        std::array<const char *, 1> layersRequested = {"VK_LAYER_KHRONOS_validation"};
        std::vector<const char *> layers;
        const auto layerProperties = vk::enumerateInstanceLayerProperties();
        for (const auto layer : layersRequested)
        {
            const auto found = std::find_if(layerProperties.begin(), layerProperties.end(), [=](const vk::LayerProperties &lp)
                                            {
                LOG(V3_VERB, "Found Vulkan layer: %s\n", lp.layerName.data());
                return strcmp(lp.layerName.data(), layer) == 0; });
            if (found == layerProperties.end())
            {
                LOG(V1_WARN, "Vulkan layer %s not found, skipping!\n", layer);
                continue;
            }
            layers.push_back(layer);
        }

        static vk::ApplicationInfo appInfo("Mallob", VK_MAKE_VERSION(1, 0, 0), "Mallob", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_3);
        vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, layers, extensions);
        setup.instance = vk::createInstance(instanceCreateInfo);

        std::vector<const char *> deviceExtensions;
        const auto physicalDevices = setup.instance.enumeratePhysicalDevices();

        vk::DeviceQueueCreateInfo queueCreateInfo({}, 0, info.queuePriorities);
        static vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features> features;
        auto &vulkan11FeaturesSet = features.get<vk::PhysicalDeviceVulkan11Features>();
        vulkan11FeaturesSet.variablePointersStorageBuffer = true;
        vulkan11FeaturesSet.variablePointers = true;
        auto &physicalDeviceFeaturesSet = features.get<vk::PhysicalDeviceFeatures2>();
        physicalDeviceFeaturesSet.features.shaderInt64 = true;
        vk::DeviceCreateInfo deviceCreateInfo({}, queueCreateInfo, {}, deviceExtensions);
        deviceCreateInfo.pNext = &physicalDeviceFeaturesSet;

        for (const auto &phyDevice : physicalDevices)
        {
            const auto queueFamilies = phyDevice.getQueueFamilyProperties();
            const auto computeQueueFamily = std::find_if(queueFamilies.begin(), queueFamilies.end(), [](const vk::QueueFamilyProperties &qf)
                                                         { return qf.queueFlags & vk::QueueFlagBits::eCompute; });
            if (computeQueueFamily == queueFamilies.end())
            {
                LOG(V1_WARN, "No compute queue family found on device, skipping!\n");
                continue;
            }
            const auto deviceProperties = phyDevice.getProperties();
            if (deviceProperties.apiVersion < VK_API_VERSION_1_3)
            {
                LOG(V1_WARN, "Vulkan device %s does not support Vulkan 1.3, skipping!\n", deviceProperties.deviceName.data());
                continue;
            }
            vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features> featurePresent;
            phyDevice.getFeatures2(&featurePresent.get<vk::PhysicalDeviceFeatures2>());
            const auto &vulkan11Features = featurePresent.get<vk::PhysicalDeviceVulkan11Features>();
            if (!vulkan11Features.variablePointersStorageBuffer || !vulkan11Features.variablePointers)
            {
                LOG(V1_WARN, "Vulkan device %s does not support variable pointers, skipping!\n", deviceProperties.deviceName.data());
                continue;
            }
            const auto &physicalDeviceFeatures = featurePresent.get<vk::PhysicalDeviceFeatures2>();
            if (!physicalDeviceFeatures.features.shaderInt64)
            {
                LOG(V1_WARN, "Vulkan device %s does not support shaderInt64, skipping!\n", deviceProperties.deviceName.data());
                continue;
            }
            LOG(V2_INFO, "Found Vulkan device: %s\n", deviceProperties.deviceName.data());

            const auto computeQueueFamilyIndex = std::distance(queueFamilies.begin(), computeQueueFamily);
            queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
            const auto device = phyDevice.createDevice(deviceCreateInfo);
            setup.devices.push_back(device);
            setup.queueFamilies.push_back(computeQueueFamilyIndex);
            setup.physicalDevices.push_back(phyDevice);
            for (size_t i = 0; i < queueCreateInfo.queueCount; i++)
            {
                setup.queues.push_back(device.getQueue(computeQueueFamilyIndex, i));
            }
            vk::CommandPoolCreateInfo poolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, computeQueueFamilyIndex);
            setup.commandPools.push_back(device.createCommandPool(poolCreateInfo));
            MemTypeTable memTypeTable;
            const auto memProperties = phyDevice.getMemoryProperties();
            bool foundDeviceLocal = false, foundHostVisible = false;
            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
            {
                const auto memType = memProperties.memoryTypes[i];
                const auto flags = memType.propertyFlags;
                if (!foundDeviceLocal && flags & vk::MemoryPropertyFlagBits::eDeviceLocal)
                {
                    memTypeTable[(uint32_t)MemoryType::DeviceLocal] = i;
                    memTypeTable[(uint32_t)MemoryType::Constant] = i;
                    foundDeviceLocal = true;
                }
                if (!foundHostVisible && flags & vk::MemoryPropertyFlagBits::eHostVisible)
                {
                    memTypeTable[(uint32_t)MemoryType::Uniform] = i;
                    memTypeTable[(uint32_t)MemoryType::Global] = i;
                    foundHostVisible = true;
                }
            }
            setup.memoryTypeIndices.push_back(memTypeTable);
        }
        setup.queuesProtects.resize(setup.queues.size());
        for (auto &m : setup.queuesProtects)
            m = new std::mutex;
        if (setup.devices.empty())
        {
            LOG(V0_CRIT, "No suitable Vulkan devices found!\n");
            return {};
        }
        return setup;
    }
}
