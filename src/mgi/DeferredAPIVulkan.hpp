#pragma once

#include "DeferredAPICommon.hpp"

#include <vulkan/vulkan.hpp>

namespace mgi
{   
    struct VulkanSetup
    {
        vk::Instance instance;
        std::vector<vk::PhysicalDevice> physicalDevices;
        std::vector<uint32_t> queueFamilies;
        std::vector<vk::Device> devices;
        std::vector<vk::Queue> queues;
    }; 

    class VulkanDeferredAPI
    {

        VulkanSetup setup;
        std::vector<vk::ShaderModule> shaderModules;
        KernelLoader loader;

        friend class KernelLoader;

        vk::PhysicalDevice selectPhyDevice()
        {
            // TODO make this device selection MPI dependent
            return setup.physicalDevices[0];
        }

        vk::Device selectDevice()
        {
            // TODO make this device selection MPI dependent
            return setup.devices[0];
        }

        vk::Queue selectQueue()
        {
            // TODO make this device selection MPI dependent
            return setup.queues[0];
        }

    public:
        VulkanDeferredAPI(VulkanSetup&& setup) : setup(std::move(setup)) {}

        Kernel loadKernel(const std::string &file, const KernelCache &cache = {})
        {
            return loader.loadKernel(this, file);
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
        VulkanSetup setup;

        std::vector<const char*> extensions;
        std::vector<const char*> layers{ "VK_LAYER_KHRONOS_VALIDATION" };
        static vk::ApplicationInfo appInfo("Mallob", VK_MAKE_VERSION(1, 0, 0), "Mallob", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_3);
        vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, layers, extensions);
        setup.instance = vk::createInstance(instanceCreateInfo);

        std::vector<const char*> deviceExtensions;
        const auto physicalDevices = setup.instance.enumeratePhysicalDevices();

        vk::DeviceQueueCreateInfo queueCreateInfo({}, 0, info.queuePriorities);
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features> features;
        auto& vulkan11Features = features.get<vk::PhysicalDeviceVulkan11Features>();
        vulkan11Features.variablePointersStorageBuffer = true;
        vulkan11Features.variablePointers = true;
        vk::DeviceCreateInfo deviceCreateInfo({}, queueCreateInfo, {}, deviceExtensions);
        deviceCreateInfo.pNext = &features;

        for (const auto &phyDevice : physicalDevices)
        {
            const auto queueFamilies = phyDevice.getQueueFamilyProperties();
            const auto computeQueueFamily = std::find_if(queueFamilies.begin(), queueFamilies.end(), [](const vk::QueueFamilyProperties &qf) {
                return qf.queueFlags & vk::QueueFlagBits::eCompute;
            });
            if(computeQueueFamily == queueFamilies.end()) {
                LOG(V1_WARN, "No compute queue family found on device, skipping!\n");
                continue;
            }
            const auto deviceProperties = phyDevice.getProperties();
            if(deviceProperties.apiVersion < VK_API_VERSION_1_3) {
                LOG(V1_WARN, "Vulkan device %s does not support Vulkan 1.3, skipping!\n", deviceProperties.deviceName.data());
                continue;
            }
            vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features> featurePresent;
            phyDevice.getFeatures2(&featurePresent.get<vk::PhysicalDeviceFeatures2>());
            const auto& vulkan11Features = featurePresent.get<vk::PhysicalDeviceVulkan11Features>();
            if(!vulkan11Features.variablePointersStorageBuffer || !vulkan11Features.variablePointers) {
                LOG(V1_WARN, "Vulkan device %s does not support variable pointers, skipping!\n", deviceProperties.deviceName.data());
                continue;
            }
            LOG(V2_INFO, "Found Vulkan device: %s\n", deviceProperties.deviceName.data());

            const auto computeQueueFamilyIndex = std::distance(queueFamilies.begin(), computeQueueFamily);
            queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
            const auto device = phyDevice.createDevice(deviceCreateInfo);
            setup.devices.push_back(device);
            setup.queueFamilies.push_back(computeQueueFamilyIndex);
            setup.physicalDevices.push_back(phyDevice);
            for (size_t i = 0; i < setup.physicalDevices.size(); i++)
            {
                setup.queues.push_back(device.getQueue(computeQueueFamilyIndex, i));
            }
            vk::CommandPoolCreateInfo poolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, computeQueueFamilyIndex);
        }
        if(setup.devices.empty()) {
            LOG(V0_CRIT, "No suitable Vulkan devices found!\n");
            return {};
        }
        return setup;
    }
}
