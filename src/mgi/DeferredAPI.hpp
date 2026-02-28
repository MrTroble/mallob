#pragma once

#include "KernelLoader.hpp"
#ifdef MGI_API_OCL
#include <CL/opencl.hpp>
#endif

namespace mgi {

struct InitInfo {
    std::vector<float> queuePriorities{1, 1.0f};
};

struct OCLSetup {
    cl::Context context;
    cl::Platform platform;
    std::vector<cl::Device> devicesUsed;
    std::vector<std::vector<cl::CommandQueue>> queues;
};

struct KernelCache {

};

#ifdef MGI_API_OCL

class OCLDeferredAPI {
    OCLSetup init;
    KernelLoaderOCL loader;
    std::vector<cl::Program> programs;

    friend class KernelLoaderOCL;
public:
    OCLDeferredAPI(OCLSetup&& init) : init(std::move(init)) {}

    Kernel loadKernel(const std::string& file, const KernelCache& cache = {}) {
        // TODO Caching
        return loader.loadKernel(this, file);
    }
};

inline OCLDeferredAPI initMGI(const InitInfo& info = {}) {
    OCLSetup setup;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform usedPlatform;
    for (auto platform : platforms)
    {
        setup.devicesUsed.clear();
        const auto profile = platform.getInfo<CL_PLATFORM_PROFILE>();
        if(profile != "FULL_PROFILE") continue;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &setup.devicesUsed);
        if (setup.devicesUsed.empty()) {
            LOG(V4_VVER, "Platform has no GPU devices!\n");
            continue;
        }
        usedPlatform = platform;
        break;
    }
    if (usedPlatform() == nullptr) {
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
        auto& deviceQueues = setup.queues[deviceID++];
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
