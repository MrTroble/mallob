#pragma once

#include "KernelLoader.hpp"
#ifdef MGI_API_OCL
#include <CL/opencl.hpp>
#endif

namespace mgi {

template<typename clType>
struct Build {
    clType value[33];
    uint32_t current = 0;
    
    template<typename ValueType>
    Build<clType>& with(int key, ValueType value) {
        value[current] = (clType)key;
        value[current+1] = (clType)value;
        current += 2;
        value[current] = 0;
    }
};

struct InitInfo {
    std::vector<float> queuePriorities{1, 1.0f};
};

struct OCLSetup {
    cl::Context context;
    cl::Platform platform;
    std::vector<cl::Device> devicesUsed;
    std::vector<std::vector<cl::CommandQueue>> queues;
};

#ifdef MGI_API_OCL
class OCLDeferredAPI {
    OCLSetup init;
    KernelLoaderOCL loader;

public:
    OCLDeferredAPI(const OCLSetup& init) : init(init) {}
};

inline OCLDeferredAPI initMGI(const InitInfo& info) {
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
#endif
    setup.platform = usedPlatform;
    const cl_platform_id platformID = usedPlatform();
    const auto contextFlags = Build<cl_context_properties>().with(CL_CONTEXT_PLATFORM, platformID);
    setup.context = cl::Context(setup.devicesUsed, contextFlags.value);

    const cl_command_queue_properties queueFlags = CL_QUEUE_ON_DEVICE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    size_t deviceID = 0;
    for (const auto device : setup.devicesUsed)
    {
        auto& deviceQueues = setup.queues[deviceID++];
        for (const auto priority : info.queuePriorities)
        {
            if (priority != 1.0f)
                LOG(V1_WARN, "Currently priorities other then 1.0f are unsupported\n");
            cl::CommandQueue queue(setup.context, device, queueFlags);
            deviceQueues.push_back(queue);
        }
    }
    return OCLDeferredAPI{setup};
}

using DeferredAPI = OCLDeferredAPI;
#endif

}
