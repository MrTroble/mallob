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
    std::vector<cl::CommandQueue> queues;
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
    std::vector<cl::Device> devices;
    for (auto platform : platforms)
    {
        devices.clear();
        const auto profile = platform.getInfo<CL_PLATFORM_PROFILE>();
        if(profile != "FULL_PROFILE") continue;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
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
    const cl_platform_id platformID = usedPlatform();
    const cl_context_properties contextFlags[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformID, 0};
    setup.context = cl::Context(devices, contextFlags);
    for (auto priority : info.queuePriorities)
    {
        cl::CommandQueue queue(cl::QueueProperties::OutOfOrder);
    }
    return OCLDeferredAPI{setup};
}

using DeferredAPI = OCLDeferredAPI;
#endif

}
