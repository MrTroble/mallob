#pragma once

#include "KernelLoader.hpp"
#ifdef MGI_API_OCL
#include <CL/opencl.hpp>
#endif

namespace mgi {

enum class QueuePriority {
    EVERYTHING,
    LAST=EVERYTHING
};
constexpr auto QUEUE_AMOUNT = (size_t)QueuePriority::LAST;

struct OCLSetup {
    cl::Context context;
    cl::Platform platform;
    std::array<cl::CommandQueue, QUEUE_AMOUNT> queues;
};

#ifdef MGI_API_OCL
class OCLDeferredAPI {
    OCLSetup init;
    KernelLoaderOCL loader;

public:
    OCLDeferredAPI(const OCLSetup& init) : init(init) {}
};

inline OCLDeferredAPI initMGI() {
    OCLSetup setup;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform usedPlatform;
    for (auto platform : platforms)
    {
        const auto profile = platform.getInfo<CL_PLATFORM_PROFILE>();
        if(profile != "FULL_PROFILE") continue;
        usedPlatform = platform;
        break;
    }
    if (usedPlatform() == nullptr) {
        LOG(V0_CRIT, "Could not find full platform!\n");
        throw std::runtime_error("Could not find full platform!");
    }
#ifdef DEBUG
    const auto profile = usedPlatform.getInfo<CL_PLATFORM_PROFILE>();
    LOG(V5_DEBG, "Platform profile: %s\n", profile.c_str());
    const auto version = usedPlatform.getInfo<CL_PLATFORM_VERSION>();
    LOG(V5_DEBG, "Platform version: %s\n", version.c_str());
    const auto name = usedPlatform.getInfo<CL_PLATFORM_NAME>();
    LOG(V5_DEBG, "Platform: %s\n", name.c_str());
#endif

    return OCLDeferredAPI{setup};
}

using DeferredAPI = OCLDeferredAPI;
#endif

}
