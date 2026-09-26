#pragma once

#include <string>
#include <util/assert.hpp>
#include "MGIUtil.hpp"

namespace mgi {
    struct Kernel : public TypeHandle {};
    MGI_DEFINE_TYPE_HASH(mgi::Kernel);

    struct VulkanDeferredAPI;
    struct OCLDeferredAPI;

    class KernelLoader {

    public:
        KernelLoader() = default;

        #ifdef MGI_API_OCL_HOST
        Kernel loadKernel(OCLDeferredAPI* api, const std::string& file);
        #elif defined(MGI_API_VULKAN_HOST)
        Kernel loadKernel(VulkanDeferredAPI* api, const std::string& file);
        #endif
    };

}