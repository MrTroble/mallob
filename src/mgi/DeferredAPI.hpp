#pragma once

#include "DeferredAPICommon.hpp"

#ifdef MGI_API_OCL_HOST
#include "DeferredAPIOCL.hpp"

namespace mgi
{
    using DeferredAPI = mgi::OCLDeferredAPI;
}
#elif defined(MGI_API_VULKAN_HOST)
#include "DeferredAPIVulkan.hpp"

namespace mgi
{
    using DeferredAPI = mgi::VulkanDeferredAPI;
}
#endif

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
