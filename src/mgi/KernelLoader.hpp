#pragma once

#include <string>
#include <util/assert.hpp>
#include "MGIUtil.hpp"

namespace mgi {
    struct Kernel : public TypeHandle {};

    struct OCLDeferredAPI;

#ifdef MGI_API_OCL
    class KernelLoaderOCL {

    public:
        KernelLoaderOCL() = default;

        Kernel loadKernel(OCLDeferredAPI* api, const std::string& file);
    };

    using KernelLoader = KernelLoaderOCL;
#endif

}