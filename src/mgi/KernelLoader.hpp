#pragma once

#include <string>

namespace mgi {
    struct Kernel {
        size_t internal;
    };

#ifdef MGI_API_OCL
    class KernelLoaderOCL {

    public:
        KernelLoaderOCL() = default;

        Kernel loadKernel(const std::string& file);
    };

    using KernelLoader = KernelLoaderOCL;
#endif

}