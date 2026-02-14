#pragma once

#include <string>

struct Kernel {
    size_t internal;
};

class KernelLoaderOCL {

public:
    KernelLoaderOCL() = default;

    Kernel loadKernel(const std::string& file);
};