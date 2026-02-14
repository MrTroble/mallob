#include "KernelLoader.hpp"

#ifdef MGI_API_OCL
#include <CL/opencl.hpp>
#endif

namespace mgi {

#ifdef MGI_API_OCL
Kernel KernelLoaderOCL::loadKernel(const std::string& file) {
    cl::Program program;
    return {};
}
#endif

}