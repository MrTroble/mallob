#include "KernelLoader.hpp"
#include "DeferredAPI.hpp"

#ifdef MGI_API_OCL
#include <CL/opencl.hpp>
#endif

namespace mgi {

#ifdef MGI_API_OCL
Kernel KernelLoaderOCL::loadKernel(OCLDeferredAPI* api, const std::string& file) {
    // TODO Auto package binaries
    const auto source = mgi::wholeFile(file);
    if(source.empty()) return {};
    cl::Program program(api->init.context, source);
    std::string compilerOptions = "-cl-std=CL2.0 -I ./ -D MGI_API_OCL=1";
    #ifdef DEBUG
        compilerOptions += " -cl-nv-verbose";
    #endif
    program.build(compilerOptions);
    const auto id = api->programs.size();
    api->programs.push_back(std::move(program));
    return {id};
}
#endif

}