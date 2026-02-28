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
    program.build("-cl-std=CLC++ -I ./ -D MGI_API_OCL=1");
    const auto id = api->programs.size();
    api->programs.push_back(std::move(program));
    return {id};
}
#endif

}