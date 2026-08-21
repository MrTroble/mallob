#include "KernelLoader.hpp"
#include "DeferredAPI.hpp"

#ifdef MGI_API_OCL_HOST
#include <CL/opencl.hpp>
#endif

namespace mgi
{

#ifdef MGI_API_OCL_HOST
    Kernel KernelLoaderOCL::loadKernel(OCLDeferredAPI *api, const std::string &file)
    {
        // TODO Auto package binaries
        const auto source = mgi::wholeFile<std::string>(std::string(MALLOB_SUBPROC_DISPATCH_PATH "/") + file);
        if (source.empty())
            return {};
        cl::Program program(api->init.context, source);

        // TODO Dynamic!!!
        const auto kernelDefs = mgi::wholeFile<std::string>(MALLOB_SUBPROC_DISPATCH_PATH "/mgi_kernel/MGIKernelDefs.hpp");
        cl::Program kernelDefsProgram(api->init.context, kernelDefs);

        const auto kernelShared = mgi::wholeFile<std::string>(MALLOB_SUBPROC_DISPATCH_PATH "/mgi_kernel/MGIShared.hpp");
        cl::Program kernelSharedProgram(api->init.context, kernelShared);

        std::vector<cl::Program> defaultIncludePrograms{kernelDefsProgram, kernelShared};
        std::vector<std::string> defaultIncludeNames{"MGIKernelDefs.hpp", "MGIShared.hpp"};
        std::string additionalOptions;
        #ifdef DEBUG
            // TODO CHECK available
            additionalOptions += "-cl-nv-verbose -cl-nv-opt-level=0";
        #endif
        std::string compilerOptions = additionalOptions + " -cl-std=CL2.0 -D MGI_API_OCL -I ./";

        try
        {
            program.compile(compilerOptions, defaultIncludePrograms, defaultIncludeNames);
            for (auto& device : api->init.devicesUsed)
            {
                const auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
#ifdef DEBUG
                if(log.empty()) {
                    LOG(V2_INFO, "Build successful for %s!\n", file.c_str());
                } else {
                    LOG(V2_INFO, "Build successful for %s with:\n", file.c_str());
                    LOG(V1_WARN, "%s\n", log.c_str());
                }
#endif
            }
            program = cl::linkProgram({program}, "");
        }
        catch (const cl::BuildError &error)
        {
            const auto log = error.getBuildLog();
            LOG(V0_CRIT, "Build failed with:\n");
            for (auto [device, line] : log)
            {
                LOG(V0_CRIT, "%s\n", line.c_str());
            }
            return {};
        }
        const auto id = api->programs.size();
        api->programs.push_back(std::move(program));
        return {id};
    }
#endif

}