#include "KernelLoader.hpp"
#include "DeferredAPI.hpp"

#ifdef MGI_API_OCL
#include <CL/opencl.hpp>
#endif

namespace mgi
{

#ifdef MGI_API_OCL
    Kernel KernelLoaderOCL::loadKernel(OCLDeferredAPI *api, const std::string &file)
    {
        // TODO Auto package binaries
        const auto source = mgi::wholeFile<std::string>(file);
        if (source.empty())
            return {};
        cl::Program program(api->init.context, source);

        const auto kernelDefs = mgi::wholeFile<std::string>("mgi_kernel/MGIKernelDefs.hpp");
        cl::Program kernelDefsProgram(api->init.context, kernelDefs);

        std::vector<cl::Program> defaultIncludePrograms{kernelDefsProgram};
        std::vector<std::string> defaultIncludeNames{"MGIKernelDefs.hpp"};
        std::string compilerOptions = "-cl-std=CL2.0 -D MGI_API_OCL -I ./";

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