#include "KernelLoader.hpp"
#include "DeferredAPI.hpp"

#ifdef MGI_API_OCL_HOST
#include <CL/opencl.hpp>
#endif

#ifdef MGI_API_VULKAN_HOST
#include <SPIRV-Reflect/spirv_reflect.h>
#endif

volatile int gdb_attached = 0;

namespace mgi
{

#ifdef MGI_API_OCL_HOST
    Kernel KernelLoader::loadKernel(OCLDeferredAPI *api, const std::string &file)
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
            for (auto &device : api->init.devicesUsed)
            {
                const auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
#ifdef DEBUG
                if (log.empty())
                {
                    LOG(V2_INFO, "Build successful for %s!\n", file.c_str());
                }
                else
                {
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

#ifdef MGI_API_VULKAN_HOST
    Kernel KernelLoader::loadKernel(VulkanDeferredAPI *api, const std::string &file)
    {
        // TODO Caching
        const auto source = mgi::wholeFile<std::vector<char>>(std::string(MALLOB_SUBPROC_DISPATCH_PATH "/") + file + "_v.spv");
        if (source.empty())
            return {};

        const auto deviceID = api->selectDevice();
        const auto device = api->setup.devices[deviceID];

        SpvReflectShaderModule moduleReflect;
        SpvReflectResult result = spvReflectCreateShaderModule(source.size(), source.data(), &moduleReflect);
        if(result != SPV_REFLECT_RESULT_SUCCESS) {
            LOG(V0_CRIT, "Could not create reflections of shader %s\n!", file.c_str());
            return {};
        }
        const auto sModule = device.createShaderModule(vk::ShaderModuleCreateInfo({}, source.size(), (const uint32_t *)source.data()));
        span entryPoints(moduleReflect.entry_points, moduleReflect.entry_point_count);
        std::unordered_map<std::string, VulkanPipelineInfo> pipeInfoLookup;
        for(const auto& ep : entryPoints) {
            LOG(V2_INFO, "Found entry point %s\n", ep.name);
            const auto& desc = *ep.descriptor_sets;
            assert(ep.descriptor_set_count == 1);
            std::vector<vk::DescriptorSetLayoutBinding> bindings;
            span descSpan(desc.bindings, desc.binding_count);
            VulkanPipelineInfo pipeInfos;
            for(const auto desc : descSpan) {
                vk::DescriptorSetLayoutBinding descBinding(desc->binding, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
                bindings.push_back(descBinding);
                pipeInfos.bindings.push_back(desc->binding);
            }

            std::vector<vk::PushConstantRange> range;
            span pushSpan(ep.used_push_constants, ep.used_push_constant_count);
            for(const auto pConst : pushSpan) {
                span searchSpan(moduleReflect.push_constant_blocks, moduleReflect.push_constant_block_count);
                auto iter = std::find_if(searchSpan.begin(), searchSpan.end(), [=](const auto& p) { return p.spirv_id == pConst; });
                if(iter == std::end(searchSpan)) {
                    LOG(V0_CRIT, "Could not find push const with SPIRV-ID %d\n", pConst);
                    continue;
                }
                range.push_back({vk::ShaderStageFlagBits::eCompute, iter->offset, iter->size});
            }
            vk::DescriptorSetLayoutCreateInfo descSetLayoutCreate({}, bindings);
            pipeInfos.setLayout = device.createDescriptorSetLayout(descSetLayoutCreate);
            vk::DescriptorPoolSize descPoolSize(vk::DescriptorType::eStorageBuffer, 1000u);
            pipeInfos.pool = device.createDescriptorPool({{}, 1000u, descPoolSize});
            vk::PipelineLayoutCreateInfo pipeLayoutCreate({}, pipeInfos.setLayout, range);
            pipeInfos.pipeLayout = device.createPipelineLayout(pipeLayoutCreate);

            vk::PipelineShaderStageCreateInfo shaderStageInfo({}, vk::ShaderStageFlagBits::eCompute, sModule, ep.name);
            vk::ComputePipelineCreateInfo computePipeCreate({}, shaderStageInfo, pipeInfos.pipeLayout);
            const auto result = device.createComputePipeline({}, computePipeCreate);
            if(result.result != vk::Result::eSuccess) {
                LOG(V0_CRIT, "Could not create pipe for shader entry point %s\n", ep.name);
            }
            pipeInfos.pipeline = result.value;
            pipeInfoLookup[ep.name] = pipeInfos;
        }
        spvReflectDestroyShaderModule(&moduleReflect);

        std::lock_guard lg(api->shaderModuleLock);
        api->shaderModules.push_back(sModule);
        const auto id = api->shaderModules.size() - 1;
        return {id};
    }
#endif

}