#pragma once

#include "DeferredAPI.hpp"
#include <mgi_kernel/MGIShared.hpp>

inline void printDebugOutput(mgi::DeferredAPI &_mgi_api, mgi::Kernel resolutionKernel, mgi::Memory mgiInfo)
{
    using namespace mgi;
    const auto readLock = _mgi_api.readMemory(mgiInfo, from(ReadInfo{sizeof(MGIInfo)}));
    MGIInfo *res = ((MGIInfo *)readLock.ptr[0]);
    if(res->__pDebugHelper.lastIndex != 0)
        LOG(V5_DEBG, "Shader: %s\n", res->__pDebugHelper.messageBuffer);
    TaskInfo taskInfo{{}, TaskType::Burst, {1, 1, 1}};
    taskInfo.kernel = resolutionKernel;
    taskInfo.function = "debugReset";
    taskInfo.descriptor.memory.push_back(mgiInfo);
    _mgi_api.queueWaitTasks(from(taskInfo));
}
