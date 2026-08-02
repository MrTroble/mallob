#include "MGIKernelDefs.hpp"

#ifdef MGI_API_OCL
#elif defined(MGI_API_VULKAN)
#else
#error "NO VALID API DEFINED THIS SHOULD NOT BE THE CASE"
#endif

MGI_KERNEL(test) (MGI_GLOBAL int* n)
{
    int gid = MGI_GID;
    n[gid] *= 2;
}