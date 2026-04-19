#include "MGIKernelDefs.hpp"

#ifndef MGI_API_OCL
#error "OCL NOT DEFINED THIS SHOULD NOT BE THE CASE"
#endif

MGI_KERNEL void test(MGI_INOUT int* n)
{
    int gid = MGI_GID;
    n[gid] *= 2;
}