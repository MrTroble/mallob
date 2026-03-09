#include "MGIKernelDefs.hpp"

#ifndef MGI_API_OCL
#error "OCL NOT DEFINED THIS SHOULD NOT BE THE CASE"
#endif

MGI_KERNEL void test(MGI_IN int* n)
{
    int m = n[0] * 2;
}