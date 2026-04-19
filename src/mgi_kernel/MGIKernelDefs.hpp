#ifndef _MGI_KERNEL_DEFS
#define _MGI_KERNEL_DEFS

#ifdef MGI_API_OCL
#define MGI_KERNEL __kernel
// TODO REVISIT!!!
#define MGI_IN __constant
#define MGI_OUT __global
#define MGI_INOUT __global
#define MGI_GID get_global_id(0)
#else
#define MGI_KERNEL
#define MGI_IN
#define MGI_OUT
#define MGI_INOUT
#define MGI_GID 1
#endif

#endif //_MGI_KERNEL_DEFS
