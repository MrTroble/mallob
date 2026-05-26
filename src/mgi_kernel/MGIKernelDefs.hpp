#ifndef _MGI_KERNEL_DEFS
#define _MGI_KERNEL_DEFS

#ifdef MGI_API_OCL
#define MGI_KERNEL __kernel
#define MGI_IN __constant const
#define MGI_OUT __global
#define MGI_INOUT __global
#define MGI_GID get_global_id(0)
#define MGI_GID_X get_global_id(0)
#define MGI_GID_Y get_global_id(1)
#define MGI_GID_Z get_global_id(2)
#define MGI_GID_ND(x) get_global_id(x)

#define MGI_GSIZE get_global_size(0)
#define MGI_GSIZE_X get_global_size(0)
#define MGI_GSIZE_Y get_global_size(1)
#define MGI_GSIZE_Z get_global_size(2)
#define MGI_GSIZE_ND(x) get_global_size(x)

#else
#define MGI_KERNEL
#define MGI_IN
#define MGI_OUT
#define MGI_INOUT
#define MGI_GID 1
#define MGI_GID_X 1
#define MGI_GID_Y 1
#define MGI_GID_Z 1
#define MGI_GID_ND(x) 1

#define MGI_GSIZE 1
#define MGI_GSIZE_X 1
#define MGI_GSIZE_Y 1
#define MGI_GSIZE_Z 1
#define MGI_GSIZE_ND(x) 1

#endif

#endif //_MGI_KERNEL_DEFS
