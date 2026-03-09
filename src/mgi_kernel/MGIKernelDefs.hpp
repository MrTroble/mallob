#ifndef _MGI_KERNEL_DEFS
#define _MGI_KERNEL_DEFS

#ifdef MGI_API_OCL
#define MGI_KERNEL __kernel
#define MGI_IN __constant
#else
#define MGI_KERNEL
#define MGI_IN
#endif

#endif //_MGI_KERNEL_DEFS
