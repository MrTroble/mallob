#pragma once

#ifdef MGI_API_OCL
#include <CL/opencl.hpp>
#endif

namespace mgi {

#ifdef MGI_API_OCL
class OCLDeferredAPI {
    
};
using DeferredAPI = OCLDeferredAPI;
#endif

}
