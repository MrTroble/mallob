#include "MGIKernelDefs.hpp"
#include "MGIShared.hpp"

const m_uint MAX_SIZE_CACHED = 256;

MGI_KERNEL void findResolvents(MGI_IN int* clauses, MGI_IN m_uint* clausesStarts, MGI_OUT MGIResolveInfo* toResolve)
{
    const m_uint x = MGI_GID_X;
    const m_uint position = clausesStarts[x];
    const m_uint sizeOfClause = clausesStarts[x + 1] - position;

    const m_uint y = (MGI_GID_Y);
    const m_uint otherPosition = clausesStarts[y];
    const m_uint otherSize = clausesStarts[y + 1] - otherPosition;

    // TODO Cached version


}