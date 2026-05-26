#include "MGIKernelDefs.hpp"
#include "MGIShared.hpp"

const m_uint MAX_SIZE_CACHED = 256;

void mgiReserviorAddSample(MGI_OUT MGIReservoir* reservior, MGIRng* rng, MGIResolveInfo* resolve, float weight) {
    float value = atomic_load(&reservior->weight);
    while(atomic_compare_exchange_strong(&reservior->weight, &value, value + weight) != value)
        value = atomic_load(&reservior->weight);
    // Can we get rid of those compare exchanges
    // Does this kill the probabilty? Look at the Markov Chain: Ergodisity even needed?
    if((weight / (value + weight)) > mgiRNGRndFloat(rng)) {
        reservior->resolve = *resolve;
    }
}

bool checkIsInverseIn(int clause, int* literalsBegin, int* literalsEnd) {
    for(int* iter = literalsBegin; iter != literalsEnd; iter++) // TODO Check unrolling
        if(*iter + clause == 0) return true; // SIMD?
    return false;
}

MGI_KERNEL void findResolvents(MGI_IN int* clauses, MGI_IN m_uint* clausesStarts, MGI_OUT MGIResolveInfo* toResolve)
{
    const m_uint x = MGI_GID_X;
    const m_uint position = clausesStarts[x];
    const m_uint sizeOfClause = clausesStarts[x + 1] - position;

    const m_uint y = ((MGI_GID_Y) + x + 1) % (MGI_GSIZE_X); // Only compares with a halfe turnaround 
                                                            // because otherwise we would double check
    const m_uint otherPosition = clausesStarts[y];
    const m_uint otherSize = clausesStarts[y + 1] - otherPosition;

    // TODO Cached version

}