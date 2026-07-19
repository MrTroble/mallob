#include "MGIKernelDefs.hpp"
#include "MGIShared.hpp"

const m_uint MAX_SIZE_CACHED = 256;

// TODO Redo with sorting
bool checkIsInverseIn(int clause, MGI_CONST int *literalsBegin, MGI_CONST int *literalsEnd)
{
    for (MGI_CONST int *iter = literalsBegin; iter != literalsEnd; iter++) // TODO Check unrolling
        if ((*iter + clause) == 0)
            return true; // SIMD?
    return false;
}

// TODO Reservoir based picking that only use different
// use MGI_LOCAL MGIReservoir in order to only have atomics from each work group

MGI_KERNEL void findResolvents(MGI_CONST int *clauses, MGI_CONST m_uint *clausesStarts, MGI_GLOBAL MGIResolveInfo *toResolve)
{
    const m_uint x = MGI_GID_X;
    const m_uint position = clausesStarts[x];
    const m_uint sizeOfClause = clausesStarts[x + 1] - position;

    const m_uint sizeX = MGI_GSIZE_X;
    const m_uint y = ((MGI_GID_Y) + x + 1) % (sizeX + 1); // Only compares with a halfe turnaround (n - 1)n/2
                                                          // because otherwise we would double check
    MGI_CONST int *otherBegin = clauses + clausesStarts[y];
    MGI_CONST int *otherEnd = clauses + clausesStarts[y + 1];

    MGI_GLOBAL MGIResolveInfo *localResolve = toResolve + x + (y - x - 1) * sizeX;
    localResolve->literal = 0;

    // TODO Cached version
    MGI_CONST int *endIter = clauses + clausesStarts[x + 1];
    for (MGI_CONST int *iter = clauses + position; iter != endIter; iter++)
    {
        if (checkIsInverseIn(*iter, otherBegin, otherEnd))
        {
            localResolve->literal = *iter;
            localResolve->clauseOne = x;
            localResolve->clauseTwo = y;
            localResolve->resolvedSize = 1; // TODO
            break;
        }
    }
}

// TODO Prefetching
MGI_KERNEL void findResolventsReservoir(MGI_GLOBAL MGIInfo *info, MGI_CONST int *clauses, MGI_CONST m_uint *clausesStarts, MGI_GLOBAL MGIReservoir *toResolve)
{
    const m_uint position = clausesStarts[MGI_GID_X];
    const m_uint sizeOfClause = clausesStarts[MGI_GID_X + 1] - position - 1;

    // TODO Use LOCAL reservoir and ONLY MERGE AT THE END! Shuffle reservoirs
    // TODO Check Register pressure
    // TODO Use shared cache for clause lookups

    MGI_LOCAL MGIReservoir currentReservoir = toResolve[MGI_GID_X];
    currentReservoir.weight = 0;
    currentReservoir.resolve.literal = 0;
    const m_uint currentBegin = position;
    const m_uint currentEnd = currentBegin + sizeOfClause;

    {
        MGI_LOCAL MGIResolveInfo resolve;
        resolve.clauseOne = MGI_GID_X;

        MGI_LOCAL MGIRng rng;
        mgiRNGInit(&rng, MGI_GID_X, 0, 1, 117007);
        const float maxValue = (float)info->maxClauseSize;

        __attribute__((opencl_unroll_hint(1)))
        for (m_uint i = 0; i < MGI_GSIZE_Y; i++)
        {
            const m_uint index = (MGI_GID_X + i + 1) % MGI_GSIZE_X;
            m_uint otherBegin = clausesStarts[index];
            const m_uint otherEnd = clausesStarts[index + 1] - 1;
            m_uint iter = currentBegin;
            resolve.literal = 0;
            resolve.clauseTwo = index;
            resolve.resolvedSize = 0;
            
            __attribute__((opencl_unroll_hint(1)))
            while (otherBegin < otherEnd && iter < currentEnd)
            {
                const int l1 = clauses[iter];
                const int l2 = clauses[otherBegin];
                
                const int l1A = abs(l1);
                const int l2A = abs(l2);

                resolve.literal += ((l1 == -l2 && resolve.literal == 0) ? l1A : 0);
                resolve.resolvedSize += (l1 != -l2 ? 1:0);
                const uint equal = l1A == l2A ? 1:0;
                const uint oneSmalerTwo = l1A < l2A ? 1:0;
                iter += oneSmalerTwo + equal;
                otherBegin += (1 - oneSmalerTwo);
            }
            if (otherBegin != otherEnd)
            {
                resolve.resolvedSize += otherEnd - otherBegin;
            }
            if (iter != currentEnd)
            {
                resolve.resolvedSize += currentEnd - iter;
            }
            float weight = (resolve.literal == 0 ? 0 : 1) * (1.0f / ((float)MGI_GSIZE_Y)) * (1 - (resolve.resolvedSize / maxValue));
            mgiReserviorAddSample(&currentReservoir, &rng, &resolve, weight);
        }
    }
    toResolve[MGI_GID_X] = currentReservoir;
}

// https://dl.acm.org/doi/10.1145/7902.7903
MGI_KERNEL void clauseOuts(MGI_GLOBAL MGIInfo *info, MGI_CONST MGIReservoir *resolve, MGI_GLOBAL m_uint *clauseOuts)
{
    const m_uint current = MGI_GID_X + 1;
    clauseOuts[0] = 0; // TODO Recheck
    {
        clauseOuts[current] = resolve[MGI_GID_X].resolve.resolvedSize;
    }

    MGI_BARRIER(MGI_MEM_GLOBAL);

    m_uint maxSteps = ceil(log2((float)MGI_GSIZE_X));
    // TODO Shared caches!!!
    for (size_t i = 0; i < maxSteps; i++)
    {
        const m_uint otherIndex = current + (1 << i);
        if (otherIndex < MGI_GSIZE_X)
        {
            clauseOuts[otherIndex] += clauseOuts[current];
        }
        MGI_BARRIER(MGI_MEM_GLOBAL);
    }
}

MGI_KERNEL void debugReset(MGI_GLOBAL MGIInfo *info) { info->__pDebugHelper.lastIndex = 0; }

// Merge for resolve
MGI_KERNEL void resolve(MGI_GLOBAL MGIInfo *info, MGI_CONST MGIReservoir *resolve, MGI_CONST int *clauses,
                        MGI_CONST m_uint *clausesStarts, MGI_CONST m_uint *clauseOuts, MGI_GLOBAL int *newClause)
{
    MGI_GLOBAL int *iterOut = newClause + clauseOuts[MGI_GID_X];
    MGI_LOCAL MGIResolveInfo localResolve = resolve[MGI_GID_X].resolve;
    if (localResolve.resolvedSize == 0)
        return;
    m_uint firstStart = clausesStarts[localResolve.clauseOne];
    m_uint sizeFirst = clausesStarts[localResolve.clauseOne + 1] - firstStart - 1;
    m_uint secondStart = clausesStarts[localResolve.clauseTwo];
    m_uint sizeSecond = clausesStarts[localResolve.clauseTwo + 1] - secondStart - 1;

    MGI_CONST int *iter = clauses + firstStart;
    MGI_CONST int *currentEnd = iter + sizeFirst;
    MGI_CONST int *otherBegin = clauses + secondStart;
    MGI_CONST int *otherEnd = otherBegin + sizeSecond;

    while(!(otherBegin == otherEnd || iter == currentEnd)) // This actually resolves
    {
        int l1 = *iter;
        int l2 = *otherBegin;
        // TODO Make this mathematical
        if (l1 == -l2)
        {
            iter++;
            otherBegin++;
        }
        else if (l1 == l2)
        {
            iter++;
            otherBegin++;
            *iterOut = l1;
            iterOut++;
        }
        else
        {
            l1 = abs(l1);
            l2 = abs(l2);
            if (l1 < l2)
            {
                iter++;
                *iterOut = l1;
            }
            else
            {
                otherBegin++;
                *iterOut = l2;
            }
            iterOut++;
        }
    }
    while (otherBegin != otherEnd)
    {
        *iterOut = *otherBegin;
        otherBegin++;
        iterOut++;
    }
    while (iter != currentEnd)
    {
        *iterOut = *iter;
        iter++;
        iterOut++;
    }
    *iterOut = 0;
}
