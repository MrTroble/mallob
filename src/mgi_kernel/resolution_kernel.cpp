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
    const m_uint x = MGI_GID_X;
    const m_uint position = clausesStarts[x];
    const m_uint sizeOfClause = clausesStarts[x + 1] - position - 1;

    MGI_LOCAL MGIRng rng;
    mgiRNGInit(&rng, x, 0, 1, 117007);

    // TODO Use LOCAL reservoir and ONLY MERGE AT THE END! Shuffle reservoirs
    // TODO Check Register pressure
    // TODO Use shared cache for clause lookups
    MGI_LOCAL MGIReservoir currentReservoir = toResolve[x];
    currentReservoir.weight = 0;
    currentReservoir.resolve.literal = 0;
    int literal = 0;
    const m_uint amountOfOtherClauses = MGI_GSIZE_Y;
    MGI_CONST int *currentBegin = clauses + position;
    MGI_CONST int *currentEnd = currentBegin + sizeOfClause;
    MGI_LOCAL MGIResolveInfo resolve;
    resolve.clauseOne = x;
    const m_uint divider = MGI_GSIZE_X;
    const float maxValue = info->maxClauseSize;
    for (m_uint i = 0; i < amountOfOtherClauses; i++)
    {
        const m_uint index = (x + i + 1) % divider;
        MGI_CONST int *otherBegin = clauses + clausesStarts[index];
        MGI_CONST int *otherEnd = clauses + clausesStarts[index + 1] - 1;
        MGI_CONST int *iter = currentBegin;
        m_uint sizeOfOther = otherEnd - otherBegin;
        m_uint heuristic = 0;
        resolve.literal = 0;
        resolve.clauseTwo = index;
        resolve.resolvedSize = 0;
        for (; !(otherBegin == otherEnd || iter == currentEnd);) // This calculates the heursitic and does merging
        {
            int l1 = *iter;
            int l2 = *otherBegin;
            // TODO Make this mathematical
            if (l1 == -l2)
            {
                resolve.literal += (resolve.literal == 0 ? abs(l1) : 0);
                iter++;
                otherBegin++;
            }
            else if (l1 == l2)
            {
                iter++;
                otherBegin++;
                heuristic++;
            }
            else
            {
                l1 = abs(l1);
                l2 = abs(l2);
                if (l1 < l2)
                {
                    iter++;
                }
                else
                {
                    otherBegin++;
                }
                heuristic++;
            }
        }
        if (otherBegin != otherEnd)
        {
            heuristic += otherEnd - otherBegin;
        }
        if (iter != currentEnd)
        {
            heuristic += currentEnd - iter;
        }
        resolve.resolvedSize = heuristic;
        float weight = (resolve.literal == 0 ? 0 : 1) * (1.0f / ((float)amountOfOtherClauses)) * (1 - (resolve.resolvedSize / maxValue));
        mgiReserviorAddSample(&currentReservoir, &rng, &resolve, weight);
    }

    toResolve[x] = currentReservoir;
}

// Merge for resolve

MGI_KERNEL void resolve(MGI_GLOBAL MGIInfo *info, MGI_CONST MGIReservoir *resolve, MGI_CONST int *clauses, 
                        MGI_CONST m_uint *clausesStarts, MGI_CONST m_uint *clauseOuts, MGI_GLOBAL int *newClause)
{
    MGI_GLOBAL int *iterOut = newClause + clauseOuts[MGI_GID_X];
    MGI_LOCAL MGIResolveInfo localResolve = resolve->resolve;
    if(localResolve.resolvedSize == 0) return;
    m_uint firstStart = clausesStarts[localResolve.clauseOne];
    m_uint sizeFirst = clausesStarts[localResolve.clauseOne + 1] - firstStart - 1;
    m_uint secondStart = clausesStarts[localResolve.clauseTwo];
    m_uint sizeSecond = clausesStarts[localResolve.clauseTwo + 1] - secondStart - 1;

    MGI_CONST int *iter = clauses + firstStart;
    MGI_CONST int *currentEnd = iter + sizeFirst;
    MGI_CONST int *otherBegin = clauses + secondStart;
    MGI_CONST int *otherEnd = otherBegin + sizeSecond;
    
    for (; !(iter == otherEnd || iter == currentEnd);) // This actually resolves
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
}
