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
MGI_KERNEL void findResolventsReservoir(MGI_CONST MGIInfo* info, MGI_CONST int *clauses, MGI_CONST m_uint *clausesStarts, MGI_GLOBAL MGIReservoir *toResolve)
{
    const m_uint x = MGI_GID_X;
    const m_uint position = clausesStarts[x];
    const m_uint sizeOfClause = clausesStarts[x + 1] - position;

    MGI_LOCAL MGIRng rng;
    mgiRNGInit(&rng, x, 0, 1, 117007);

    // TODO Use LOCAL reservoir and ONLY MERGE AT THE END! Shuffle reservoirs
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
    for (m_uint i = 0; i < amountOfOtherClauses; i++)
    {
        const m_uint index = (x + i + 1) % divider;
        MGI_CONST int *otherBegin = clauses + clausesStarts[index];
        MGI_CONST int *otherEnd = clauses + clausesStarts[index + 1];
        MGI_CONST int *iter = currentBegin;
        m_uint sizeOfOther = otherEnd - otherBegin;
        int difference = sizeOfClause - sizeOfOther;
        m_uint heuristic = difference;
        resolve.literal = 0;
        resolve.clauseTwo = index;
        resolve.resolvedSize = 0;
        for (;
             !(otherBegin == otherEnd || iter == currentEnd);) // This calculates the heursitic and does merging
        {
            int l1 = *iter;
            int l2 = *otherBegin;
            // TODO Make this mathematical
            if (l1 == -l2)
            {
                resolve.literal = abs(l1);
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
        resolve.resolvedSize = heuristic;
        float maxValue = 2000.0f; // TODO Get max number
        float weight = (resolve.literal == 0 ? 0:1) * (1.0f / ((float)amountOfOtherClauses)) * (1 - (resolve.resolvedSize / maxValue));
        mgiReserviorAddSample(&currentReservoir, &rng, &resolve, weight);
    }

    toResolve[x] = currentReservoir;
}

// Merge for resolve

MGI_KERNEL void resolve(MGI_CONST MGIResolveInfo *resolve, MGI_CONST int *clauses, MGI_CONST m_uint *clausesStarts, MGI_GLOBAL int *newClause)
{
    MGI_GLOBAL int *iter = newClause;
    MGIResolveInfo localResolve = *resolve;
    m_uint firstStart = clausesStarts[localResolve.clauseOne];
    m_uint sizeFirst = clausesStarts[localResolve.clauseOne + 1] - firstStart;
    m_uint secondStart = clausesStarts[localResolve.clauseTwo];
    m_uint sizeSecond = clausesStarts[localResolve.clauseTwo + 1] - secondStart;

    MGI_CONST int *clausStartFirst = clauses + firstStart;
    for (m_uint x = 0; x < sizeFirst; x++)
    {
        m_uint current = clausStartFirst[x];
        if (current == localResolve.literal)
            continue;
        *iter = current;
        iter++;
    }
}
