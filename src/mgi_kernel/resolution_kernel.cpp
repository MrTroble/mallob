#include "MGIKernelDefs.hpp"
#include "MGIShared.hpp"

const m_uint MAX_SIZE_CACHED = 256;

// TODO Redo with sorting
bool checkIsInverseIn(int clause, MGI_IN int *literalsBegin, MGI_IN int *literalsEnd)
{
    for (MGI_IN int *iter = literalsBegin; iter != literalsEnd; iter++) // TODO Check unrolling
        if ((*iter + clause) == 0)
            return true; // SIMD?
    return false;
}

// TODO Reservoir based picking that only use different
// use MGI_LOCAL MGIReservoir in order to only have atomics from each work group

MGI_KERNEL void findResolvents(MGI_IN int *clauses, MGI_IN m_uint *clausesStarts, MGI_OUT MGIResolveInfo *toResolve)
{
    const m_uint x = MGI_GID_X;
    const m_uint position = clausesStarts[x];
    const m_uint sizeOfClause = clausesStarts[x + 1] - position;

    const m_uint sizeX = MGI_GSIZE_X;
    const m_uint y = ((MGI_GID_Y) + x + 1) % (sizeX + 1); // Only compares with a halfe turnaround (n - 1)n/2
                                                          // because otherwise we would double check
    MGI_IN int *otherBegin = clauses + clausesStarts[y];
    MGI_IN int *otherEnd = clauses + clausesStarts[y + 1];

    MGI_OUT MGIResolveInfo *localResolve = toResolve + x + (y - x - 1) * sizeX;
    localResolve->literal = 0;

    // TODO Cached version
    MGI_IN int *endIter = clauses + clausesStarts[x + 1];
    for (MGI_IN int *iter = clauses + position; iter != endIter; iter++)
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
MGI_KERNEL void findResolventsReservoir(MGI_IN int *clauses, MGI_IN m_uint *clausesStarts, MGI_OUT MGIReservoir *toResolve)
{
    const m_uint x = MGI_GID_X;
    const m_uint position = clausesStarts[x];
    const m_uint sizeOfClause = clausesStarts[x + 1] - position;

    MGI_PRIVATE MGIRng rng;
    mgiRNGInit(&rng, x, 0, 1, 117007);

    // Shuffle reservoirs
    MGI_PRIVATE MGIReservoir currentReservoir = toResolve[x];
    int literal = 0;
    const m_uint xSize = MGI_GSIZE_X; // Assume K - 1
    MGI_GLOBAL int *currentBegin = clauses + position;
    MGI_GLOBAL int *currentEnd = currentBegin + sizeOfClause;
    MGI_PRIVATE MGIResolveInfo resolve;
    resolve.clauseOne = x;
    for (m_uint i = x + 1; i <= xSize; i++)
    {
        MGI_GLOBAL int *otherBegin = clauses + clausesStarts[i];
        MGI_GLOBAL int *otherEnd = clauses + clausesStarts[i + 1];
        MGI_GLOBAL int *iter = currentBegin;
        m_uint sizeOfOther = otherEnd - otherBegin;
        int difference = sizeOfClause - sizeOfOther;
        m_uint heuristic = difference;
        resolve.literal = 0;
        resolve.clauseTwo = i;
        for (;;) // This calculates the heursitic and does merging
        {
            if (otherBegin == otherEnd || iter == currentEnd)
                break;
            int l1 = *iter;
            int l2 = *otherBegin;
            if (l1 == -l2)
            {
                if (resolve.literal != 0)
                {
                    resolve.resolvedSize = 0;
                    break;
                }
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
        float weight = (1.0f / ((float)xSize)) * (1 - (resolve.resolvedSize / maxValue));
        mgiReserviorAddSample(&currentReservoir, &rng, &resolve, weight);
    }

    toResolve[x] = currentReservoir;
}

// Merge for resolve

MGI_KERNEL void resolve(MGI_IN MGIResolveInfo *resolve, MGI_IN int *clauses, MGI_IN m_uint *clausesStarts, MGI_OUT int *newClause)
{
    MGI_OUT int *iter = newClause;
    MGIResolveInfo localResolve = *resolve;
    m_uint firstStart = clausesStarts[localResolve.clauseOne];
    m_uint sizeFirst = clausesStarts[localResolve.clauseOne + 1] - firstStart;
    m_uint secondStart = clausesStarts[localResolve.clauseTwo];
    m_uint sizeSecond = clausesStarts[localResolve.clauseTwo + 1] - secondStart;

    MGI_IN int *clausStartFirst = clauses + firstStart;
    for (m_uint x = 0; x < sizeFirst; x++)
    {
        m_uint current = clausStartFirst[x];
        if (current == localResolve.literal)
            continue;
        *iter = current;
        iter++;
    }
}
