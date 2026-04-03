#include "DeferredAPI.hpp"

namespace mgi
{

    std::vector<AllocationSlab> AllocationStrategy::slabs(span<const AllocationInfo> infos) const
    {
        std::vector<AllocationSlab> sizeValues(infos.size());
        std::transform(infos.begin(), infos.end(), sizeValues.begin(), [](const auto &value)
                       { return AllocationSlab{value.size, value.type}; });
        return sizeValues;
    }

    bool AllocationStrategy::needsSubBuffers(span<const AllocationInfo> infos) const
    {
        return false;
    }

    std::vector<AllocationRegions> AllocationStrategy::regions(span<const AllocationInfo> infos) const
    {
        std::vector<AllocationRegions> sizeValues(infos.size());
        std::transform(infos.begin(), infos.end(), sizeValues.begin(), [i = 0u](const auto &value) mutable
                       { return AllocationRegions{0, value.size, i++}; });
        return sizeValues;
    }
}