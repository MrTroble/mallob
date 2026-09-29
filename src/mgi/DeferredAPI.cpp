#include "DeferredAPI.hpp"
#include "mgi_kernel/MGIShared.hpp"

namespace mgi
{
    AllocationStrategy::~AllocationStrategy() {}

    std::vector<AllocationSlab> AllocationStrategy::slabs(span<const AllocationInfo> infos, span<const MemoryRequirements> requirements) const
    {
        assert(infos.size() == requirements.size());
        std::vector<AllocationSlab> sizeValues(infos.size());
        std::transform(infos.begin(), infos.end(), sizeValues.begin(), [&,i = 0u](const auto &value) mutable
                       { const auto size = requirements[i++].size;
                         assert(size && "Allocation must be bigger then zero"); 
                         return AllocationSlab{size, value.type}; });
        return sizeValues;
    }

    bool AllocationStrategy::needsSubBuffers(span<const AllocationInfo> infos) const
    {
        return false;
    }

    std::vector<AllocationRegions> AllocationStrategy::regions(span<const AllocationInfo> infos, span<const MemoryRequirements> requirements) const
    {
        assert(infos.size() == requirements.size());
        std::vector<AllocationRegions> sizeValues(infos.size());
        std::transform(infos.begin(), infos.end(), sizeValues.begin(), [i = 0u](const auto &value) mutable
                       { return AllocationRegions{0, value.size, i++}; });
        return sizeValues;
    }
}