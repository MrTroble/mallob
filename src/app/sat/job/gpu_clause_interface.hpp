
#pragma once

#include <vector>

#include "mgi/DeferredAPI.hpp"
#include "mgi_kernel/MGIShared.hpp"

// Manages data flow from and to the GPU.
// Owned by ForkedSatJob (same life scope as the DeferredAPI object),
// supplied to AnytimeSatClauseCommunicator by reference.
//
// TODO: Is this class autonomous? (Does it have its own thread that does the GPU interfacing?)
// Or is it controlled only from the sharing side, perhaps with an additional loop() function?
class GpuClauseInterface {

private:
    mgi::DeferredAPI& _mgi_api;
    mgi::Kernel resolutionKernel;

    // TODO getLoad() function or sth similar?
    // Function called from within (?)
    inline void pushClausesToGpu(mgi::span<const int> values) {
        // TODO add compression stages and use the correct kernel
        using namespace mgi;

        // TODO Prefix calculations multi threaded!!!
        // THIS IS BULLSHIT!
        std::vector<uint32_t> prefixes;
        prefixes.push_back(0);
        for (auto i = std::find(values.begin(), values.end(), 0); 
                  i != values.end(); i = std::find(i + 1, values.end(), 0))
        {
            prefixes.push_back(std::distance(values.begin(), i) + 1);
        }
        const auto clauseAmount = prefixes.size();
        // Past the end
        prefixes.push_back(std::distance(values.begin(), values.end()));

        // We need n^2 / 2 to compare each to each
        const auto sizeOfY = (size_t)ceil((float)(clauseAmount) / 2.0f);
        const auto sizeOfResolventInfos = sizeOfY * clauseAmount * sizeof(MGIResolveInfo);
        std::array allocations = { AllocationInfo::from(MemoryType::Constant, values),
                                   AllocationInfo::from<uint32_t>(MemoryType::Constant, prefixes), // CTAD is bad in 17 ... :(
                                   AllocationInfo::from(MemoryType::DeviceLocal, sizeOfResolventInfos) };
        const auto memories = _mgi_api.allocate(allocations);
        
    }

public:
    GpuClauseInterface(mgi::DeferredAPI& mgiApi) : _mgi_api(mgiApi) {
        resolutionKernel = mgiApi.loadKernel("mgi_kernel/resolution_kernel.cpp");
    } 

    inline constexpr static bool canUseGPU() {
        #ifdef MALLOB_USE_GPU
            return true;
        #else
            return false;
        #endif
    }

    // Called from MPI (sharing) side
    // Could be considered the proper "start" if there is an internal thread here.
    inline void insertOriginalClauses(mgi::span<const int> values) {
        if constexpr (canUseGPU()) {
            pushClausesToGpu(values);
        }
    }

    // Called from MPI (sharing) side
    void insertClausesFromSharing(mgi::span<const int> values) {
        if constexpr (canUseGPU()) {
            pushClausesToGpu(values);
        }
    }

    // Called from MPI (sharing) side
    std::vector<int> retrieveClausesToShare();

private: // ?

    // Function called from within (?)
    bool isGpuReadyForClauses();
    
    inline mgi::Memory getFromCacheOr();

    // Called from GPU (?) / callback?
    std::vector<int> fetchClausesFromGpu();
};
