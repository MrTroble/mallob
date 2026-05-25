
#pragma once

#include <vector>

#include "mgi/DeferredAPI.hpp"

// Manages data flow from and to the GPU.
// Owned by ForkedSatJob (same life scope as the DeferredAPI object),
// supplied to AnytimeSatClauseCommunicator by reference.
//
// TODO: Is this class autonomous? (Does it have its own thread that does the GPU interfacing?)
// Or is it controlled only from the sharing side, perhaps with an additional loop() function?
class GpuClauseInterface {

private:

    mgi::DeferredAPI& _mgi_api;

    // TODO getLoad() function or sth similar?
    // Function called from within (?)
    inline void pushClausesToGpu(mgi::span<const int> values) {
        // TODO add compression stages and use the correct kernel
        using namespace mgi;

        std::array allocations = { AllocationInfo::from(MemoryType::Constant, values), 
                                   AllocationInfo::from(MemoryType::DeviceLocal, values.size_bytes()) };
        const auto memories = _mgi_api.allocate(allocations);
    }

public:
    GpuClauseInterface(mgi::DeferredAPI& mgiApi) : _mgi_api(mgiApi) {} 

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
