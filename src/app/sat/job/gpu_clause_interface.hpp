
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

public:
    GpuClauseInterface(mgi::DeferredAPI& mgiApi) : _mgi_api(mgiApi) {} 

    // Called from MPI (sharing) side
    // Could be considered the proper "start" if there is an internal thread here.
    void insertOriginalClauses(const int* data, size_t size);

    // Called from MPI (sharing) side
    void insertClausesFromSharing(const std::vector<int>& clauses);
    // Called from MPI (sharing) side
    std::vector<int> retrieveClausesToShare();


private: // ?

    // Function called from within (?)
    bool isGpuReadyForClauses();
    // TODO getLoad() function or sth similar?
    // Function called from within (?)
    void pushClausesToGpu(const std::vector<int>& clauses);
    // Called from GPU (?) / callback?
    std::vector<int> fetchClausesFromGpu();
};
