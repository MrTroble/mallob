
#pragma once

#include <future>
#include <limits>
#include <vector>

#include "app/sat/sharing/buffer/buffer_reader.hpp"
#include "app/sat/sharing/store/static_clause_store.hpp"
#include "mgi/DeferredAPI.hpp"
#include "mgi_kernel/MGIShared.hpp"
#include "util/sys/thread_pool.hpp"
#include "app/sat/data/environmental_clause_store.hpp"

class InterfaceTestGpuClause;

// Manages data flow from and to the GPU.
// Owned by ForkedSatJob (same life scope as the DeferredAPI object),
// supplied to AnytimeSatClauseCommunicator by reference.
class GpuClauseInterface {
friend InterfaceTestGpuClause;
private:
    mgi::DeferredAPI& _mgi_api;
    StaticClauseStore<true> _post_buffer;
    
    mgi::Kernel resolutionKernel;
    mgi::Task lastTask;
    std::vector<mgi::Task> tasksToRetire;
    size_t lastClauseAmount = 0;
    mgi::Memory currentReservoir;
    mgi::Memory mgiInfo;

    const int pageSize {65536};

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
            const auto last = prefixes.back();
            const auto current = std::distance(values.begin(), i);
            prefixes.push_back(current);
        }
        const auto clauseAmount = prefixes.size();
        // Past the end
        prefixes.push_back(std::distance(values.begin(), values.end()));

        // We need n^2 / 2 to compare each to each
        const auto sizeOfY = (size_t)floor((float)(clauseAmount) / 2.0f);
        const auto sizeOfResolventInfos = clauseAmount * sizeof(MGIResolveInfo);
        std::array allocations = { AllocationInfo::from(MemoryType::Constant, values),
                                   AllocationInfo::from<uint32_t>(MemoryType::Constant, prefixes) // CTAD is bad in 17 ... :(
                                 };
        auto memories = _mgi_api.allocate(allocations);
        if(lastClauseAmount < clauseAmount) { // Reallocate after size changes
            const auto realloc = AllocationInfo::from(MemoryType::DeviceLocal, sizeOfResolventInfos);
            const auto reservoirMemory = _mgi_api.allocate(from(realloc));
            // TODO COPY OLD
            if(currentReservoir) _mgi_api.freeObj(currentReservoir);
            currentReservoir = reservoirMemory.back();
            lastClauseAmount = clauseAmount;
        }
        memories.insert(memories.begin(), mgiInfo);
        memories.push_back(currentReservoir);
        // TODO Reuse allocation

        TaskInfo taskInfo{{}, TaskType::Long, {clauseAmount, sizeOfY, 1}};
        taskInfo.kernel = this->resolutionKernel;
        taskInfo.function = "findResolventsReservoir";
        taskInfo.descriptor.memory = memories;
        if(lastTask) {
            taskInfo.waitForTasks.push_back(lastTask);
            tasksToRetire.push_back(lastTask);
        }
        lastTask = _mgi_api.queueTasks(from(taskInfo)).back();
    }

    bool useBackgroundThreads = true;
public:
    GpuClauseInterface(mgi::DeferredAPI& mgiApi, const Parameters& params, bool useBackgroundThreads = true) : _mgi_api(mgiApi),
            _post_buffer(params, false, 256, true, 1<<20) {
        using namespace mgi;
        resolutionKernel = mgiApi.loadKernel("mgi_kernel/resolution_kernel.cpp");
        mgiInfo = mgiApi.allocate(from(AllocationInfo::from(MemoryType::Constant, sizeof(MGIInfo)))).back();
        this->useBackgroundThreads = useBackgroundThreads;
        if(useBackgroundThreads)
            launchBackgroundThreads();
    }
    ~GpuClauseInterface() {
        if(useBackgroundThreads)
            joinBackgroundThreads();
    }

    inline constexpr static bool canUseGPU() {
        #ifdef MALLOB_USE_GPU
            return true;
        #else
            return false;
        #endif
    }

    // Called from MPI (sharing) side
    void insertOriginalClauses(mgi::span<const int> values) {
        insertClausesFromSharing(values); // TODO(Dominik) any special treatment needed?
    }

    // Called from MPI (sharing) side
    void insertClausesFromSharing(mgi::span<const int> values) {
        _pre_buffer.insert(values.begin(), values.end());
    }

    // Called from MPI (sharing) side
    std::vector<int> retrieveClausesToShare(int limit) {
        int nbExportedClauses, nbExportedLits;
        return _post_buffer.exportBuffer(limit, nbExportedClauses, nbExportedLits);
    }

private:
    // Our two background workers:
    std::future<void> _fut_pre; // prepares and submits GPU tasks
    std::future<void> _fut_post; // retrieves and processes GPU results
    bool _terminate {false};

    EnvironmentalClauseStore _pre_buffer;

    void launchBackgroundThreads() {
        if (!canUseGPU()) return;
        _fut_pre = ProcessWideThreadPool::get().addTask([&]() {
            runPrepareGpuCalls();
        });
        _fut_post = ProcessWideThreadPool::get().addTask([&]() {
            runProcessGpuResults();
        });
    }
    void joinBackgroundThreads() {
        _terminate = true;
        if (_fut_pre.valid()) _fut_pre.get();
        if (_fut_post.valid()) _fut_post.get();
    }

    void runPrepareGpuCalls() {
        while (!_terminate) {
            // Occasionally prepare a page of cohesive clauses
            // from the prebuffer and forward it to the GPU.
            const auto& clauses = _pre_buffer.getSelection(pageSize);
            pushClausesToGpu(mgi::span<const int>(clauses));

            // TODO find a better periodicity / trigger
            usleep(1000 * 1000); // 1s
        }
    }
    void runProcessGpuResults() {
        while (!_terminate) {
            // Occasionally retrieve clauses from the GPU
            // and insert them into the postbuffer.
            auto clauses = fetchClausesFromGpu();
            int clausePos = 0;
            for (int i = 0; i < clauses.size(); i++) {
                if (clauses[i] == 0) {
                    _post_buffer.addClause({clauses.data() + clausePos, i-clausePos, i-clausePos});
                    clausePos = i+1;
                }
            }

            // TODO find a better periodicity / trigger
            usleep(1000 * 1000); // 1s
        }
    }
    
    inline mgi::Memory getFromCacheOr();

    // TODO(Nico) implement fetch
    // Should be called in the same thread as push
    // Not thread safe!
    std::vector<int> fetchClausesFromGpu() {
        _mgi_api.waitTasks(from(lastTask));
        
        return {};
    }
};
