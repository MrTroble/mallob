
#pragma once

#include <future>
#include <limits>
#include <vector>

#include "app/sat/data/clause_metadata.hpp"
#include "app/sat/sharing/buffer/buffer_reader.hpp"
#include "app/sat/sharing/store/static_clause_store.hpp"
#include "mgi/DeferredAPI.hpp"
#include "mgi/MGIHelper.hpp"
#include "mgi_kernel/MGIShared.hpp"
#include "util/logger.hpp"
#include "util/sys/thread_pool.hpp"
#include "app/sat/data/environmental_clause_store.hpp"

class InterfaceTestGpuClause;

// Manages data flow from and to the GPU.
// Owned by ForkedSatJob (same life scope as the DeferredAPI object),
// supplied to AnytimeSatClauseCommunicator by reference.
class GpuClauseInterface
{
    friend InterfaceTestGpuClause;

private:
    mgi::DeferredAPI &_mgi_api;
    StaticClauseStore<true> _post_buffer;

    mgi::Kernel resolutionKernel;
    mgi::Task lastTask;
    std::vector<mgi::Task> tasksToRetire;
    size_t lastClauseAmount = 0;
    mgi::Memory currentReservoir;
    mgi::Memory mgiInfo;
    mgi::Memory outputResolveIndices;
    mgi::Memory outputResolve;
    std::vector<std::vector<mgi::Memory>> pagesLoaded;

    bool _verify_gpu_resolvents = false;

    const int pageSize{65536};

    // TODO getLoad() function or sth similar?
    // Function called from within (?)
    inline void pushClausesToGpu(mgi::span<const int> values)
    {
        LOG(V3_VERB, "Start push to GPU\n");
        // TODO add compression stages and use the correct kernel
        using namespace mgi;

        // TODO Prefix calculations multi threaded!!!
        std::vector<uint32_t> prefixes;
        prefixes.push_back(0);
        uint32_t maxSize = 0;
        // TODO Do not extra copy! Sort somewhere else
        std::vector<int> copy(values.begin(), values.end());
        for (auto i = std::find(values.begin(), values.end(), 0);
             i != values.end(); i = std::find(i + 1, values.end(), 0))
        {
            const auto last = prefixes.back();
            const auto current = std::distance(values.begin(), i);
            prefixes.push_back(current + 1);
            maxSize = std::max(maxSize, (uint32_t)(current - last));
            std::sort(copy.begin() + last, copy.begin() + current, [](auto valueL, auto valueR)
                      { return abs(valueL) < abs(valueR); });
        }
        maxSize *= maxSize; // Could be quadratic

        auto clauseAmount = prefixes.size();
        // Past the end
        const auto last = prefixes.back();
        if (last < values.size())
        {
            std::sort(copy.begin() + last, copy.end(), [](auto valueL, auto valueR)
                      { return abs(valueL) < abs(valueR); });
            prefixes.push_back(std::distance(values.begin(), values.end()) + 1);
        }
        else
        {
            clauseAmount--; // We have a trailing zero;
        }
        if(clauseAmount == 0) {
            LOG(V1_WARN, "No resolvents submitted to gpu!\n");
            return;
        }
        assert(clauseAmount != SIZE_MAX);

        prefixes.push_back(pagesLoaded.size()); // ADD CURRENT PAGE ID AT THE END
        // We need n^2 / 2 to compare each to each
        const auto sizeOfY = (size_t)floor((float)(clauseAmount) / 2.0f);
        const auto sizeOfResolventInfos = clauseAmount * sizeof(MGIReservoir);
        std::array allocations = {
            AllocationInfo::from<int>(MemoryType::Constant, copy),         // TODO remove copy
            AllocationInfo::from<uint32_t>(MemoryType::Constant, prefixes) // CTAD is bad in 17 ... :(
        };
        auto memories = _mgi_api.allocate(allocations);
        pagesLoaded.push_back(memories);
        if (lastClauseAmount < clauseAmount)
        { // Reallocate after size changes
            const std::array realloc = {AllocationInfo::from(MemoryType::DeviceLocal, sizeOfResolventInfos),
                                        AllocationInfo::from(MemoryType::DeviceLocal, (clauseAmount + 1) * sizeof(m_uint))};
            const auto reservoirMemory = _mgi_api.allocate(realloc);
            if (currentReservoir) {
                _mgi_api.copyMemoryWait(currentReservoir, reservoirMemory[0], from(MemoryCopyInfo{lastClauseAmount * sizeof(MGIReservoir)}));
                _mgi_api.freeObj(currentReservoir);
            }
            if (outputResolveIndices)
                _mgi_api.freeObj(outputResolveIndices);
            currentReservoir = reservoirMemory[0];
            outputResolveIndices = reservoirMemory[1];
            lastClauseAmount = clauseAmount;
        }
        _mgi_api.writeMemory(mgiInfo, from(BufferUpdateInfo::from(from(maxSize))));
        std::vector<mgi::Memory> descriptors = { mgiInfo, memories[0], memories[1], currentReservoir };
        // TODO Reuse allocation

        TaskInfo taskInfo{{}, TaskType::Long, {clauseAmount, sizeOfY, 1}};
        taskInfo.kernel = this->resolutionKernel;
        taskInfo.function = "findResolventsReservoir";
        taskInfo.descriptor.memory = descriptors;
        taskInfo.groupSizes[0] = std::min(clauseAmount, (size_t)4);
        taskInfo.groupSizes[1] = std::min(sizeOfY, (size_t)4);
        if (lastTask)
        {
            taskInfo.waitForTasks.push_back(lastTask);
            tasksToRetire.push_back(lastTask);
        }
        lastTask = _mgi_api.queueTasks(from(taskInfo)).back();
    }

    inline std::vector<int> pullResolveFromGPU()
    {
        using namespace mgi;
        if(!lastTask) {
            LOG(V1_WARN, "No resolvent task was started, therefore could not pull from GPU!\n");
            return {};
        }
        _mgi_api.waitTasks(from(lastTask));
        for (const auto t : tasksToRetire)
            _mgi_api.freeObj(t);
        _mgi_api.freeObj(lastTask);
        tasksToRetire.clear();
        lastTask = {};
        // Compute outputs
        TaskInfo taskInfo{{}, TaskType::Burst, {lastClauseAmount, 1, 1}};
        taskInfo.kernel = this->resolutionKernel;
        taskInfo.function = "clauseOuts";
        taskInfo.descriptor.memory = {mgiInfo, currentReservoir, outputResolveIndices};
        taskInfo.groupSizes[0] = std::min(lastClauseAmount, (size_t)16);
        _mgi_api.queueWaitTasks(from(taskInfo));

        ReadInfo readSize{sizeof(uint32_t), lastClauseAmount * sizeof(uint32_t)};
        uint32_t sizeRead = 0;
        {
            ReadLock lock = _mgi_api.readMemory(outputResolveIndices, from(readSize));
            sizeRead = *((uint32_t *)lock.ptr[0]);
        }
        if (sizeRead == 0)
        {
            LOG(V3_VERB, "No resolvents found!\n");
            return {};
        }
        const auto output = mgi::AllocationInfo::from(MemoryType::Global, sizeRead * sizeof(int));
        if (outputResolve)
            _mgi_api.freeObj(outputResolve); // TODO Reuse if smaller
        outputResolve = _mgi_api.allocate(from(output)).back();

        // TODO use all pages
        std::vector<TaskInfo> tasks(pagesLoaded.size());
        size_t pageIdx = 0;
        for (const auto &page : pagesLoaded)
        {
            TaskInfo resolveTask{{}, TaskType::Burst, {lastClauseAmount, 1, 1}};
            resolveTask.kernel = this->resolutionKernel;
            resolveTask.function = "resolve";
            resolveTask.descriptor.memory = {mgiInfo, currentReservoir, page[0], page[1], outputResolveIndices, outputResolve};
            resolveTask.groupSizes[0] = std::min(lastClauseAmount, (size_t)16);
            tasks[pageIdx++] = resolveTask;
        }
        _mgi_api.queueWaitTasks(tasks);
        printDebugOutput(_mgi_api, resolutionKernel, mgiInfo);

        ReadInfo readInfo{output.size};
        ReadLock lock = _mgi_api.readMemory(outputResolve, from(readInfo));
        const auto start = (int *)lock.ptr[0];
        std::vector<int> result(start, start + sizeRead);

        if(_verify_gpu_resolvents && !result.empty()) {
            LOG(V1_WARN, "Verifying GPU resolvents, this might be slow!\n");
            ReadInfo readSize{lastClauseAmount * sizeof(MGIReservoir)};
            ReadLock lock = _mgi_api.readMemory(currentReservoir, from(readSize));
            auto reservoirsCurrent = ((MGIReservoir *)lock.ptr[0]);
            
            auto startPtr = result.begin();
            uint32_t clauseIdx = 0;
            uint32_t emptyClauseCount = 0;
            for(size_t i = 0; i < result.size(); i++) {
                if(result[i] == 0) {
                    const auto endPtr = result.begin() + i;
                    auto clauseSize = std::distance(startPtr, endPtr);
                    if(clauseSize <= 0) {
                        LOG(V0_CRIT, "Produced empty clause %lu!\n", clauseIdx);
                        assert(false);
                    }
                    auto& reservoir = reservoirsCurrent[clauseIdx];
                    while(reservoir.weight == 0.0f || reservoir.resolve.literal == 0) {
                        if(clauseIdx >= lastClauseAmount) {
                            LOG(V0_CRIT, "Not enough full reservoirs with %lu empty from %lu!\n", emptyClauseCount, lastClauseAmount);
                            assert(false);
                        }
                        reservoir = reservoirsCurrent[++clauseIdx];
                        emptyClauseCount++;
                    }
                    if(reservoir.resolve.resolvedSize != clauseSize) {
                        LOG(V0_CRIT, "Produced %lu clause of size %lu but reservoir expected %u!\n", clauseIdx, clauseSize, reservoir.resolve.resolvedSize);
                        assert(false);
                    }
                    startPtr = endPtr + 1;
                    clauseIdx++;
                }
            }
        }

        for (const auto &page : pagesLoaded)
        {
            for (const auto m : page)
                _mgi_api.freeObj(m);
        }
        pagesLoaded.clear();

        return result;
    }

    bool useBackgroundThreads = true;

public:
    GpuClauseInterface(mgi::DeferredAPI &mgiApi, const Parameters &params, bool useBackgroundThreads = true) : _mgi_api(mgiApi),
                                                                                                               _post_buffer(params, false, 256, true, 1 << 20)
    {
        using namespace mgi;
        // WAIT_FOR_DEBUGGER
        resolutionKernel = mgiApi.loadKernel("mgi_kernel/resolution_kernel.cpp");
        mgiInfo = mgiApi.allocate(from(AllocationInfo::from(MemoryType::Global, sizeof(MGIInfo)))).back();
        this->useBackgroundThreads = useBackgroundThreads;
        this->_verify_gpu_resolvents = params.verifyGPUResolvents();
        if (useBackgroundThreads)
            launchBackgroundThreads();
        LOG(V3_VERB, "Finished loading GPUClauseInterface\n");
    }
    ~GpuClauseInterface()
    {
        if (useBackgroundThreads)
            joinBackgroundThreads();
    }

    inline constexpr static bool canUseGPU()
    {
#ifdef MALLOB_USE_GPU
        return true;
#else
        return false;
#endif
    }

    // Called from MPI (sharing) side
    void insertOriginalClauses(const int* begin, size_t size)
    {
        insertClausesFromSharing(begin, size);
    }

    // Called from MPI (sharing) side; variant for sharing buffer
    void insertClausesFromSharing(BufferReader& reader)
    {
        size_t nbAdded = 0;

        while (true) {
            Mallob::Clause clause = reader.getNextIncomingClause();
            if (!clause.begin) break;
            insertClausesFromSharing(
                clause.begin + ClauseMetadata::numInts(),
                clause.size - ClauseMetadata::numInts()
            );
            nbAdded++;
        }

        LOG(V2_INFO, "[GPU] pre-buf received %lu clauses from sharing for GPU\n", nbAdded);
    }
    // Called from MPI (sharing) side; variant for plain list of zero-terminated clauses
    void insertClausesFromSharing(const int* begin, size_t size)
    {
        _pre_buffer.insert(begin, begin+size);
    }

    // Called from MPI (sharing) side
    std::vector<int> retrieveClauseBufferToShare(int limit)
    {
        int nbExportedClauses, nbExportedLits;
        auto result = _post_buffer.exportBuffer(limit, nbExportedClauses, nbExportedLits);
        LOG(V2_INFO, "[GPU] post-buf yielded %lu clauses from GPU for sharing\n", nbExportedClauses);
        return result;
    }

private:
    // Our two background workers:
    std::future<void> _fut_pre;  // prepares and submits GPU tasks
    std::future<void> _fut_post; // retrieves and processes GPU results
    bool _terminate{false};

    EnvironmentalClauseStore _pre_buffer;

    void launchBackgroundThreads()
    {
        if (!canUseGPU())
            return;
        _fut_pre = ProcessWideThreadPool::get().addTask([&]()
                                                        { runPrepareGpuCalls(); });
        _fut_post = ProcessWideThreadPool::get().addTask([&]()
                                                         { runProcessGpuResults(); });
    }
    void joinBackgroundThreads()
    {
        _terminate = true;
        if (_fut_pre.valid())
            _fut_pre.get();
        if (_fut_post.valid())
            _fut_post.get();
    }

    void runPrepareGpuCalls()
    {
        while (!_terminate)
        {
            // Occasionally prepare a page of cohesive clauses
            // from the prebuffer and forward it to the GPU.
            const auto &clauses = _pre_buffer.getSelection(pageSize);
            pushClausesToGpu(mgi::span<const int>(clauses));

            // TODO find a better periodicity / trigger
            usleep(1000 * 1000); // 1s
        }
    }
    void runProcessGpuResults()
    {
        while (!_terminate)
        {
            // Occasionally retrieve clauses from the GPU
            // and insert them into the postbuffer.
            auto clauses = fetchClausesFromGpu();
            int clausePos = 0;
            for (int i = 0; i < clauses.size(); i++)
            {
                if (clauses[i] == 0)
                {
                    _post_buffer.addClause({clauses.data() + clausePos, i - clausePos, i - clausePos});
                    clausePos = i + 1;
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
    std::vector<int> fetchClausesFromGpu()
    {
        return pullResolveFromGPU();
    }
};
