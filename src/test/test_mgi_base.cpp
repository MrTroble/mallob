#include "mpi.h"
#include "util/random.hpp"
#include "util/logger.hpp"
#include "util/sys/timer.hpp"
#include "comm/mympi.hpp"
#include "util/params.hpp"
#include "util/sys/process.hpp"
#include "mgi/KernelLoader.hpp"
#include "mgi/DeferredAPI.hpp"
#include "mgi/MGIHelper.hpp"
#include "app/sat/job/gpu_clause_interface.hpp"

using namespace mgi;

class InterfaceTestGpuClause
{
public:
    GpuClauseInterface interface;

    void testGpuPush(mgi::span<const int> value)
    {
        if (!interface._verify_gpu_resolvents)
        {
            LOG(V2_INFO, "Force on gpu resolvent verification!\n");
            interface._verify_gpu_resolvents = true;
        }
        interface.pushClausesToGpu(value);
    }

    void preTests()
    {
        const auto v = interface.fetchClausesFromGpu();
        assert(v.empty());
        interface.pushClausesToGpu(mgi::span<const int>{});
        const auto v2 = interface.fetchClausesFromGpu();
        assert(v2.empty());
    }

    std::vector<int> testWaitForTasks()
    {
        auto status = interface._mgi_api.getStatus(from(interface.lastTask)).back();
        LOG(V2_INFO, "Status: %d!\n", status);
        while (status != TaskStatus::Complete)
        {
            status = interface._mgi_api.getStatus(from(interface.lastTask)).back();
            LOG(V2_INFO, "Status: %d!\n", status);
            sleep(1);
        }
        if (interface.lastTask)
            interface._mgi_api.waitTasks(from(interface.lastTask));
        if (!interface.tasksToRetire.empty())
            interface._mgi_api.waitTasks(interface.tasksToRetire);
        return interface.fetchClausesFromGpu();
    }

    inline void print()
    {
        printDebugOutput(interface._mgi_api, interface.resolutionKernel, interface.mgiInfo);
    }

    std::vector<MGIReservoir> testAfterPush(size_t count)
    {
        std::vector<MGIReservoir> copyRes(count);
        const auto readLock = interface._mgi_api.readMemory(interface.currentReservoir, from(ReadInfo{sizeof(MGIReservoir) * count}));
        std::copy((MGIReservoir *)readLock.ptr[0], (MGIReservoir *)readLock.ptr[0] + count, copyRes.begin());
        return copyRes;
    }
};

inline std::pair<std::vector<int>, std::vector<uint32_t>> mgi_generate(const std::vector<std::vector<int>> &values, uint32_t &sizeX)
{
    std::vector<int> assig;
    std::vector<uint32_t> starts;
    for (auto &vecs : values)
    {
        const auto current = assig.size();
        starts.push_back(current);
        assig.resize(current + vecs.size() + 1);
        std::copy(vecs.begin(), vecs.end(), assig.begin() + current);
        assig[current + vecs.size()] = 0;
    }
    starts.push_back(assig.size());
    sizeX = values.size();
    starts.push_back(0); // PAGE ID at the end
    return {assig, starts};
}

void testRoutine()
{
    DeferredAPI deferred = initMGI();
    Kernel kernel = deferred.loadKernel("mgi_kernel/test_kernel.cpp");
    assert(kernel);

    const char testData[] = "HelloThisIsTestData!";
    constexpr size_t sizeOfTestData = sizeof(testData);

    LOG(V2_INFO, "Starting Allocation Tests!\n");
    AllocationInfo globalAllocation = {{}, MemoryType::Global, 128};
    AllocationInfo globalWithInitalMemory = {{}, MemoryType::Global, sizeOfTestData, testData, sizeOfTestData};
    AllocationInfo deviceLocal = {{}, MemoryType::DeviceLocal, 128};
    AllocationInfo deviceLocalWithInitalMemory = {{}, MemoryType::DeviceLocal, sizeOfTestData, testData, sizeOfTestData};
    AllocationInfo constWithInitalMemory = {{}, MemoryType::Constant, sizeOfTestData, testData, sizeOfTestData};
    std::vector<AllocationInfo> infos{globalAllocation, globalWithInitalMemory, deviceLocal, deviceLocalWithInitalMemory, constWithInitalMemory};
    const auto memories = deferred.allocate(infos);
    for (const auto memory : memories)
    {
        assert(memory);
    }
    LOG(V2_INFO, "Allocations finished!\n");
    LOG(V2_INFO, "Start Read Testing!\n");
    std::vector<ReadInfo> readInfos = {ReadInfo{sizeOfTestData}};
    {
        const auto readData = deferred.readMemory(memories[1], readInfos);
        LOG(V2_INFO, "Read Global: %s\n", (const char *)readData.ptr[0]);
        assert(strcmp((const char *)readData.ptr[0], testData) == 0);
    }
    {
        const auto readData = deferred.readMemory(memories[3], readInfos);
        LOG(V2_INFO, "Read Device Local: %s\n", (const char *)readData.ptr[0]);
        assert(strcmp((const char *)readData.ptr[0], testData) == 0);
    }
    {
        const auto readData = deferred.readMemory(memories[4], readInfos);
        LOG(V2_INFO, "Read Constant Local: %s\n", (const char *)readData.ptr[0]);
        assert(strcmp((const char *)readData.ptr[0], testData) == 0);
    }
    LOG(V2_INFO, "Read finished!\n");

    LOG(V2_INFO, "Start Task Testing!\n");
    std::vector<uint32_t> values(64);
    std::iota(values.begin(), values.end(), 0u);
    std::vector taskTestAllocs = {AllocationInfo::from<uint32_t>(MemoryType::DeviceLocal, values)};
    const auto taskMems = deferred.allocate(taskTestAllocs);
    TaskInfo taskInfo{{}, TaskType::Long, {64, 1, 1}, kernel, "test", taskMems};
    std::array taskInfos = {taskInfo};
    const auto status = deferred.queueWaitTasks(taskInfos);
    {
        std::array readsIn = {ReadInfo{values.size() * sizeof(uint32_t)}};
        const auto readData = deferred.readMemory(taskMems[0], readsIn);
        for (size_t i = 0; i < values.size(); i++)
        {
            const auto shaderValue = ((uint32_t *)readData.ptr[0])[i];
            assert(shaderValue == i * 2);
        }
    }
    const auto tasksOut = deferred.queueTasks(taskInfos);
    const auto preWait = deferred.getStatus(tasksOut);
    assert(preWait.size() == 1);
    assert(preWait[0] != TaskStatus::Error);
    deferred.waitTasks(tasksOut);
    const auto postWait = deferred.getStatus(tasksOut);
    assert(postWait.size() == 1);
    assert(postWait[0] == TaskStatus::Complete);
    LOG(V2_INFO, "Kernel Tasks finished!\n");
    LOG(V2_INFO, "Memory Write Test Start!\n");
    std::vector<uint32_t> valuesLeft(values.size() / 2, 1u);
    std::vector<uint32_t> valuesRight(values.size() / 2, 2u);
    std::vector updateInfos = {BufferUpdateInfo::from<uint32_t>(valuesLeft),
                               BufferUpdateInfo::from<uint32_t>(valuesRight, valuesLeft.size() * sizeof(uint32_t))};
    deferred.writeMemory(taskMems[0], updateInfos);
    {
        std::array readsIn = {ReadInfo{values.size() * sizeof(uint32_t)}};
        const auto readData = deferred.readMemory(taskMems[0], readsIn);
        const uint32_t *ptr = (uint32_t *)readData.ptr[0];
        for (size_t i = 0; i < valuesLeft.size(); i++)
        {
            assert(ptr[i] == 1);
        }
        for (size_t i = 0; i < valuesRight.size(); i++)
        {
            assert(ptr[i + valuesLeft.size()] == 2);
        }
    }
    LOG(V2_INFO, "Memory Write finished!\n");

    LOG(V2_INFO, "GPU Clause Interface test\n");

    {
        InterfaceTestGpuClause gpuInterface{GpuClauseInterface{deferred, Parameters(), false}};
        gpuInterface.preTests();

        // Test clauses: All positiv + All negativ
        const size_t elementsPerClaus = 16;
        std::vector<int> clauses(2 * elementsPerClaus + 1);
        clauses[elementsPerClaus] = 0;
        for (size_t i = 0; i < elementsPerClaus; i++)
        {
            clauses[i] = i + 1;
            clauses[i + elementsPerClaus + 1] = -(int)i - 1;
        }
        assert(GpuClauseInterface::canUseGPU());
        gpuInterface.testGpuPush(clauses);
        gpuInterface.testWaitForTasks();

        LOG(V2_INFO, "Finished GPU Tasks\n");

        gpuInterface.print();

        const auto reservoirsLast = gpuInterface.testAfterPush(2);
        assert(reservoirsLast[0].resolve.literal == 0);
        // Tautology should not be added to reservoir
    }
    {
        InterfaceTestGpuClause gpuInterface{GpuClauseInterface{deferred, Parameters(), false}};

        uint32_t sizeX = 0;
        auto [literals, ends] = mgi_generate({{1, -2, 3}, {1, 5, 6}, {1, 2, 6}}, sizeX); // 1 and 3 are resolvable
        gpuInterface.testGpuPush(literals);
        const auto toTest = gpuInterface.testWaitForTasks();

        std::string toTestStr = "Resolved: ";
        for (auto x : toTest)
            toTestStr += std::to_string(x) + ",";
        toTestStr += "\nFrom: ";
        for (auto x : literals)
            toTestStr += std::to_string(x) + ",";
        LOG(V2_INFO, "%s\n", toTestStr.c_str());

        assert(toTest[0] == 1);
        assert(toTest[1] == 3);
        assert(toTest[2] == 6);

        LOG(V2_INFO, "Finished GPU Tasks\n");

        gpuInterface.print();

        const auto reservoirsLast = gpuInterface.testAfterPush(3);
        assert(reservoirsLast[2].resolve.clauseOne == 2);
        assert(reservoirsLast[2].resolve.clauseTwo == 0);
        assert(reservoirsLast[2].resolve.literal == 2);
        assert(reservoirsLast[2].resolve.resolvedSize == 3);
    }
    LOG(V2_INFO, "Finished Clause Interface test\n");
}

namespace test
{
#include "mgi_kernel/resolution_kernel.cpp"
    inline std::pair<std::vector<int>, std::vector<uint32_t>> generate(const std::vector<std::vector<int>> &values)
    {
        return mgi_generate(values, MGI_GSIZE_X);
    }

    void testResolutionKernel()
    {
        LOG(V2_INFO, "Begin Kernel TESTS on HOST!\n");

        // Test clauses: All positiv + All negativ
        const size_t elementsPerClaus = 16;
        std::vector<int> clauses(2 * elementsPerClaus + 1);
        clauses[elementsPerClaus] = 0;
        for (size_t i = 0; i < elementsPerClaus; i++)
        {
            clauses[i] = i + 1;
            clauses[i + elementsPerClaus + 1] = -(int)i - 1;
        }

        std::vector<uint32_t> beginings{0, elementsPerClaus + 1, 2 * elementsPerClaus + 2};
        beginings.push_back(0); // Page ID
        std::vector<MGIResolveInfo> resolves(2);
        findResolvents(clauses.data(), beginings.data(), resolves.data());
        MGIResolveInfo localResolve = resolves[0];
        assert(localResolve.clauseTwo == 1);
        assert(localResolve.clauseOne == 0);
        assert(localResolve.literal != 0);
        assert(std::abs(localResolve.literal) <= elementsPerClaus);

        MGI_GSIZE_X = 2;
        MGI_GSIZE_Y = 1;
        MGI_GID_X = 0;

        MGIInfo mgiInfo;
        mgiInfo.maxClauseSize = 20000;

        /*
         * Self note it  is ySize = floor(n/2)
         */

        {
            MGIReservoir reservoir;
            findResolventsReservoir(&mgiInfo, clauses.data(), beginings.data(), &reservoir);
            assert(reservoir.resolve.literal == 0);
            // Tautology should not be added to reservoir
        }

        {
            auto [literals, ends] = generate({{1, -2, 3}, {1, 5, 6}}); // Not resolvable
            MGIReservoir reservoir{{0}, 0};
            findResolventsReservoir(&mgiInfo, literals.data(), ends.data(), &reservoir);
            assert(reservoir.resolve.literal == 0);
            resolves.clear();
            resolves.resize(2);
            findResolvents(literals.data(), ends.data(), resolves.data());
            MGIResolveInfo localResolve = resolves[0];
            assert(localResolve.literal == 0);
        }
        {
            MGI_GSIZE_X = 3;
            MGI_GSIZE_Y = 1;
            auto [literals, ends] = generate({{1, -2, 3}, {1, 5, 6}, {1, 2, 6}}); // 1 and 3 are resolvable
            std::array<MGIReservoir, 3> reservoir;
            for (size_t i = 0; i < reservoir.size(); i++)
            {
                MGI_GID_X = i;
                findResolventsReservoir(&mgiInfo, literals.data(), ends.data(), reservoir.data());
            }
            assert(reservoir[2].resolve.literal == 2);
            assert(reservoir[2].resolve.resolvedSize == 3);
            assert(reservoir[2].resolve.clauseOne == 2);
            assert(reservoir[2].resolve.clauseTwo == 0);

            assert(reservoir[0].resolve.literal == 0);
            assert(reservoir[1].resolve.literal == 0);

            std::vector<m_uint> outs{0, 0, 0};
            std::vector<int> values{};
            values.resize(3);
            resolve(&mgiInfo, reservoir.data(), literals.data(), ends.data(), outs.data(), values.data());
            assert(values[0] == 1);
            assert(values[1] == 3);
            assert(values[2] == 6);
        }
        {
            auto [literals, ends] = generate({{1, -2, 3}, {1, 2, 6}}); // 1 and 3 are resolvable
            MGI_GSIZE_X = 2;
            MGI_GSIZE_Y = 1;
            MGI_GID_X = 0;
            std::array<MGIReservoir, 2> reservoir;
            findResolventsReservoir(&mgiInfo, literals.data(), ends.data(), reservoir.data());
            MGI_GID_X = 1;
            findResolventsReservoir(&mgiInfo, literals.data(), ends.data(), reservoir.data());
            assert(reservoir[0].resolve.literal == 2);
            assert(reservoir[0].resolve.resolvedSize == 3);
            assert(reservoir[0].resolve.clauseOne == 0);
            assert(reservoir[0].resolve.clauseTwo == 1);

            assert(reservoir[1].resolve.literal == 2);
            assert(reservoir[1].resolve.resolvedSize == 3);
            assert(reservoir[1].resolve.clauseOne == 1);
            assert(reservoir[1].resolve.clauseTwo == 0);
        }
        LOG(V2_INFO, "End Kernel TESTS on HOST!\n");
    }
}

int main(int argc, char *argv[])
{

    // MyMpi::init();
    Timer::init();
    int rank = 0; // MyMpi::rank(MPI_COMM_WORLD);

    Process::init(rank);

    Random::init(rand(), rand());
    Logger::init(rank, V5_DEBG);

    Parameters params;
    params.init(argc, argv);
    // MyMpi::setOptions(params);

    test::testResolutionKernel();
    testRoutine();

    // MPI_Finalize();
}