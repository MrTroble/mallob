#include "mpi.h"
#include "util/random.hpp"
#include "util/logger.hpp"
#include "util/sys/timer.hpp"
#include "comm/mympi.hpp"
#include "util/params.hpp"
#include "util/sys/process.hpp"
#include "mgi/KernelLoader.hpp"
#include "mgi/DeferredAPI.hpp"
#include "app/sat/job/gpu_clause_interface.hpp"

using namespace mgi;

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
    std::vector<AllocationInfo> infos{globalAllocation, globalWithInitalMemory, deviceLocal, deviceLocalWithInitalMemory};
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

    GpuClauseInterface gpuInterface(deferred);
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
    gpuInterface.insertClausesFromSharing(clauses);

    LOG(V2_INFO, "Finished Clause Interface test\n");
}

namespace test
{
#include "mgi_kernel/resolution_kernel.cpp"

    inline std::pair<std::vector<int>, std::vector<uint32_t>> generate(const std::vector<std::vector<int>> &values)
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
        MGI_GSIZE_X = values.size() - 1;
        return {assig, starts};
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

        std::vector<uint32_t> beginings{0, elementsPerClaus + 1, 2 * elementsPerClaus + 1};
        std::vector<MGIResolveInfo> resolves(2);
        findResolvents(clauses.data(), beginings.data(), resolves.data());
        MGIResolveInfo localResolve = resolves[0];
        assert(localResolve.clauseTwo == 1);
        assert(localResolve.clauseOne == 0);
        assert(localResolve.literal != 0);
        assert(std::abs(localResolve.literal) <= elementsPerClaus);

        {
            MGIReservoir reservoir;
            findResolventsReservoir(clauses.data(), beginings.data(), &reservoir);
            assert(reservoir.resolve.resolvedSize == 0);
            assert(reservoir.resolve.literal != 0);
            assert(reservoir.resolve.clauseOne == 0);
            assert(reservoir.resolve.clauseTwo == 1);
        }

        {
            auto [literals, ends] = generate({{1,-2, 3}, {1, 5, 6}}); // Not resolvable
            MGIReservoir reservoir{{0}, 0};
            findResolventsReservoir(literals.data(), ends.data(), &reservoir);
            assert(reservoir.resolve.literal == 0);
            resolves.clear();
            resolves.resize(2);
            findResolvents(literals.data(), ends.data(), resolves.data());
            MGIResolveInfo localResolve = resolves[0];
            assert(localResolve.literal == 0);
        }
        {
            auto [literals, ends] = generate({{1,-2, 3}, {1, 5, 6}, {1, 2, 6}}); // 1 and 3 are resolvable
            MGIReservoir reservoir{{0}, 0};
            findResolventsReservoir(literals.data(), ends.data(), &reservoir);
            assert(reservoir.resolve.literal == 2);
            assert(reservoir.resolve.resolvedSize == 3);
            assert(reservoir.resolve.clauseOne == 0);
            assert(reservoir.resolve.clauseTwo == 2);
        }
        LOG(V2_INFO, "End Kernel TESTS on HOST!\n");
    }
}

int main(int argc, char *argv[])
{

    MyMpi::init();
    Timer::init();
    int rank = MyMpi::rank(MPI_COMM_WORLD);

    Process::init(rank);

    Random::init(rand(), rand());
    Logger::init(rank, V5_DEBG);

    Parameters params;
    params.init(argc, argv);
    MyMpi::setOptions(params);

    test::testResolutionKernel();
    testRoutine();

    MPI_Finalize();
}