#include "mpi.h"
#include "util/random.hpp"
#include "util/logger.hpp"
#include "util/sys/timer.hpp"
#include "comm/mympi.hpp"
#include "util/params.hpp"
#include "util/sys/process.hpp"
#include "mgi/KernelLoader.hpp"
#include "mgi/DeferredAPI.hpp"

using namespace mgi;

void testRoutine() {
    DeferredAPI deferred = initMGI();
    Kernel kernel = deferred.loadKernel("mgi_kernel/test_kernel.cpp");
    assert(kernel);

    const char testData[] = "HelloThisIsTestData!";
    constexpr size_t sizeOfTestData = sizeof(testData);

    LOG(V2_INFO, "Starting Allocation Tests!\n");
    AllocationInfo globalAllocation = { {}, MemoryType::Global, 128};
    AllocationInfo globalWithInitalMemory = { {}, MemoryType::Global, sizeOfTestData, testData, sizeOfTestData };
    AllocationInfo deviceLocal = { {}, MemoryType::DeviceLocal, 128 };
    AllocationInfo deviceLocalWithInitalMemory = { {}, MemoryType::DeviceLocal, sizeOfTestData, testData, sizeOfTestData };
    std::vector<AllocationInfo> infos{globalAllocation, globalWithInitalMemory, deviceLocal, deviceLocalWithInitalMemory};
    const auto memories = deferred.allocate(infos);
    for(const auto memory : memories) {
        assert(memory);
    }
    LOG(V2_INFO, "Allocations finished!\n");
    LOG(V2_INFO, "Start Read Testing!\n");
    std::vector<ReadInfo> readInfos = { ReadInfo{sizeOfTestData} };
    {
        const auto readData = deferred.readMemory(memories[1], readInfos);
        LOG(V2_INFO, "Read Global: %s\n", (const char*)readData.ptr[0]);
        assert(strcmp((const char*)readData.ptr[0], testData) == 0);
    }
    {
        const auto readData = deferred.readMemory(memories[3], readInfos);
        LOG(V2_INFO, "Read Device Local: %s\n", (const char*)readData.ptr[0]);
        assert(strcmp((const char*)readData.ptr[0], testData) == 0);
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
        std::array readsIn = {ReadInfo{ values.size() * sizeof(uint32_t) }};
        const auto readData = deferred.readMemory(taskMems[0], readsIn);
        for (size_t i = 0; i < values.size(); i++)
        {
            const auto shaderValue = ((uint32_t*)readData.ptr[0])[i];
            assert(shaderValue == i*2);
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
}

int main(int argc, char* argv[]) {

    MyMpi::init();
    Timer::init();
    int rank = MyMpi::rank(MPI_COMM_WORLD);

    Process::init(rank);

    Random::init(rand(), rand());
    Logger::init(rank, V5_DEBG);

    Parameters params;
    params.init(argc, argv);
    MyMpi::setOptions(params);

    testRoutine();

    MPI_Finalize();
}