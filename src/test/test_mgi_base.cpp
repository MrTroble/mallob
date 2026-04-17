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
    LOG(V2_INFO, "Start Read Testing finished!\n");
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