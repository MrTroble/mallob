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

    AllocationInfo globalAllocation = { {}, MemoryType::Global, 128};
    AllocationInfo globalWithInitalMemory = { {}, MemoryType::Global, sizeOfTestData, testData, sizeOfTestData };
    AllocationInfo deviceLocal = { {}, MemoryType::DeviceLocal, 128 };
    AllocationInfo deviceLocalWithInitalMemory = { {}, MemoryType::DeviceLocal, sizeOfTestData, testData, sizeOfTestData };
    std::vector<AllocationInfo> infos{globalAllocation, globalWithInitalMemory, deviceLocal};
    const auto memories = deferred.allocate(infos);
    for(const auto memory : memories) {
        assert(memory);
    }
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