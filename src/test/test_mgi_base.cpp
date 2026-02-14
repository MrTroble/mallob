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
    
    KernelLoader loaderOCL;
    Kernel kernel = loaderOCL.loadKernel("test");
    assert(kernel.internal == 0);
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