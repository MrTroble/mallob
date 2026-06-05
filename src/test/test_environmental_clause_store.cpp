
#include <cstdint>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <cmath>

#include "app/sat/data/environmental_clause_store.hpp"
#include "util/string_utils.hpp"
#include "util/sys/process.hpp"
#include "util/sys/thread_pool.hpp"
#include "util/random.hpp"
#include "util/logger.hpp"
#include "util/sys/timer.hpp"

void test() {
    EnvironmentalClauseStore store;
    std::vector<int> clauses = {
        -1, 2, 0,
        -2, 3, 0,
        -3, 4, 0,
        -4, 5, 0,
        -5, 1, 0,

        -11, 12, 0,
        -12, 13, 0,
        -13, 14, 0,
        -14, 15, 0,
        -15, 11, 0,
    };
    store.insert(clauses.data(), clauses.data() + clauses.size());
    assert(store.getSelection(100).size() == 30);
    assert(store.getSelection(15).size() == 15);
    assert(store.getSelection(10).size() == 9);
    assert(store.getSelection(3).size() == 3);
    assert(store.getSelection(2).size() == 0);
}

int main() {
    Timer::init();
    Random::init(rand(), rand());
    Logger::init(0, V5_DEBG);
    Process::init(0);
    ProcessWideThreadPool::init(1);

    test();
}
