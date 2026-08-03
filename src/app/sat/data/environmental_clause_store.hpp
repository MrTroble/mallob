
#pragma once

#include "robin_map.h"
#include "robin_set.h"
#include "util/hashing.hpp"
#include "util/logger.hpp"
#include "util/random.hpp"
#include "util/sys/proc.hpp"
#include "util/sys/threading.hpp"
#include <vector>

struct EnvironmentalClauseStore {

private:
    // raw clause literals with termination zeroes
    std::vector<int> data;
    // occurrence list: maps literal to a list of indices, where each index
    // is the position in @data where a clause with that literal begins.
    tsl::robin_map<int, std::vector<int>> occ;
    // guards all reads and writes
    Mutex mtx;

    int nbVars {0};
    SplitMix64Rng rng;

public:
    EnvironmentalClauseStore() : rng(Proc::getTid()) {}

    void insert(const int* begin, const int* end) {
        auto lock = mtx.getLock();
        int sizeBefore = data.size();
        data.insert(data.end(), begin, end);

        // build occurrence list for new clauses
        int clausePos = sizeBefore; // beginning of current clause
        for (int i = sizeBefore; i < data.size(); i++) {
            if (data[i] == 0) clausePos = i+1; // point to beginning of next clause
            else {
                const int v = std::abs(data[i]);
                nbVars = std::max(nbVars, v);
                occ[v].push_back(clausePos);
            }
        }
    }

    const std::vector<int>& getSelection(int maxLits, int nbTries = 10) {
        auto lock = mtx.getLock();
        output.clear();
        selectedClauses.clear();
        selectedVolume = 0;

        if (nbVars == 0) {
            LOG(V1_WARN, "[WARN] no clauses in store for GPU!\n");
            return output;
        }

        while (selectedVolume < 0.9 * maxLits && nbTries > 0) {
            int var = (int) std::round(rng.randomInRange(1, nbVars));
            addEnvironment(var, maxLits - selectedVolume);
            nbTries--;
        }

        // convert selection to array of zero-terminated clause literals
        for (int pos : selectedClauses) {
            for (int i = pos; data[i] != 0; i++) output.push_back(data[i]);
            output.push_back(0);
        }

        return output;
    }

private:
    std::vector<int> output;
    tsl::robin_set<int, IntHasher> selectedClauses; // positions to clauses - no duplicates
    int selectedVolume = 0; // # selected literals in total

    void addEnvironment(int var, int maxLits) {

        // stack of positions to clauses: initially the clauses where var or -var occurs
        std::vector<int> stack;
        stack.insert(stack.end(), occ[var].begin(), occ[var].end());

        // separately store positions for subsequent iteration - we want BFS, not DFS
        std::vector<int> newStack;

        // outer loop
        while (!stack.empty() && selectedVolume < maxLits) {
            // randomly shuffle processing order of clauses
            random_shuffle(stack.data(), stack.size(), [&]() {
                return rng.randomInRange(0, 1);
            });

            // process all clauses of this iteration
            while (!stack.empty()) {
                const int pos = stack.back();
                stack.pop_back();

                // clause already selected?
                if (selectedClauses.count(pos)) continue;

                // mark clause as selected
                selectedClauses.insert(pos);

                // traverse clause literals (and determine clause length)
                int clsLength = 0;
                for (int i = pos; data[i] != 0; i++) {
                    clsLength++;
                    // push (still unselected) occurrences of literal
                    for (int p : occ[std::abs(data[i])]) {
                        if (!selectedClauses.count(p)) newStack.push_back(p);
                    }
                }

                // limit exhausted?
                if (selectedVolume + clsLength+1 > maxLits) {
                    selectedClauses.erase(pos); // un-do this clause's selection
                    return;
                }

                // add clause to selection
                selectedVolume += clsLength+1;
            }

            // swap in stack for next iteration
            stack = std::move(newStack);
            newStack.clear();

        }
    }

};
