// test_gpap_mgi: GPAP's engine on the GPU through MGI, checked against known answers and against
// GPAP's own OpenCL plug on the same device.
// Run from the repository root (docs/gpap.md has the build and, for WSL2, the PoCL setup):
//     build/test_gpap_mgi [instances-dir] [--compare]
// --compare also times the same probe sweeps on both plugs (compareSpeed).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gpap/gpap.h>
#include <gpap/gpu_api.hpp>

#include "app/sat/gpap/mgi_gpu.hpp"
#include "mgi/DeferredAPI.hpp"
#include "util/logger.hpp"
#include "util/random.hpp"
#include "util/sys/process.hpp"
#include "util/sys/timer.hpp"

// abort with a message on failure: plain assert may be compiled out by NDEBUG
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "CHECK failed %s:%d: %s (gpap: %s)\n", \
    __FILE__, __LINE__, #cond, gpap_last_error()); std::abort(); } } while (0)

using Pair = std::array<int, 2>;
struct Mined {
    std::vector<int> units;    // the negations of the width-1 conflicts, sorted
    std::vector<Pair> pairs;   // width-2 conflicts after folding `units`; each pair sorted, list sorted
    bool unsat = false;        // a probe stopped with GPAP_PROBE_UNSAT
    bool operator==(const Mined &o) const { return units == o.units && pairs == o.pairs && unsat == o.unsat; }
};
// (-1 2) (-1 3) (-2 -3) (-4 -5 6) (-6 7) (-6 -7), 7 variables: literals 1 and 6 fail on their own;
// with -1 and -6 folded in, exactly the pairs (2,3) and (4,5) conflict.
const Mined kTinyExpected{{-6, -1}, {{2, 3}, {4, 5}}, false};

static gpap_formula *tinyFormula() {
    const std::vector<int32_t> lits = {-1, 2, -1, 3, -2, -3, -4, -5, 6, -6, 7, -6, -7};
    const std::vector<int32_t> lens = {2, 2, 2, 3, 2, 2};
    gpap_formula *f = nullptr;
    CHECK(gpap_formula_from_clauses(7, lits.data(), lens.data(), (int32_t)lens.size(), &f) == GPAP_OK);
    return f;
}

// One sweep over every k-subset of the free literals: the conflicting tuples (each sorted, the
// list sorted), or nothing with `*unsat` set when the formula's units already conflict.
static std::vector<std::vector<int>> sweep(gpap_engine *e, int k, bool *unsat) {
    gpap_probe *p = nullptr;
    CHECK(gpap_probe_combinations(e, nullptr, 0, k, nullptr, &p) == GPAP_OK);
    gpap_probe_stats st{};
    gpap_probe_stats_get(p, &st);
    std::vector<std::vector<int>> tuples;
    if (st.stop == GPAP_PROBE_UNSAT) {
        *unsat = true;
    } else {
        CHECK(st.exhausted == 1);
        std::vector<int32_t> flat((size_t)st.conflicts * k);
        if (st.conflicts > 0) CHECK(gpap_probe_conflicts(p, flat.data(), st.conflicts) == GPAP_OK);
        for (int64_t i = 0; i < st.conflicts; i++) {
            std::vector<int> t(flat.begin() + i * k, flat.begin() + (i + 1) * k);
            std::sort(t.begin(), t.end());
            tuples.push_back(std::move(t));
        }
        std::sort(tuples.begin(), tuples.end());
    }
    gpap_probe_free(p);
    return tuples;
}

// Width 1, fold its units in, then (maxWidth >= 2) width 2.
static Mined mine(gpap_engine *e, int maxWidth) {
    Mined m;
    for (const auto &t : sweep(e, 1, &m.unsat)) m.units.push_back(-t[0]);
    std::sort(m.units.begin(), m.units.end());
    if (m.unsat) return m;
    if (!m.units.empty()) CHECK(gpap_assume_units(e, m.units.data(), (int32_t)m.units.size()) == GPAP_OK);
    if (maxWidth >= 2)
        for (const auto &t : sweep(e, 2, &m.unsat)) m.pairs.push_back({t[0], t[1]});
    return m;
}

static void testOwnPlugKnownAnswer() {
    setenv("GPAP_BACKEND", "opencl", 1);
    gpap_formula *f = tinyFormula();
    gpap_engine *e = nullptr;
    CHECK(gpap_engine_create(f, nullptr, &e) == GPAP_OK);
    CHECK(std::string(gpap_engine_backend(e)) == "opencl");
    CHECK(mine(e, 2) == kTinyExpected);
    gpap_engine_free(e);
    gpap_formula_free(f);
    LOG(V2_INFO, "testOwnPlugKnownAnswer: ok\n");
}

// GPAP's own OpenCL plug, handed in from outside through the hook.
static void testHookWithOwnPlug() {
    setenv("GPAP_BACKEND", "opencl", 1);
    gpap_formula *f = tinyFormula();
    std::unique_ptr<gpap::Gpu> g;
    CHECK(gpap::open_gpu("opencl", 0, &g) == GPAP_OK && g);
    gpap_engine *e = nullptr;
    CHECK(gpap::engine_create_on(f, nullptr, std::move(g), &e) == GPAP_OK);
    CHECK(std::string(gpap_engine_backend(e)) == "opencl");
    CHECK(mine(e, 2) == kTinyExpected);
    gpap_engine_free(e);
    gpap_engine *none = reinterpret_cast<gpap_engine *>(1);
    CHECK(gpap::engine_create_on(f, nullptr, nullptr, &none) == GPAP_ERR_ARG && none == nullptr);
    gpap_formula_free(f);
    LOG(V2_INFO, "testHookWithOwnPlug: ok\n");
}

static std::vector<int8_t> down(gpap::Gpu &g, gpap::GpuMem m, size_t offset, size_t n) {
    std::vector<int8_t> out(n);
    CHECK(g.download(out.data(), m, offset, n) == GPAP_OK);
    return out;
}

static void testMgiGpuBasics(mgi::DeferredAPI &api) {
    using gpap::buf; using gpap::i32; using gpap::GpuMem;
    const size_t kMem = size_t(1) << 30;
    std::unique_ptr<MgiGpu> missing;
    CHECK(MgiGpu::open(api, kMem, &missing, "gpap_kernel/no_such_file.cl") == GPAP_ERR_KERNEL && !missing);
    std::unique_ptr<MgiGpu> big;
    CHECK(MgiGpu::open(api, size_t(4) << 30, &big) == GPAP_OK && big->max_buffer_bytes() == (int64_t(1) << 30));
    std::unique_ptr<MgiGpu> g;
    CHECK(MgiGpu::open(api, kMem, &g) == GPAP_OK && g);
    CHECK(std::string(g->backend()) == "mgi" && std::string(g->device_name()) == "MGI");
    int64_t fr = 0, tot = 0;
    CHECK(g->mem_info(&fr, &tot) == GPAP_OK && fr == (int64_t)kMem && tot == (int64_t)kMem);
    CHECK(!g->free_is_measured() && g->resident_warps() == 0);
    CHECK(g->has_kernel("fill_i8") && g->has_kernel("fixpoint.s16.w1") && !g->has_kernel("no_such_kernel"));
    GpuMem unused = 0;
    CHECK(g->adopt(0x1000, &unused) == GPAP_ERR_ARG);
    // A buffer past max_buffer_bytes() is refused with a message: MGI would abort the process.
    CHECK(g->alloc(size_t(g->max_buffer_bytes()) + 1, &unused) == GPAP_ERR_MEMORY && unused == 0);
    CHECK(std::string(gpap_last_error()).find("past the MGI plug's") != std::string::npos);

    std::vector<int8_t> a(1024);
    for (size_t i = 0; i < a.size(); i++) a[i] = int8_t(i * 7);
    GpuMem A = 0, B = 0;
    CHECK(g->alloc(1024, &A) == GPAP_OK && g->alloc(1024, &B) == GPAP_OK);
    CHECK(g->upload(A, a.data(), a.size()) == GPAP_OK);
    CHECK(down(*g, A, 0, 1024) == a);
    CHECK(down(*g, A, 100, 100) == std::vector<int8_t>(a.begin() + 100, a.begin() + 200));
    CHECK(g->copy(B, A, 1024) == GPAP_OK && down(*g, B, 0, 1024) == a);
    CHECK(g->zero(B, 1024) == GPAP_OK && down(*g, B, 0, 1024) == std::vector<int8_t>(1024, 0));
    CHECK(g->run("fill_i8", 1000, 1, {buf(B), i32(7), i32(1000)}) == GPAP_OK);   // the engine's argument list
    std::vector<int8_t> expect(1024, 0);
    std::fill(expect.begin(), expect.begin() + 1000, 7);
    CHECK(down(*g, B, 0, 1024) == expect);

    // Empty extents and zero-byte transfers are no-ops.
    CHECK(g->run("fill_i8", 0, 1, {buf(B), i32(9), i32(0)}) == GPAP_OK);
    CHECK(g->upload(B, a.data(), 0) == GPAP_OK && g->download(a.data(), B, 0, 0) == GPAP_OK);
    CHECK(g->copy(B, A, 0) == GPAP_OK && g->zero(B, 0) == GPAP_OK && down(*g, B, 0, 1024) == expect);

    // Ten parameter blocks for one kernel, more than the 8 MgiGpu caches.
    std::vector<GpuMem> fills(10);
    for (int v = 0; v < 10; v++) {
        CHECK(g->alloc(64, &fills[v]) == GPAP_OK);
        CHECK(g->run("fill_i8", 64, 1, {buf(fills[v]), i32(v + 1), i32(64)}) == GPAP_OK);
    }
    for (int v = 0; v < 10; v++) CHECK(down(*g, fills[v], 0, 64) == std::vector<int8_t>(64, int8_t(v + 1)));

    // 20 MiB, more than the 8 MiB bounce buffer, and an offset tail. The bytes must
    // not repeat with any period that divides the chunk size, or a chunk landing in the wrong
    // place would look right: a multiplicative hash of the index.
    const size_t n = size_t(20) << 20;
    std::vector<int8_t> x(n);
    for (size_t i = 0; i < n; i++) x[i] = int8_t((uint64_t(i) * 0x9E3779B97F4A7C15ull) >> 56);
    GpuMem X = 0;
    CHECK(g->alloc(n, &X) == GPAP_OK && g->upload(X, x.data(), n) == GPAP_OK);
    CHECK(down(*g, X, 0, n) == x);
    CHECK(down(*g, X, n - 9000000, 9000000) == std::vector<int8_t>(x.end() - 9000000, x.end()));

    // An input view (an offset argument): sum_i8_to over src[40..100) with src[i] = i % 7.
    std::vector<int8_t> s(100);
    for (int i = 0; i < 100; i++) s[i] = int8_t(i % 7);
    GpuMem S = 0, O = 0;
    int32_t sum = -1;
    CHECK(g->alloc(100, &S) == GPAP_OK && g->alloc(4, &O) == GPAP_OK && g->upload(S, s.data(), 100) == GPAP_OK);
    CHECK(g->run("sum_i8_to", 1, 1, {buf(S, 40), i32(60), buf(O)}) == GPAP_OK);
    CHECK(g->download(&sum, O, 0, 4) == GPAP_OK && sum == 180);   // 295 - (5 * 21 + 0+1+2+3+4)
    CHECK(g->run("sum_i8_to", 1, 1, {buf(S), i32(100), buf(O)}) == GPAP_OK);
    CHECK(g->download(&sum, O, 0, 4) == GPAP_OK && sum == 295);   // 14 * 21 + 0 + 1
    for (GpuMem m : {A, B, X, S, O}) g->free(m);
    for (GpuMem m : fills) g->free(m);
    LOG(V2_INFO, "testMgiGpuBasics: ok\n");
}

static void testMgiKnownAnswer(mgi::DeferredAPI &api) {
    gpap_formula *f = tinyFormula();
    std::unique_ptr<MgiGpu> g;
    CHECK(MgiGpu::open(api, size_t(1) << 30, &g) == GPAP_OK);
    gpap_engine *e = nullptr;
    CHECK(gpap::engine_create_on(f, nullptr, std::move(g), &e) == GPAP_OK);
    CHECK(std::string(gpap_engine_backend(e)) == "mgi");
    CHECK(mine(e, 2) == kTinyExpected);
    gpap_engine_free(e);
    gpap_formula_free(f);
    LOG(V2_INFO, "testMgiKnownAnswer: ok\n");
}

// The clauses of a DIMACS file. Lines starting with 'c', and an incremental instance's 'a'
// (assumptions) lines, carry none; GPAP's own reader rejects the latter.
static gpap_formula *formulaFromCnf(const std::string &path) {
    FILE *in = std::fopen(path.c_str(), "r");
    CHECK(in != nullptr);
    int32_t numVars = 0, len = 0;
    std::vector<int32_t> lits, lens;
    for (int c = std::fgetc(in); c != EOF; c = std::fgetc(in)) {
        if (c == 'c' || c == 'a') {
            while (c != '\n' && c != EOF) c = std::fgetc(in);
        } else if (c == 'p') {
            CHECK(std::fscanf(in, " cnf %d %*d", &numVars) == 1);
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            std::ungetc(c, in);
            int32_t lit = 0;
            CHECK(std::fscanf(in, "%d", &lit) == 1);
            if (lit == 0) {
                lens.push_back(len);
                len = 0;
            } else {
                lits.push_back(lit);
                len++;
            }
        }
    }
    std::fclose(in);
    CHECK(numVars > 0 && len == 0 && !lens.empty());
    gpap_formula *f = nullptr;
    CHECK(gpap_formula_from_clauses(numVars, lits.data(), lens.data(), (int32_t)lens.size(), &f) == GPAP_OK);
    return f;
}

static void testMgiMatchesOwnPlug(mgi::DeferredAPI &api, const std::string &cnf, int maxWidth, gpap_mode mode) {
    setenv("GPAP_BACKEND", "opencl", 1);
    gpap_formula *f = formulaFromCnf(cnf);
    gpap_engine *own = nullptr, *viaMgi = nullptr;
    CHECK(gpap_engine_create(f, nullptr, &own) == GPAP_OK);
    std::unique_ptr<MgiGpu> g;
    CHECK(MgiGpu::open(api, size_t(1) << 30, &g) == GPAP_OK);
    CHECK(gpap::engine_create_on(f, nullptr, std::move(g), &viaMgi) == GPAP_OK);
    const Mined a = mine(own, maxWidth), b = mine(viaMgi, maxWidth);
    CHECK(a == b);
    CHECK(gpap_engine_mode(own) == mode && gpap_engine_mode(viaMgi) == mode);
    LOG(V2_INFO, "%s width<=%d: %zu units, %zu pairs, unsat=%d: identical\n", cnf.c_str(), maxWidth,
        b.units.size(), b.pairs.size(), (int)b.unsat);
    gpap_engine_free(own);
    gpap_engine_free(viaMgi);
    gpap_formula_free(f);
}

// A probe sweep: the first `maxTuples` k-subsets of the free literals, `batch` per launch. Its
// wall time, and the conflicts found.
static double sweepSeconds(gpap_engine *e, int k, int64_t maxTuples, int32_t batch, int64_t *conflicts,
                           int64_t *probed) {
    gpap_probe_options o{};
    o.max_tuples = maxTuples;
    o.batch = batch;
    const auto start = std::chrono::steady_clock::now();
    gpap_probe *p = nullptr;
    CHECK(gpap_probe_combinations(e, nullptr, 0, k, &o, &p) == GPAP_OK);
    gpap_probe_stats st{};
    gpap_probe_stats_get(p, &st);
    gpap_probe_free(p);
    *conflicts = st.conflicts;
    *probed = st.probed;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static double since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// The same probe sweeps on GPAP's own OpenCL plug and on MgiGpu, one device, one batch size: a rough
// sense of what going through MGI costs, not a benchmark. Each sweep runs once to warm up, then
// `reps` times per plug, in alternating order; the best and the median are reported.
static void compareSpeed(mgi::DeferredAPI &api, const std::string &dir) {
    setenv("GPAP_BACKEND", "opencl", 1);
    struct Case { const char *file; int width; };
    const Case cases[] = {{"incremental/entertainment08-1.cnf", 3}, {"incremental/towers05-10.cnf", 3},
                          {"incremental/roverg10-1.cnf", 3}, {"incremental/roverg10-2.cnf", 2},
                          {"r3unknown_10k.cnf", 2}};
    const int reps = 5;
    const int64_t maxTuples = 20000000;
    std::printf("\n%-22s %7s %6s %2s %10s %6s | %6s %6s | %8s %8s %8s | %7s %7s\n", "instance", "clauses",
                "mode", "k", "tuples", "batch", "create", "(mgi)", "own s", "mgi s", "(median)", "mgi/own",
                "Mtup/s");
    for (const Case &c : cases) {
        gpap_formula *f = formulaFromCnf(dir + "/" + c.file);
        auto t = std::chrono::steady_clock::now();
        gpap_engine *own = nullptr, *viaMgi = nullptr;
        CHECK(gpap_engine_create(f, nullptr, &own) == GPAP_OK);
        const double createOwn = since(t);
        t = std::chrono::steady_clock::now();
        std::unique_ptr<MgiGpu> g;
        CHECK(MgiGpu::open(api, size_t(4) << 30, &g) == GPAP_OK);
        CHECK(gpap::engine_create_on(f, nullptr, std::move(g), &viaMgi) == GPAP_OK);
        const double createMgi = since(t);
        // One batch for both, from GPAP's memory model at 512 MiB: every buffer stays under the
        // MGI plug's 1 GiB limit, and both plugs run the same launches.
        const int32_t batch = gpap_compute_k_safe(gpap_engine_num_long(own), gpap_engine_num_vars(own),
                                                  int64_t(512) << 20, 65536);

        int64_t conflictsOwn = 0, conflictsMgi = 0, probedOwn = 0, probedMgi = 0;
        sweepSeconds(own, c.width, maxTuples, batch, &conflictsOwn, &probedOwn);
        sweepSeconds(viaMgi, c.width, maxTuples, batch, &conflictsMgi, &probedMgi);
        CHECK(conflictsOwn == conflictsMgi && probedOwn == probedMgi);
        std::vector<double> tOwn, tMgi;
        for (int r = 0; r < reps; r++) {
            if (r % 2 == 0) {
                tOwn.push_back(sweepSeconds(own, c.width, maxTuples, batch, &conflictsOwn, &probedOwn));
                tMgi.push_back(sweepSeconds(viaMgi, c.width, maxTuples, batch, &conflictsMgi, &probedMgi));
            } else {
                tMgi.push_back(sweepSeconds(viaMgi, c.width, maxTuples, batch, &conflictsMgi, &probedMgi));
                tOwn.push_back(sweepSeconds(own, c.width, maxTuples, batch, &conflictsOwn, &probedOwn));
            }
            CHECK(conflictsOwn == conflictsMgi && probedOwn == probedMgi);
        }
        CHECK(gpap_engine_mode(own) == gpap_engine_mode(viaMgi));
        std::sort(tOwn.begin(), tOwn.end());
        std::sort(tMgi.begin(), tMgi.end());
        const std::string name = std::string(c.file).substr(std::string(c.file).rfind('/') + 1);
        std::printf("%-22s %7d %6s %2d %10lld %6d | %5.2fs %5.2fs | %8.3f %8.3f %8.3f | %7.2f %7.1f\n",
                    name.c_str(), gpap_engine_num_clauses(own),
                    gpap_engine_mode(own) == GPAP_MODE_WAVE ? "wave" : "scalar", c.width, (long long)probedOwn,
                    batch, createOwn, createMgi, tOwn[0], tMgi[0], tMgi[reps / 2], tMgi[0] / tOwn[0],
                    probedOwn / tOwn[0] / 1e6);
        std::fflush(stdout);
        gpap_engine_free(own);
        gpap_engine_free(viaMgi);
        gpap_formula_free(f);
    }
    std::printf("own s, mgi s: the best of %d sweeps (median: MGI's); mgi/own: best over best; Mtup/s: own plug;\n"
                "create: engine creation, MGI's includes compiling bcp.cl\n", reps);
}

int main(int argc, char *argv[]) {
    Timer::init();
    Process::init(0);
    Random::init(rand(), rand());
    Logger::init(0, V5_DEBG);

    testOwnPlugKnownAnswer();
    testHookWithOwnPlug();

    mgi::DeferredAPI api = mgi::initMGI();
    testMgiGpuBasics(api);
    testMgiKnownAnswer(api);

    std::string dir = "instances";
    bool compare = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--compare") compare = true;
        else dir = argv[i];
    }
    // Formulas under 2,048 clauses run the scalar kernels; the others the WAVE kernels.
    testMgiMatchesOwnPlug(api, dir + "/r3sat_200.cnf", 2, GPAP_MODE_SCALAR);
    testMgiMatchesOwnPlug(api, dir + "/r3unsat_200.cnf", 2, GPAP_MODE_SCALAR);
    testMgiMatchesOwnPlug(api, dir + "/r3unknown_10k.cnf", 1, GPAP_MODE_WAVE);
    // Structured instances, where probing finds clauses: 689, 567 and 5451 at width 2, 24 at width 1.
    testMgiMatchesOwnPlug(api, dir + "/incremental/entertainment08-1.cnf", 2, GPAP_MODE_SCALAR);
    testMgiMatchesOwnPlug(api, dir + "/incremental/towers05-10.cnf", 2, GPAP_MODE_SCALAR);
    testMgiMatchesOwnPlug(api, dir + "/incremental/roverg10-1.cnf", 2, GPAP_MODE_WAVE);
    testMgiMatchesOwnPlug(api, dir + "/incremental/roverg10-2.cnf", 1, GPAP_MODE_WAVE);

    LOG(V2_INFO, "test_gpap_mgi: all tests passed\n");
    if (compare) compareSpeed(api, dir);
    return 0;
}
