// ---------------------------------------------------------------------------
//  OpenMP triangle sampler for ViruLink
//  Build as:  ViruLink.sampler.triangle_sampler
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <omp.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace py = pybind11;

// ────────────────────────────────────────────────────────────────────────────
using EdgeKey = std::uint64_t;

static inline EdgeKey make_key(std::uint32_t a, std::uint32_t b) {
    return (a < b) ? (static_cast<EdgeKey>(a) << 32) | b
                   : (static_cast<EdgeKey>(b) << 32) | a;
}

// ===========================================================================
//  Core sampler
// ===========================================================================
py::list sample_triangles(
        const std::vector<std::string>& nodes,
        const std::vector<std::string>& src,
        const std::vector<std::string>& dst,
        const std::vector<std::uint8_t>& lower,
        const std::vector<std::uint8_t>& upper,
        int  num_per_class,
        int  k_classes,
        int  threads          = 0,
        std::uint64_t seed    = 42)
{
    /* make sure OpenMP really uses the requested thread count */
    if (threads > 0) { omp_set_dynamic(0);  omp_set_num_threads(threads); }

    const std::size_t M = src.size(), N = nodes.size();
    if (lower.size() != M || upper.size() != M)
        throw std::runtime_error("edge / bounds size mismatch");

    // ---- 0. node-index LUT ----------------------------------------------
    std::unordered_map<std::string,std::uint32_t> idx_of;
    idx_of.reserve(N*2);
    for (std::uint32_t i=0;i<N;++i) idx_of.emplace(nodes[i], i);

    // ---- 1. buckets & neighbour table ------------------------------------
    std::vector<std::vector<std::pair<std::uint32_t,std::uint32_t>>> up_bucket(k_classes),
                                                                     lo_bucket(k_classes);
    const int DIR_UP = 0, DIR_LO = 1;
    std::vector<std::vector<std::vector<std::uint32_t>>> nbr(
            N, std::vector<std::vector<std::uint32_t>>(2*k_classes));

    std::unordered_map<EdgeKey,std::pair<std::uint8_t,std::uint8_t>> bounds;
    bounds.reserve(M*2);

    for (std::size_t i=0;i<M;++i) {
        auto u = idx_of[src[i]], v = idx_of[dst[i]];
        auto lo = lower[i],      up = upper[i];

        up_bucket[up].push_back({u,v});
        lo_bucket[lo].push_back({u,v});

        nbr[u][DIR_UP*k_classes + up].push_back(v);
        nbr[v][DIR_UP*k_classes + up].push_back(u);
        nbr[u][DIR_LO*k_classes + lo].push_back(v);
        nbr[v][DIR_LO*k_classes + lo].push_back(u);

        bounds.emplace(make_key(u,v), std::make_pair(lo,up));
    }

    // ---- 2. primary edge sample (q,r1) -----------------------------------
    std::vector<std::pair<std::uint32_t,std::uint32_t>> prim;
    prim.reserve(2*num_per_class*k_classes);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<> dist;
    for (int r=0;r<k_classes;++r) {
        if (up_bucket[r].empty() || lo_bucket[r].empty())
            throw std::runtime_error("rank bucket empty");

        auto pick=[&](auto& bucket) {
            for (int i=0;i<num_per_class;++i)
                prim.push_back(bucket[dist(rng)%bucket.size()]);
        };
        pick(up_bucket[r]);  pick(lo_bucket[r]);
    }
    std::bernoulli_distribution coin(0.5);
    for (auto& p: prim) if (coin(rng)) std::swap(p.first,p.second);

    const std::size_t P = prim.size();

    // ---- 3. parallel r2 sampling -----------------------------------------
    struct TriRec {
        std::string q,r1,r2;
        std::uint8_t b1_lo,b1_up,b2_lo,b2_up,b3_lo,b3_up;
    };
    std::vector<TriRec> out(P);

    std::atomic<bool> abort_flag{false};

#pragma omp parallel
    {
        std::mt19937_64 thr_rng(seed ^ static_cast<std::uint64_t>(omp_get_thread_num()));
        std::uniform_int_distribution<int> dir_pick(0, 2*k_classes-1);
        std::uniform_int_distribution<std::uint32_t> node_pick(0, N-1);

#pragma omp for schedule(static)
        for (std::size_t i=0;i<P;++i) {

            /* cooperative Ctrl-C check */
            if ((i & 0xFFu) == 0u &&
                !abort_flag.load(std::memory_order_relaxed) &&
                PyErr_CheckSignals() != 0)
            {
                abort_flag.store(true, std::memory_order_relaxed);
            }
            if (abort_flag.load(std::memory_order_relaxed))
                continue;

            auto q  = prim[i].first;
            auto r1 = prim[i].second;

            int key = dir_pick(thr_rng);
            int dir = key < k_classes ? DIR_UP : DIR_LO;
            int rk  = key % k_classes;

            auto& cand = nbr[q][dir*k_classes + rk];

            std::uint32_t r2 = q;
            if (!cand.empty()) {
                std::uniform_int_distribution<> cpick(0, static_cast<int>(cand.size())-1);
                for (int t=0;t<4;++t) {
                    r2 = cand[cpick(thr_rng)];
                    if (r2!=q && r2!=r1) break;
                }
            }
            if (r2==q || r2==r1)
                do{ r2=node_pick(thr_rng);} while(r2==q||r2==r1);

            auto b1 = bounds[make_key(q ,r1)];
            auto b2 = bounds[make_key(q ,r2)];
            auto b3 = bounds[make_key(r1,r2)];

            out[i] = { nodes[q], nodes[r1], nodes[r2],
                       b1.first,b1.second,
                       b2.first,b2.second,
                       b3.first,b3.second };
        }
    }   // end parallel region

    if (abort_flag.load())
        throw py::error_already_set();   // propagate KeyboardInterrupt

    // ---- 4. serial Python-level conversion --------------------------------
    py::list py_out;
    for (const auto& o : out) {
        py_out.append(py::make_tuple(
            o.q, o.r1, o.r2,
            py::make_tuple(o.b1_lo,o.b1_up),
            py::make_tuple(o.b2_lo,o.b2_up),
            py::make_tuple(o.b3_lo,o.b3_up)
        ));
    }
    return py_out;
}

// ======================================================================
PYBIND11_MODULE(triangle_sampler, m)
{
    m.doc() = "OpenMP triangle sampler for ViruLink";
    m.def("sample_triangles", &sample_triangles,
          py::arg("nodes"), py::arg("src"), py::arg("dst"),
          py::arg("lower"), py::arg("upper"),
          py::arg("num_per_class"), py::arg("k_classes"),
          py::arg("threads") = 0,
          py::arg("seed")    = 42);
}
