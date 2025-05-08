/********************************************************************
 *  hypergeom.cpp  ―  genome-pair hyper-geometric test (global U)
 *                    output =  c / min(a,b)  when p ≤ α, else 0
 *
 *    • Global universe  U = |keep|
 *    • Weight formula   c/min(a,b)   instead of −log10 p
 *    • Direct NumPy allocation (no capsule needed)
 *******************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>

#ifdef _OPENMP
  #include <omp.h>
#endif
namespace py = pybind11;

/* ---------- popcount -------------------------------------------------- */
static inline int pop64(uint64_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int c = 0; while (x) { x &= x - 1; ++c; } return c;
#endif
}

/* ---------- build bit-rows ------------------------------------------- */
static std::vector<std::vector<uint64_t>>
make_rows(const bool* data, int G, int P, const std::vector<int>& keep)
{
    const int blocks = (static_cast<int>(keep.size()) + 63) >> 6;
    std::vector<std::vector<uint64_t>> rows(
        G, std::vector<uint64_t>(blocks, 0ULL));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int g = 0; g < G; ++g)
        for (int m = 0; m < static_cast<int>(keep.size()); ++m)
            if (data[g * P + keep[m]])
                rows[g][m >> 6] |= 1ULL << (m & 63);

    return rows;
}

static inline int intersect(const std::vector<uint64_t>& a,
                             const std::vector<uint64_t>& b)
{
    int s = 0, n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) s += pop64(a[i] & b[i]);
    return s;
}

/* ---------- log-binomial C(n,k)  via lgamma -------------------------- */
static inline double logC(int n, int k)
{
    static std::vector<double> lg(1, 0.0);      // lg[0] = log Γ(1)

    if (n >= static_cast<int>(lg.size())) {
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            size_t old = lg.size();
            if (n >= static_cast<int>(lg.size())) {
                lg.resize(n + 1);
                for (size_t i = old; i <= static_cast<size_t>(n); ++i)
                    lg[i] = std::lgamma(static_cast<double>(i) + 1.0);
            }
        }
    }
    return lg[n] - lg[k] - lg[n - k];
}

/* ---------- log-space hyper-geom tail ------------------------------- */
static double log_hyper_tail(int c, int a, int b, int U)
{
    const double logDen  = logC(U, b);
    double       log_sum = -INFINITY;

    for (int k = c; k <= std::min(a, b); ++k) {
        double log_term = logC(a, k) + logC(U - a, b - k) - logDen;
        double hi = std::max(log_sum, log_term);
        double lo = std::min(log_sum, log_term);
        log_sum   = hi + std::log1p(std::exp(lo - hi));   // log-sum-exp
    }
    return log_sum;   // ln P[X ≥ c]
}

/* ---------- main kernel --------------------------------------------- */
py::array_t<double>
compute_hypergeom(py::array_t<bool> pa,
                  int    nthreads     = 1,
                  double pval_thresh  = 0.01,
                  double max_freq     = 0.8)
{
    /* ---- input checks ---------------------------------------- */
    auto buf = pa.request();
    if (buf.ndim != 2)
        throw std::runtime_error("matrix must be 2-D (genomes × proteins)");
    const bool* data = static_cast<const bool*>(buf.ptr);
    const int   G    = static_cast<int>(buf.shape[0]);
    const int   P    = static_cast<int>(buf.shape[1]);

    /* ---- ubiquity filter ------------------------------------- */
    std::vector<int> col_sum(P, 0);
    for (int g = 0; g < G; ++g)
        for (int p = 0; p < P; ++p)
            if (data[g * P + p]) ++col_sum[p];

    std::vector<int> keep;
    const int max_allowed = static_cast<int>(std::ceil(max_freq * G));
    for (int p = 0; p < P; ++p)
        if (col_sum[p] <= max_allowed) keep.push_back(p);

    std::cout << "[hypergeom] ubiquitous proteins filtered out: "
              << P - keep.size() << std::endl;
    if (keep.empty())
        throw std::runtime_error("All proteins exceed max_freq threshold!");

    const int U = static_cast<int>(keep.size());      // global universe size

    /* ---- bit-rows & per-genome sizes ------------------------- */
    auto rows = make_rows(data, G, P, keep);
    std::vector<int> sz(G, 0);
    for (int g = 0; g < G; ++g)
        for (auto blk : rows[g]) sz[g] += pop64(blk);

    const double log_alpha = std::log(pval_thresh);

    /* ---- allocate NumPy array (owned by Python) -------------- */
    py::array_t<double> out_arr({ G, G });
    double* out = static_cast<double*>(out_arr.request().ptr);
    std::fill(out, out + static_cast<size_t>(G) * G, 0.0);

#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#endif
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < G; ++i) {
        for (int j = i + 1; j < G; ++j) {

            int a = sz[i], b = sz[j];
            if (a == 0 || b == 0) continue;

            int c = intersect(rows[i], rows[j]);
            if (c == 0) continue;

            double log_tail = log_hyper_tail(c, a, b, U);
            if (log_tail > log_alpha || !std::isfinite(log_tail)) continue;

            double weight = static_cast<double>(c) / std::min(a, b);  // (0,1]
            out[static_cast<size_t>(i) * G + j] =
            out[static_cast<size_t>(j) * G + i] = weight;
        }
    }
    return out_arr;
}

/* ---------- module export ------------------------------------ */
PYBIND11_MODULE(hypergeom, m)
{
    m.doc() =
        "Pair-specific hyper-geometric tail (global universe).\n"
        "Returns  c/min(a,b)  if  p ≤ α, otherwise 0.";

    m.def("compute_hypergeom", &compute_hypergeom,
          py::arg("matrix_in"),
          py::arg("nthreads")    = 1,
          py::arg("pval_thresh") = 0.01,
          py::arg("max_freq")    = 0.8,
          R"pbdoc(
matrix_in   : bool ndarray (genomes × proteins)
pval_thresh : α  (default 0.01)
max_freq    : discard proteins in > max_freq·G genomes

Weight(i,j) = shared / min(k_i, k_j)  ∈ (0,1]   if p-value ≤ α
            = 0                                  otherwise
)pbdoc");
}
