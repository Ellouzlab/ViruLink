#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

// -----------------------------------------------------------------------------
// 1. Bitset Representation
//    Convert the input N×M boolean array (row-major) into a vector of bitset
//    representations to speed up intersection counting.
// -----------------------------------------------------------------------------
static std::vector<std::vector<uint64_t>> build_bitset_representation(const bool* data_ptr, int N, int M) {
    int blocks = (M + 63) / 64;
    std::vector<std::vector<uint64_t>> bitrows(N, std::vector<uint64_t>(blocks, 0ULL));

    // Optionally, you can parallelize this loop if N is very large.
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (data_ptr[i * M + j]) {
                int block_idx = j / 64;
                int bit_idx = j % 64;
                bitrows[i][block_idx] |= (1ULL << bit_idx);
            }
        }
    }
    return bitrows;
}

// -----------------------------------------------------------------------------
// 2. 64-bit Popcount
// -----------------------------------------------------------------------------
static inline int popcount_64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#elif defined(_MSC_VER)
    return __popcnt64(x);
#else
    int count = 0;
    while (x) {
        x &= (x - 1);
        count++;
    }
    return count;
#endif
}

// -----------------------------------------------------------------------------
// 3. Count Intersection Between Two Bitset Rows
// -----------------------------------------------------------------------------
static inline int count_intersection(const std::vector<uint64_t>& rowi, const std::vector<uint64_t>& rowj) {
    int blocks = static_cast<int>(rowi.size());
    int intersect = 0;
    for (int b = 0; b < blocks; b++) {
        uint64_t and_val = rowi[b] & rowj[b];
        intersect += popcount_64(and_val);
    }
    return intersect;
}

// -----------------------------------------------------------------------------
// 4. Main Function: compute_hypergeom
//
// Input:  matrix_in is an N x M boolean NumPy array (rows = genomes, columns = PCs)
// Output: Returns an N x N NumPy array (double) where element [i,j] is the raw
//         hypergeometric p-value computed between genome i and genome j.
// -----------------------------------------------------------------------------
py::array_t<double> compute_hypergeom(py::array_t<bool> matrix_in, int nthreads = 1) {
    // Check input dimensions.
    py::buffer_info buf = matrix_in.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input must be a 2D (N x M) array.");
    int N = static_cast<int>(buf.shape[0]);
    int M = static_cast<int>(buf.shape[1]);
    const bool* data_ptr = static_cast<const bool*>(buf.ptr);

    // Build bitset representation and compute row sums.
    auto bitrows = build_bitset_representation(data_ptr, N, M);
    std::vector<int> row_sums(N, 0);
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (auto block : bitrows[i])
            sum += popcount_64(block);
        row_sums[i] = sum;
    }

    // Precompute lgamma table for numbers 0..M (we use lgamma(x+1) for x!)
    std::vector<double> lgamma_table(M + 1, 0.0);
    for (int i = 0; i <= M; i++) {
        lgamma_table[i] = std::lgamma(i + 1.0);
    }
    // Precompute constant part for denominator: lgamma(M+1) is lgamma_table[M]
    // We define an inline lambda for hypergeometric p-value computation.
    auto hypergeom_func = [&lgamma_table, M](int c, int a, int b) -> double {
        // Compute: log(C(M, b)) = lgamma(M+1) - lgamma(b+1) - lgamma(M - b + 1)
        double log_denom = lgamma_table[M] - lgamma_table[b] - lgamma_table[M - b];
        double p_sum = 0.0;
        int max_i = std::min(a, b);
        for (int i = c; i <= max_i; i++) {
            // Compute log(C(a, i)) = lgamma(a+1) - lgamma(i+1) - lgamma(a-i+1)
            // and    log(C(M - a, b - i)) = lgamma(M - a + 1) - lgamma(b-i+1) - lgamma(M - a - (b-i) + 1)
            double log_num = (lgamma_table[a] - lgamma_table[i] - lgamma_table[a - i]) +
                             (lgamma_table[M - a] - lgamma_table[b - i] - lgamma_table[M - a - (b - i)]);
            p_sum += std::exp(log_num - log_denom);
        }
        return p_sum;
    };

    // Allocate the output dense matrix as a flat vector of length N*N.
    std::vector<double> out_vals(N * N, 0.0);

#ifdef _OPENMP
    if (nthreads > 0)
        omp_set_num_threads(nthreads);
#endif

    // For every pair (i, j), compute the hypergeometric p-value.
    // We use the upper triangle and mirror the value to the lower triangle.
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            int intersect = count_intersection(bitrows[i], bitrows[j]);
            double pval = hypergeom_func(intersect, row_sums[i], row_sums[j]);
            out_vals[i * N + j] = pval;
            out_vals[j * N + i] = pval;
        }
    }

    // Wrap the flat vector into an N x N NumPy array.
    auto result = py::array_t<double>({N, N});
    py::buffer_info buf_out = result.request();
    double* out_ptr = static_cast<double*>(buf_out.ptr);
    std::memcpy(out_ptr, out_vals.data(), N * N * sizeof(double));
    return result;
}

// -----------------------------------------------------------------------------
// 5. Module Definition
// -----------------------------------------------------------------------------
PYBIND11_MODULE(hypergeom, m) {
    m.doc() = "Compute raw hypergeometric p-values from a presence–absence matrix based on protein sharing.";
    m.def("compute_hypergeom", &compute_hypergeom,
          py::arg("matrix_in"), py::arg("nthreads") = 1,
          "Compute raw hypergeometric p-values between all genome pairs in a 2D bool array.");
}
