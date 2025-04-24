#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstring>          // std::memcpy

namespace py = pybind11;

/* ================================================================
 *  compute_pairs
 *  ---------------------------------------------------------------
 *  Parameters
 *  ----------
 *  codes  : (n, L) int32 NumPy array
 *           -1  means "unknown" at that rank.
 *  dropNR : if true, omit pairs where lower == upper == NR
 *
 *  Returns
 *  -------
 *  (src_idx, tgt_idx, lower, upper)
 *      src_idx, tgt_idx : int32
 *      lower, upper     : int16   (-1 == NR)
 * ================================================================*/
py::tuple compute_pairs(const py::array_t<int32_t,
                                          py::array::c_style |
                                          py::array::forcecast> &codes,
                        bool dropNR = false) {

    /* ---------------- basic sizes ------------------ */
    if (codes.ndim() != 2)
        throw std::runtime_error("codes must be 2‑D (n × L)");

    const int64_t n = codes.shape(0);
    const int64_t L = codes.shape(1);
    const int64_t max_pairs = n * (n - 1) / 2;   // worst‑case

    /* --------------- allocate output --------------- */
    py::array_t<int32_t> src(max_pairs);
    py::array_t<int32_t> tgt(max_pairs);
    py::array_t<int16_t>  lower(max_pairs);
    py::array_t<int16_t>  upper(max_pairs);

    auto *src_ptr   = reinterpret_cast<int32_t *>(src.mutable_data());
    auto *tgt_ptr   = reinterpret_cast<int32_t *>(tgt.mutable_data());
    auto *lower_ptr = reinterpret_cast<int16_t *>(lower.mutable_data());
    auto *upper_ptr = reinterpret_cast<int16_t *>(upper.mutable_data());

    const int32_t *data = codes.data();

    /* ------------ parallel outer loop -------------- */
    int64_t global_idx = 0;

    #pragma omp parallel
    {
        std::vector<int32_t> t_src, t_tgt;
        std::vector<int16_t> t_low, t_up;

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n - 1; ++i) {
            const int32_t *row_i = data + i * L;

            for (int64_t j = i + 1; j < n; ++j) {
                const int32_t *row_j = data + j * L;

                int16_t lc = -1;        /* NR */
                int16_t uc = L - 1;     /* most specific */
                int16_t prev = -1;

                for (int16_t k = 0; k < L; ++k) {
                    int32_t a = row_i[k];
                    int32_t b = row_j[k];

                    if (a == -1 || b == -1) {      /* unknown */
                        prev = k;
                        continue;
                    }
                    if (a != b) {                  /* first strict mismatch */
                        uc = prev;
                        break;
                    }
                    lc = prev = k;                 /* still matching */
                }
                if (uc < 0) uc = -1;               /* mismatch at first rank */

                if (dropNR && lc == -1 && uc == -1)
                    continue;

                t_src.push_back(static_cast<int32_t>(i));
                t_tgt.push_back(static_cast<int32_t>(j));
                t_low.push_back(lc);
                t_up.push_back(uc);
            }
        }

        /* -------- copy thread‑local → global -------- */
        int64_t my_n = static_cast<int64_t>(t_src.size());

        #pragma omp critical
        {
            std::memcpy(src_ptr   + global_idx, t_src.data(), my_n * sizeof(int32_t));
            std::memcpy(tgt_ptr   + global_idx, t_tgt.data(), my_n * sizeof(int32_t));
            std::memcpy(lower_ptr + global_idx, t_low.data(), my_n * sizeof(int16_t));
            std::memcpy(upper_ptr + global_idx, t_up.data(), my_n * sizeof(int16_t));
            global_idx += my_n;
        }
    } /* end parallel */

    /* ------------- trim to actual size ------------- */
    src.resize({global_idx});
    tgt.resize({global_idx});
    lower.resize({global_idx});
    upper.resize({global_idx});

    return py::make_tuple(src, tgt, lower, upper);
}

/* ------------------  module  ----------------------- */
PYBIND11_MODULE(relationship_edges_cpp, m) {
    m.doc() = "Multi‑threaded taxonomy pair generator (OpenMP)";
    m.def("compute_pairs", &compute_pairs,
          py::arg("codes"), py::arg("dropNR") = false,
          "Return (src_idx, tgt_idx, lower, upper) arrays");
}
