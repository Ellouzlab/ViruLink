/********************************************************************
 *  hypergeom.cpp  —  compute shared‑protein edge weights
 *                   (raw hypergeometric p‑value gate + %shared PCs)
 *******************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

/* ---------- 1. helpers (unchanged) -------------------------------- */
static std::vector<std::vector<uint64_t>>
build_bitset_representation(const bool* data_ptr,int N,int M){
    int blocks=(M+63)/64;
    std::vector<std::vector<uint64_t>> bitrows(N,std::vector<uint64_t>(blocks));
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for(int i=0;i<N;i++)
        for(int j=0;j<M;j++)
            if(data_ptr[i*M+j]){
                int b=j>>6;                 // j/64
                int k=j&63;                 // j%64
                bitrows[i][b]|=(1ULL<<k);
            }
    return bitrows;
}

static inline int popcount_64(uint64_t x){
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#elif defined(_MSC_VER)
    return __popcnt64(x);
#else
    int c=0; while(x){x&=x-1; ++c;} return c;
#endif
}

static inline int count_intersection(const std::vector<uint64_t>& a,
                                     const std::vector<uint64_t>& b){
    int blocks=a.size(), s=0;
    for(int i=0;i<blocks;i++) s+=popcount_64(a[i]&b[i]);
    return s;
}

/* ---------- 2. main kernel ---------------------------------------- */
/**
 *  compute_hypergeom
 *
 *  @param matrix_in   N×M bool (presence/absence)
 *  @param nthreads    OpenMP threads  (default 1)
 *  @param pval_thresh P‑value cut‑off (default 0.01)
 *
 *  @return  N×N double matrix whose entry (i,j) is
 *           0                               if p >  pval_thresh
 *           shared_pc_percent ∈ [0,1]       if p <= pval_thresh
 */
py::array_t<double>
compute_hypergeom(py::array_t<bool> matrix_in,
                  int nthreads                 = 1,
                  double pval_thresh           = 0.01)
{
    /* --- dimension checks --------------------------------------- */
    py::buffer_info buf = matrix_in.request();
    if(buf.ndim!=2) throw std::runtime_error("matrix must be 2‑D");
    int N = static_cast<int>(buf.shape[0]);
    int M = static_cast<int>(buf.shape[1]);
    const bool* data_ptr = static_cast<const bool*>(buf.ptr);

    /* --- bitset + row sums -------------------------------------- */
    auto bitrows   = build_bitset_representation(data_ptr,N,M);
    std::vector<int> row_sums(N);
    for(int i=0;i<N;i++){
        int s=0; for(auto blk:bitrows[i]) s+=popcount_64(blk);
        row_sums[i]=s;
    }

    /* --- lgamma table for log‑combinatorics --------------------- */
    std::vector<double> lg(M+1);
    for(int i=0;i<=M;i++) lg[i]=std::lgamma(i+1.0);

    auto logC=[&lg](int n,int k)->double{
        return lg[n]-lg[k]-lg[n-k];
    };
    auto hypergeom_p=[&](int c,int a,int b)->double{
        /* p‑value = Σ_{i=c}^{min(a,b)}  [ C(a,i) C(M-a, b-i) ] / C(M,b) */
        double log_denom = logC(M,b);
        double p=0.0;
        int max_i=std::min(a,b);
        for(int i=c;i<=max_i;i++){
            double log_num = logC(a,i)+logC(M-a,b-i);
            p += std::exp(log_num-log_denom);
        }
        return p;
    };

    /* --- allocate output ---------------------------------------- */
    std::vector<double> out(N*N,0.0);

#ifdef _OPENMP
    if(nthreads>0) omp_set_num_threads(nthreads);
#endif
#pragma omp parallel for schedule(dynamic)
    for(int i=0;i<N;i++){
        for(int j=i;j<N;j++){
            int  c     = count_intersection(bitrows[i],bitrows[j]);
            int  a     = row_sums[i];
            int  b     = row_sums[j];
            double p   = hypergeom_p(c,a,b);

            double w   = 0.0;
            if(p <= pval_thresh && c>0){
                w = static_cast<double>(c) /
                    static_cast<double>(std::min(a,b));   // shared‑PC %
            }
            out[i*N+j]=out[j*N+i]=w;
        }
    }

    /* --- wrap into NumPy ---------------------------------------- */
    py::array_t<double> result({N,N});
    std::memcpy(result.mutable_data(), out.data(), sizeof(double)*N*N);
    return result;
}

/* ---------- 3. module export ------------------------------------ */
PYBIND11_MODULE(hypergeom, m){
    m.doc()="Shared‑protein edge weights (hypergeom gate)";
    m.def("compute_hypergeom", &compute_hypergeom,
          py::arg("matrix_in"),
          py::arg("nthreads")    = 1,
          py::arg("pval_thresh") = 0.01,
          R"pbdoc(
Compute an N×N matrix where entry (i,j) is

    shared_protein_percent = intersect / min(k_i , k_j)   if p ≤ pval_thresh
    0.0                                                       otherwise

The p‑value is the raw cumulative hypergeometric test on the
presence/absence matrix of protein clusters (rows = genomes,
columns = PCs).)pbdoc");
}
