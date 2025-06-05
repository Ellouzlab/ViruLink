// ViruLink/ani/ani_calc.cpp

#include <omp.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// --- DATA STRUCTURES -------------------------------------------------------

struct Hit
{
    std::string q, t;
    double pident;
    int alnlen, qs, qe, qlen, tlen;
};

struct Result
{
    std::string q, t;
    double ani;
};

// --- PARSING ----------------------------------------------------------------

static bool parse_line(const std::string &line, Hit &h)
{
    std::istringstream in(line);
    std::string field;
    if (!std::getline(in, h.q, '\t'))
        return false;
    if (!std::getline(in, h.t, '\t'))
        return false;
    std::getline(in, field, '\t');
    h.pident = std::stod(field);
    std::getline(in, field, '\t');
    h.alnlen = std::stoi(field);
    // skip mismatch, gaps
    for (int i = 0; i < 2; i++)
        std::getline(in, field, '\t');
    // qstart, qend
    std::getline(in, field, '\t');
    h.qs = std::stoi(field);
    std::getline(in, field, '\t');
    h.qe = std::stoi(field);
    // skip tstart,tend,evalue,bitscore
    for (int i = 0; i < 4; i++)
        std::getline(in, field, '\t');
    // qlen, tlen
    for (int i = 0; i < 2; i++)
        std::getline(in, field, '\t');
    std::getline(in, field, '\t');
    h.qlen = std::stoi(field);
    std::getline(in, field, '\t');
    h.tlen = std::stoi(field);
    return true;
}

// --- OVERLAP FILTER --------------------------------------------------------

static std::vector<bool> filter_overlaps(const std::vector<Hit> &hits)
{
    int n = hits.size();
    std::vector<double> weight(n);
    for (int i = 0; i < n; i++)
        weight[i] = hits[i].pident * hits[i].alnlen;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b)
              { return weight[a] > weight[b]; });

    std::vector<bool> keep(n, false);
    std::vector<std::pair<int, int>> occ;
    occ.reserve(n);

    for (int i : idx)
    {
        int qs = hits[i].qs, qe = hits[i].qe;
        bool overlap = false;
        for (auto &iv : occ)
        {
            if (qs < iv.second && iv.first < qe)
            {
                overlap = true;
                break;
            }
        }
        if (!overlap)
        {
            keep[i] = true;
            occ.emplace_back(qs, qe);
        }
    }
    return keep;
}

// --- MAIN COMPUTATION ------------------------------------------------------

void m8_to_ani_cpp(const std::string &m8_path,
                   const std::string &out_path,
                   int threads = 1,
                   bool len_weight = false,
                   double min_ani = 0.0,
                   int alpha = 2)
{
    omp_set_num_threads(threads);

    std::ifstream fin(m8_path);
    if (!fin)
        throw std::runtime_error("Cannot open " + m8_path);

    std::vector<Hit> all;
    all.reserve(1 << 20);
    std::string line;
    while (std::getline(fin, line))
    {
        if (line.empty())
            continue;
        Hit h;
        if (parse_line(line, h))
            all.push_back(std::move(h));
    }
    fin.close();

    // Build (query,target) keys and sort indices
    std::vector<std::pair<std::string, std::string>> keys;
    keys.reserve(all.size());
    for (auto &h : all)
        keys.emplace_back(h.q, h.t);

    std::vector<int> order(all.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b)
              { return keys[a] < keys[b]; });

    // Identify group boundaries
    std::vector<std::pair<int, int>> groups;
    {
        int start = 0;
        for (int i = 1, N = order.size(); i < N; i++)
        {
            if (keys[order[i]] != keys[order[i - 1]])
            {
                groups.emplace_back(start, i);
                start = i;
            }
        }
        groups.emplace_back(start, order.size());
    }

    // Compute in parallel
    std::vector<Result> results(groups.size());
#pragma omp parallel for schedule(dynamic)
    for (int gi = 0; gi < (int)groups.size(); gi++)
    {
        auto [st, en] = groups[gi];
        std::vector<Hit> sub;
        sub.reserve(en - st);
        for (int k = st; k < en; k++)
            sub.push_back(all[order[k]]);

        auto keep = filter_overlaps(sub);

        double sum_pid_aln = 0;
        long sum_aln = 0;
        int min_len = std::min(sub[0].qlen, sub[0].tlen);
        for (int i = 0; i < (int)sub.size(); i++)
        {
            if (keep[i])
            {
                sum_pid_aln += sub[i].pident * sub[i].alnlen;
                sum_aln += sub[i].alnlen;
            }
        }

        double raw_ani = std::pow(sum_pid_aln / sum_aln , alpha);
        double cov_ratio = static_cast<double>(sum_aln) / min_len;
        double out_ani = len_weight ? cov_ratio : raw_ani;
        if (out_ani < min_ani)
            out_ani = min_ani;

        results[gi] = {sub[0].q, sub[0].t, out_ani};
    }

    std::ofstream fout(out_path);
    fout << "target\tsource\tweight\n";
    for (auto &r : results)
        fout << r.q << '\t' << r.t << '\t' << r.ani << '\n';
    fout.close();
}

// --- PYBIND11 MODULE -------------------------------------------------------

PYBIND11_MODULE(ani_calc, m)
{
    m.def("m8_to_ani", &m8_to_ani_cpp,
          "Compute ANI from M8 file (multi-threaded, options: len_weight,min_ani)",
          py::arg("m8_path"),
          py::arg("out_path"),
          py::arg("threads") = 1,
          py::arg("len_weight") = false,
          py::arg("min_ani") = 0.0,
          py::arg("alpha") = 2);
}
