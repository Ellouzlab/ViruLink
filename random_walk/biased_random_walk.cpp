#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_set>
#include <random>
#include <thread>
#include <numeric>
#include <algorithm>
#include <limits>

namespace py = pybind11;

// Structure-of-Arrays adjacency for each node
struct Adjacency {
    std::vector<int> nbrs;    // List of neighbors
    std::vector<float> wts;   // Corresponding edge weights
};

// Graph container with adjacency info + adjacency sets
struct Graph {
    std::vector<Adjacency> adjacency;                  // adjacency[u].nbrs / adjacency[u].wts
    std::vector<std::unordered_set<int>> adjacency_sets; // adjacency_sets[u] for O(1) membership checks
};

/**
 * Build adjacency in "structure of arrays" form plus a set of neighbors
 * for each node for fast membership checks.
 */
Graph build_adjacency(
    const std::vector<int> &row,
    const std::vector<int> &col,
    const std::vector<float> &weights,
    int num_nodes)
{
    Graph graph;
    graph.adjacency.resize(num_nodes);
    graph.adjacency_sets.resize(num_nodes);

    for (size_t i = 0; i < row.size(); ++i) {
        int u = row[i];
        int v = col[i];
        float w = weights[i];
        graph.adjacency[u].nbrs.push_back(v);
        graph.adjacency[u].wts.push_back(w);

        // Also insert into the adjacency set for membership checks
        graph.adjacency_sets[u].insert(v);
    }

    return graph;
}

/**
 * Perform a Node2Vec-like biased random walk for each start node (multithreaded).
 *
 * If p=1 and q=1, it effectively becomes an unbiased random walk w.r.t. the "previous node".
 *
 * Adding `walks_per_node` parameter means: for each node in `start_nodes`,
 * we generate `walks_per_node` distinct walks. The total return size is
 * `start_nodes.size() * walks_per_node`.
 */
std::vector<std::vector<int>> biased_random_walk(
    const std::vector<int> &row,
    const std::vector<int> &col,
    const std::vector<int> &start_nodes,
    const std::vector<float> &weights,
    int walk_length,
    float p,
    float q,
    int num_threads,
    int walks_per_node)
{
    // Find number of nodes (largest index in row or col, plus 1)
    int max_row = row.empty() ? 0 : *std::max_element(row.begin(), row.end());
    int max_col = col.empty() ? 0 : *std::max_element(col.begin(), col.end());
    int num_nodes = std::max(max_row, max_col) + 1;

    // Build adjacency structure-of-arrays + adjacency sets
    Graph graph = build_adjacency(row, col, weights, num_nodes);

    // We'll generate multiple walks for each start node
    const size_t total_starts = start_nodes.size();
    const size_t total_walks = total_starts * walks_per_node;
    std::vector<std::vector<int>> walks(total_walks);

    // The function each thread will run
    auto walk_function = [&](int start_idx, int end_idx) {
        // Each thread uses its own random generator
        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector<float> probs;
        probs.reserve(10000);

        for (int i = start_idx; i < end_idx; ++i) {
            // For each start node, generate multiple walks
            for (int wpn = 0; wpn < walks_per_node; ++wpn) {
                int current = start_nodes[i];
                std::vector<int> walk;
                walk.reserve(walk_length + 1);
                walk.push_back(current);

                // 'prev' node for Node2Vec transitions
                int prev = -1;

                for (int step = 0; step < walk_length; ++step) {
                    const auto &nbrs = graph.adjacency[current].nbrs;
                    const auto &wts  = graph.adjacency[current].wts;

                    if (nbrs.empty()) {
                        // No neighbors -> dead end
                        break;
                    }

                    // Build unnormalized probabilities, factoring in Node2Vec p/q
                    probs.resize(nbrs.size());
                    float sum_prob = 0.0f;

                    for (size_t k = 0; k < nbrs.size(); ++k) {
                        int next_node = nbrs[k];
                        float w = wts[k];

                        // Node2Vec bias: if we have a 'prev',
                        //   - if next_node == prev => multiply by 1/p
                        //   - else if next_node not neighbor of prev => multiply by 1/q
                        float bias = 1.0f;
                        if (prev != -1) {
                            if (next_node == prev) {
                                bias = 1.0f / p;
                            } else {
                                if (graph.adjacency_sets[prev].find(next_node) == graph.adjacency_sets[prev].end()) {
                                    bias = 1.0f / q;
                                }
                            }
                        }

                        float prob_val = w * bias;
                        probs[k] = prob_val;
                        sum_prob += prob_val;
                    }

                    if (sum_prob <= 0.0f) {
                        // All neighbors have zero probability => stop
                        break;
                    }

                    // Sample next_node using prefix-sum
                    std::uniform_real_distribution<float> dist(0.0f, sum_prob);
                    float r = dist(gen);
                    float cumsum = 0.0f;
                    size_t chosen_idx = 0;
                    for (size_t k = 0; k < probs.size(); ++k) {
                        cumsum += probs[k];
                        if (r <= cumsum) {
                            chosen_idx = k;
                            break;
                        }
                    }

                    int next_node = nbrs[chosen_idx];
                    walk.push_back(next_node);

                    // Update for next iteration
                    prev = current;
                    current = next_node;
                }

                // The global index for this walk
                size_t global_index = static_cast<size_t>(i) * walks_per_node + wpn;
                walks[global_index] = std::move(walk);
            }
        }
    };

    // Multi-threaded execution
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int chunk_size = (num_threads > 0) 
                     ? std::max(1, static_cast<int>(total_starts) / num_threads)
                     : static_cast<int>(total_starts);

    int start_idx = 0;
    for (int t = 0; t < num_threads; ++t) {
        int end_idx = (t == num_threads - 1)
                      ? static_cast<int>(total_starts)
                      : start_idx + chunk_size;
        threads.emplace_back(walk_function, start_idx, end_idx);
        start_idx = end_idx;
        if (start_idx >= static_cast<int>(total_starts)) break;
    }

    // Join threads
    for (auto &th : threads) {
        th.join();
    }

    return walks;
}

// Pybind11 module definition
PYBIND11_MODULE(biased_random_walk, m) {
    m.doc() = "Node2Vec-like random walk with p/q bias, multiple walks per node, and multithreading.";

    m.def(
        "random_walk", 
        &biased_random_walk,
        py::arg("row"),
        py::arg("col"),
        py::arg("start_nodes"),
        py::arg("weights"),
        py::arg("walk_length"),
        py::arg("p") = 1.0f,
        py::arg("q") = 1.0f,
        py::arg("num_threads") = 1,
        py::arg("walks_per_node") = 1,
        R"pbdoc(
Perform a Node2Vec-like biased random walk from each node in `start_nodes`. 
Generates `walks_per_node` walks per start node (in total, start_nodes.size() * walks_per_node walks).

Parameters
----------
row : list of int
    Source nodes for edges
col : list of int
    Destination nodes for edges
start_nodes : list of int
    Nodes from which to start the random walks
weights : list of float
    Edge weights (same length as row/col)
walk_length : int
    Number of steps in each random walk
p : float
    Return parameter (Node2Vec)
q : float
    In-out parameter (Node2Vec)
num_threads : int
    Number of threads for parallelization
walks_per_node : int
    How many walks to run from each start node

Returns
-------
List of walks
    Each element is a list of node IDs in one random walk.
)pbdoc"
    );
}
