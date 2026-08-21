#include "vision/tracking/matching/hungarian.h"

#include <algorithm>

namespace modeldeploy::vision::tracking {
    namespace {
        // Cost used to pad a non-square matrix into a square one. Must be much
        // larger than any real cost but smaller than the algorithm sentinel so
        // that padding cells remain selectable.
        constexpr double kPadCost = 1e7;
        // Algorithm sentinel for the Hungarian (minv / delta initialization).
        constexpr double kHungarianInf = 1e12;

        // O(n^3) Hungarian (Kuhn-Munkres) for a square n x n cost matrix.
        // Returns for each row i the assigned column index (0-based), -1 if none.
        std::vector<int> hungarian_square(const std::vector<std::vector<double>>& cost, int n) {
            std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
            std::vector<int> p(n + 1, 0), way(n + 1, 0);

            for (int i = 1; i <= n; ++i) {
                p[0] = i;
                int j0 = 0;
                std::vector<double> minv(n + 1, kHungarianInf);
                std::vector<char> used(n + 1, 0);
                do {
                    used[j0] = 1;
                    const int i0 = p[j0];
                    int j1 = -1;
                    double delta = kHungarianInf;
                    for (int j = 1; j <= n; ++j) {
                        if (!used[j]) {
                            const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                            if (cur < minv[j]) {
                                minv[j] = cur;
                                way[j] = j0;
                            }
                            if (minv[j] < delta) {
                                delta = minv[j];
                                j1 = j;
                            }
                        }
                    }
                    for (int j = 0; j <= n; ++j) {
                        if (used[j]) {
                            u[p[j]] += delta;
                            v[j] -= delta;
                        } else {
                            minv[j] -= delta;
                        }
                    }
                    j0 = j1;
                } while (p[j0] != 0);

                do {
                    const int j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                } while (j0 != 0);
            }

            std::vector<int> assign(n, -1);
            for (int j = 1; j <= n; ++j) {
                const int row = p[j] - 1;
                const int col = j - 1;
                if (row >= 0 && row < n) assign[row] = col;
            }
            return assign;
        }

        // Solver for matrices where rows <= cols. Returns (row, col) pairs,
        // one per row, minimizing total cost.
        std::vector<std::pair<int, int>> solve_rows_le_cols(
            const std::vector<std::vector<float>>& cost) {
            const int n = static_cast<int>(cost.size());
            const int m = static_cast<int>(cost[0].size());
            const int size = m;

            std::vector<std::vector<double>> square(size, std::vector<double>(size, kPadCost));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    square[i][j] = static_cast<double>(cost[i][j]);

            const auto row_assign = hungarian_square(square, size);

            std::vector<std::pair<int, int>> res;
            for (int i = 0; i < n; ++i) {
                if (row_assign[i] >= 0 && row_assign[i] < m) {
                    res.emplace_back(i, row_assign[i]);
                }
            }
            return res;
        }
    }  // namespace

    std::vector<std::pair<int, int>> linear_sum_assignment(
        const std::vector<std::vector<float>>& cost) {
        const int n = static_cast<int>(cost.size());
        std::vector<std::pair<int, int>> res;
        if (n == 0) return res;
        const int m = static_cast<int>(cost[0].size());
        if (m == 0) return res;

        if (n <= m) {
            return solve_rows_le_cols(cost);
        }

        // More rows than columns: solve on the transposed problem, then swap.
        std::vector<std::vector<float>> t(static_cast<size_t>(m),
                                          std::vector<float>(static_cast<size_t>(n)));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                t[j][i] = cost[i][j];

        const auto transposed = solve_rows_le_cols(t);
        res.reserve(transposed.size());
        for (const auto& pr : transposed) {
            res.emplace_back(pr.second, pr.first);
        }
        return res;
    }
}
