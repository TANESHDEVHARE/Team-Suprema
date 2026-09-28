#include "sparse_ldl.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

// ------------------------------------------------------------------ ordering
//
// Minimum degree on the quotient graph. Each uneliminated variable i keeps
//   adj[i]   : variable neighbours not yet covered by an element
//   elems[i] : elements (eliminated pivots) it belongs to
// Eliminating p turns it into an element whose variable set is
//   L_p = adj[p] U (union of L_e for e in elems[p]) \ {p}
// and absorbs the elements of elems[p]. Degrees are AMD's approximate
// external degree  |adj[i]| + |L_p \ i| + sum_{e != p} |L_e \ L_p|.
std::vector<int> min_degree_order(const SymMatrix& K, const std::atomic<bool>* cancel) {
    const int n = K.n;
    std::vector<std::vector<int>> adj(n), elems(n), evars(n);
    for (int j = 0; j < n; ++j)
        for (int p = K.colptr[j]; p < K.colptr[j + 1]; ++p) {
            int i = K.rowidx[p];
            if (i == j) continue;
            adj[i].push_back(j); adj[j].push_back(i);
        }
    for (auto& a : adj) { std::sort(a.begin(), a.end()); a.erase(std::unique(a.begin(), a.end()), a.end()); }

    std::vector<int> degree(n), stamp(n, 0), wext(n, -1);
    std::vector<char> eliminated(n, 0), dead(n, 0), inL(n, 0);
    using QE = std::pair<int, int>;      // (degree, node), lazily invalidated
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    for (int i = 0; i < n; ++i) { degree[i] = (int)adj[i].size(); pq.push({degree[i], i}); }

    std::vector<int> order;
    order.reserve(n);
    std::vector<int> L;
    for (int k = 0; k < n; ++k) {
        if ((k & 255) == 0 && cancel && cancel->load(std::memory_order_relaxed)) {
            // Cancelled: finish with the remaining nodes in index order (a
            // valid permutation; the caller discards the factorization).
            for (int i = 0; i < n; ++i) if (!eliminated[i]) order.push_back(i);
            break;
        }
        int p = -1;
        while (!pq.empty()) {
            auto [d, i] = pq.top(); pq.pop();
            if (!eliminated[i] && d == degree[i]) { p = i; break; }
        }
        if (p < 0) break;
        eliminated[p] = 1;
        order.push_back(p);

        // L_p
        L.clear();
        for (int j : adj[p]) if (!eliminated[j] && !inL[j]) { inL[j] = 1; L.push_back(j); }
        for (int e : elems[p]) {
            if (dead[e]) continue;
            for (int j : evars[e]) if (j != p && !eliminated[j] && !inL[j]) { inL[j] = 1; L.push_back(j); }
            dead[e] = 1;                               // absorbed into p
            std::vector<int>().swap(evars[e]);
        }
        evars[p] = L;
        std::vector<int>().swap(adj[p]);
        std::vector<int>().swap(elems[p]);

        // |L_e \ L_p| for every live element touching L_p
        for (int i : L)
            for (int e : elems[i]) {
                if (dead[e]) continue;
                if (wext[e] < 0) wext[e] = (int)evars[e].size();
                wext[e]--;
            }
        for (int i : L) {
            // elements: drop absorbed ones, add p
            auto& Ei = elems[i];
            size_t w = 0;
            for (size_t t = 0; t < Ei.size(); ++t) if (!dead[Ei[t]]) Ei[w++] = Ei[t];
            Ei.resize(w);
            Ei.push_back(p);
            // variables: drop eliminated ones and those now covered by element p
            auto& Ai = adj[i];
            w = 0;
            for (size_t t = 0; t < Ai.size(); ++t) { int j = Ai[t]; if (!eliminated[j] && !inL[j]) Ai[w++] = j; }
            Ai.resize(w);
            long long d = (long long)Ai.size() + (long long)L.size() - 1;
            for (int e : Ei) if (e != p) d += std::max(0, wext[e]);
            d = std::min<long long>(d, n - k - 1);
            degree[i] = (int)d;
            pq.push({degree[i], i});
        }
        for (int i : L) {
            inL[i] = 0;
            for (int e : elems[i]) wext[e] = -1;
        }
    }
    return order;
}

// ------------------------------------------------------------ factorization
//
// Supernodal left-looking LDL^T. Columns of L with nested patterns (column
// j-1 = {j} + column j) are grouped into supernodes; each supernode is a dense
// column-major block of (w + r) x w values (w columns, r rows below the
// diagonal block). Factoring a supernode = assemble its columns of K, subtract
// the dense updates of every descendant supernode that reaches it, then a
// dense LDL^T of the diagonal block with inertia control.

void SparseLDL::analyze(const SymMatrix& K) {
    n_ = K.n;
    perm_ = min_degree_order(K, cancel_);
    if (cancelled()) return;
    iperm_.assign(n_, 0);
    for (int k = 0; k < n_; ++k) iperm_[perm_[k]] = k;

    // Permuted matrix twice: upper by columns (for the elimination tree) and
    // lower by columns (for assembly), each entry remembering its position in
    // K.val so refactorizations are a gather.
    std::vector<int> cu(n_ + 1, 0), cl(n_ + 1, 0);
    for (int j = 0; j < n_; ++j)
        for (int p = K.colptr[j]; p < K.colptr[j + 1]; ++p) {
            int a = iperm_[K.rowidx[p]], b = iperm_[j];
            cu[std::max(a, b) + 1]++;
            cl[std::min(a, b) + 1]++;
        }
    Cp_.assign(n_ + 1, 0); Lcp_.assign(n_ + 1, 0);
    for (int k = 0; k < n_; ++k) { Cp_[k + 1] = Cp_[k] + cu[k + 1]; Lcp_[k + 1] = Lcp_[k] + cl[k + 1]; }
    Ci_.assign(Cp_[n_], 0); map_.assign(Cp_[n_], 0);
    Lci_.assign(Lcp_[n_], 0); Lmap_.assign(Lcp_[n_], 0);
    {
        std::vector<int> nu(Cp_.begin(), Cp_.end() - 1), nl(Lcp_.begin(), Lcp_.end() - 1);
        for (int j = 0; j < n_; ++j)
            for (int p = K.colptr[j]; p < K.colptr[j + 1]; ++p) {
                int a = iperm_[K.rowidx[p]], b = iperm_[j];
                int hi = std::max(a, b), lo = std::min(a, b);
                int q = nu[hi]++; Ci_[q] = lo; map_[q] = p;
                int r = nl[lo]++; Lci_[r] = hi; Lmap_[r] = p;
            }
    }

    // Elimination tree and column counts (Davis, "Direct Methods", 4.4).
    parent_.assign(n_, -1);
    std::vector<int> flag(n_), lnz(n_, 0);
    for (int k = 0; k < n_; ++k) {
        flag[k] = k;
        for (int p = Cp_[k]; p < Cp_[k + 1]; ++p)
            for (int i = Ci_[p]; i < k && flag[i] != k; i = parent_[i]) {
                if (parent_[i] == -1) parent_[i] = k;
                lnz[i]++;
                flag[i] = k;
            }
    }
    // Column patterns of L (row reach of every row k, sorted by construction).
    std::vector<int> Lp(n_ + 1, 0);
    for (int k = 0; k < n_; ++k) Lp[k + 1] = Lp[k] + lnz[k];
    nnzL_ = Lp[n_];
    std::vector<int> Li(Lp[n_]), fill(Lp.begin(), Lp.end() - 1);
    for (int k = 0; k < n_; ++k) {
        flag[k] = k;
        for (int p = Cp_[k]; p < Cp_[k + 1]; ++p)
            for (int i = Ci_[p]; i < k && flag[i] != k; i = parent_[i]) { Li[fill[i]++] = k; flag[i] = k; }
    }

    // Supernodes: j joins j-1's supernode when column j-1 = {j} + column j.
    sn_start_.clear();
    sn_of_.assign(n_, 0);
    for (int j = 0; j < n_; ++j) {
        bool merge = j > 0 && parent_[j - 1] == j && lnz[j - 1] == lnz[j] + 1 && j - sn_start_.back() < 128;
        if (!merge) sn_start_.push_back(j);
        sn_of_[j] = (int)sn_start_.size() - 1;
    }
    const int ns = (int)sn_start_.size();
    sn_start_.push_back(n_);
    sn_rowptr_.assign(ns + 1, 0);
    sn_valptr_.assign(ns + 1, 0);
    sn_rows_.clear();
    for (int s = 0; s < ns; ++s) {
        int f = sn_start_[s], l = sn_start_[s + 1], w = l - f;
        for (int p = Lp[f] + (w - 1); p < Lp[f + 1]; ++p) sn_rows_.push_back(Li[p]);   // rows >= l
        sn_rowptr_[s + 1] = (int)sn_rows_.size();
        long long r = sn_rowptr_[s + 1] - sn_rowptr_[s];
        sn_valptr_[s + 1] = sn_valptr_[s] + (w + r) * (long long)w;
    }
    sn_val_.assign((size_t)sn_valptr_[ns], 0.0);
    D_.assign(n_, 0.0);
}

int SparseLDL::factor(const SymMatrix& K, const std::vector<signed char>& sign, double reg) {
    const int ns = (int)sn_start_.size() - 1;
    std::fill(sn_val_.begin(), sn_val_.end(), 0.0);
    regularized_.clear();
    std::vector<int> rel(n_, -1), head(ns, -1), next(ns, -1), ptr(ns, 0);
    std::vector<double> wk, tmp;
    for (int s = 0; s < ns; ++s) {
        if ((s & 15) == 0 && cancelled()) return -1;
        const int f = sn_start_[s], l = sn_start_[s + 1], w = l - f;
        const int* R = sn_rows_.data() + sn_rowptr_[s];
        const int r = sn_rowptr_[s + 1] - sn_rowptr_[s], nr = w + r;
        double* B = sn_val_.data() + sn_valptr_[s];
        for (int i = 0; i < w; ++i) rel[f + i] = i;
        for (int q = 0; q < r; ++q) rel[R[q]] = w + q;
        // assemble columns f..l-1 of K
        for (int j = f; j < l; ++j)
            for (int p = Lcp_[j]; p < Lcp_[j + 1]; ++p) B[rel[Lci_[p]] + (j - f) * nr] += K.val[Lmap_[p]];
        // updates from descendant supernodes whose next row falls in [f, l)
        for (int d = head[s], nd; d != -1; d = nd) {
            nd = next[d];
            const int fd = sn_start_[d], wd = sn_start_[d + 1] - fd;
            const int* Rd = sn_rows_.data() + sn_rowptr_[d];
            const int rd = sn_rowptr_[d + 1] - sn_rowptr_[d], nrd = wd + rd;
            const double* Ld = sn_val_.data() + sn_valptr_[d];
            const int start = ptr[d];
            int mid = start;
            while (mid < rd && Rd[mid] < l) ++mid;
            wk.resize(wd);
            tmp.resize(rd);
            for (int q = start; q < mid; ++q) {            // target column Rd[q] of s
                for (int k = 0; k < wd; ++k) wk[k] = Ld[wd + q + k * nrd] * D_[fd + k];
                std::fill(tmp.begin() + q, tmp.begin() + rd, 0.0);
                for (int k = 0; k < wd; ++k) {
                    const double wkk = wk[k];
                    if (wkk == 0.0) continue;
                    const double* col = Ld + k * nrd + wd;
                    for (int i = q; i < rd; ++i) tmp[i] += col[i] * wkk;
                }
                double* Bc = B + (Rd[q] - f) * nr;
                for (int i = q; i < rd; ++i) Bc[rel[Rd[i]]] -= tmp[i];
            }
            ptr[d] = mid;
            if (mid < rd) { int t = sn_of_[Rd[mid]]; next[d] = head[t]; head[t] = d; }
        }
        // dense LDL^T of the block with inertia control
        for (int k = 0; k < w; ++k) {
            double dk = B[k + k * nr];
            const double sg = sign[perm_[f + k]];
            if (!(sg * dk >= reg)) { dk = sg * std::max(std::fabs(dk), reg); regularized_.push_back(perm_[f + k]); }
            D_[f + k] = dk;
            double* ck = B + k * nr;
            for (int i = k + 1; i < nr; ++i) ck[i] /= dk;
            for (int j = k + 1; j < w; ++j) {
                const double t = ck[j] * dk;
                if (t == 0.0) continue;
                double* cj = B + j * nr;
                for (int i = j; i < nr; ++i) cj[i] -= ck[i] * t;
            }
        }
        for (int i = 0; i < w; ++i) rel[f + i] = -1;
        for (int q = 0; q < r; ++q) rel[R[q]] = -1;
        if (r > 0) { int t = sn_of_[R[0]]; next[s] = head[t]; head[t] = s; ptr[s] = 0; }
    }
    return (int)regularized_.size();
}

void SparseLDL::solve(std::vector<double>& b) const {
    const int ns = (int)sn_start_.size() - 1;
    std::vector<double> x(n_);
    for (int k = 0; k < n_; ++k) x[k] = b[perm_[k]];
    for (int s = 0; s < ns; ++s) {
        const int f = sn_start_[s], w = sn_start_[s + 1] - f;
        const int* R = sn_rows_.data() + sn_rowptr_[s];
        const int r = sn_rowptr_[s + 1] - sn_rowptr_[s], nr = w + r;
        const double* B = sn_val_.data() + sn_valptr_[s];
        for (int k = 0; k < w; ++k) {
            const double xj = x[f + k];
            if (xj == 0.0) continue;
            const double* c = B + k * nr;
            for (int i = k + 1; i < w; ++i) x[f + i] -= c[i] * xj;
            for (int q = 0; q < r; ++q) x[R[q]] -= c[w + q] * xj;
        }
    }
    for (int j = 0; j < n_; ++j) x[j] /= D_[j];
    for (int s = ns - 1; s >= 0; --s) {
        const int f = sn_start_[s], w = sn_start_[s + 1] - f;
        const int* R = sn_rows_.data() + sn_rowptr_[s];
        const int r = sn_rowptr_[s + 1] - sn_rowptr_[s], nr = w + r;
        const double* B = sn_val_.data() + sn_valptr_[s];
        for (int k = w - 1; k >= 0; --k) {
            const double* c = B + k * nr;
            double sum = x[f + k];
            for (int i = k + 1; i < w; ++i) sum -= c[i] * x[f + i];
            for (int q = 0; q < r; ++q) sum -= c[w + q] * x[R[q]];
            x[f + k] = sum;
        }
    }
    for (int k = 0; k < n_; ++k) b[perm_[k]] = x[k];
}
