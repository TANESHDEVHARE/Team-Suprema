#include "basis_lu.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace {

// Doubly-linked count buckets: O(1) insert/remove, lets the Markowitz search
// visit rows/columns in order of increasing nonzero count.
struct Buckets {
    std::vector<int> head, next, prev, count;
    void init(int n_items, int max_count) {
        head.assign(max_count + 2, -1);
        next.assign(n_items, -1); prev.assign(n_items, -1); count.assign(n_items, -1);
    }
    void insert(int x, int k) {
        count[x] = k; prev[x] = -1; next[x] = head[k];
        if (head[k] >= 0) prev[head[k]] = x;
        head[k] = x;
    }
    void remove(int x) {
        int k = count[x];
        if (k < 0) return;
        if (prev[x] >= 0) next[prev[x]] = next[x]; else head[k] = next[x];
        if (next[x] >= 0) prev[next[x]] = prev[x];
        count[x] = -1;
    }
    void move(int x, int k) { remove(x); insert(x, k); }
};

double find_val(const std::vector<std::pair<int,double>>& row, int j) {
    for (const auto& e : row) if (e.first == j) return e.second;
    return 0.0;
}

void erase_key(std::vector<std::pair<int,double>>& v, int key) {
    for (size_t k = 0; k < v.size(); ++k) if (v[k].first == key) { v[k] = v.back(); v.pop_back(); return; }
}

} // namespace

std::vector<std::pair<int,int>> BasisLU::factorize(int m, const ColumnFn& get_col) {
    m_ = m;
    l_etas_.clear(); r_etas_.clear();
    ucol_.assign(m, {}); urow_.assign(m, {});
    diag_.assign(m, 0.0);
    row_of_.assign(m, -1); slot_of_.assign(m, -1);
    t_of_row_.assign(m, -1); t_of_slot_.assign(m, -1);
    work_.assign(m, 0.0);

    // ---- active submatrix, stored twice with values (Suhl & Suhl): rows for
    // the elimination, columns so the pivot search never has to look a value
    // up by scanning a row. Elimination keeps both copies identical.
    std::vector<std::vector<std::pair<int,double>>> arow(m), acol(m);
    std::vector<int> rows; std::vector<double> vals;
    for (int s = 0; s < m; ++s) {
        get_col(s, rows, vals);
        for (size_t k = 0; k < rows.size(); ++k) {
            if (vals[k] == 0.0) continue;
            arow[rows[k]].emplace_back(s, vals[k]);
            acol[s].emplace_back(rows[k], vals[k]);
        }
    }

    // Explicit active counts. Row/column lists are compacted by the elimination
    // pass that already walks them, so no pivot needs a linear-scan deletion
    // (with dense columns such as FIT2P's that deletion dominated factor time).
    // The only stale entries are those of dropped singular columns, skipped
    // via col_done until their row is next compacted.
    std::vector<int> rcnt(m), ccnt(m);
    for (int i = 0; i < m; ++i) rcnt[i] = (int)arow[i].size();
    for (int j = 0; j < m; ++j) ccnt[j] = (int)acol[j].size();
    Buckets cb, rb;
    cb.init(m, m); rb.init(m, m);
    // Removed rows stay in the column lists until the column is next updated
    // by an elimination (skipped via row_done), so a pivot with no multipliers
    // -- e.g. every slack -- costs O(|pivot row|), not O(sum of its columns).
    std::vector<char> col_done(m, 0), row_done(m, 0);
    for (int j = 0; j < m; ++j) cb.insert(j, ccnt[j]);
    for (int i = 0; i < m; ++i) rb.insert(i, rcnt[i]);

    std::vector<std::vector<std::pair<int,double>>> urow_slot(m);   // U row entries keyed by slot, per step
    std::vector<int> pos_in_row(m, -1), pos_in_col(m, -1);
    const double u = pivot_threshold;

    // Column maxima are cached: a pivot only changes the columns of the pivot
    // row, so only those are invalidated.
    std::vector<double> cmax_cache(m, 0.0);
    std::vector<char> cmax_valid(m, 0);
    auto col_max = [&](int j) {
        if (cmax_valid[j]) return cmax_cache[j];
        double mx = 0.0;
        for (const auto& e : acol[j]) if (!row_done[e.first]) mx = std::max(mx, std::fabs(e.second));
        cmax_cache[j] = mx; cmax_valid[j] = 1;
        return mx;
    };
    auto drop_singular_col = [&](int j) {
        for (const auto& e : acol[j]) if (!row_done[e.first]) { rcnt[e.first]--; rb.move(e.first, rcnt[e.first]); }
        acol[j].clear();
        cb.remove(j);
        col_done[j] = 1;
    };

    int step = 0;
    std::vector<std::pair<int,double>> prow;
    while (step < m) {
        // Columns with no active entries can never be pivoted.
        while (cb.head[0] >= 0) drop_singular_col(cb.head[0]);

        // ---- Markowitz search ----
        int best_r = -1, best_c = -1;
        double best_cost = std::numeric_limits<double>::infinity(), best_abs = 0.0;
        int searched = 0;
        std::vector<int> tiny_cols;
        for (int k = 1; k <= m; ++k) {
            for (int j = cb.head[k]; j >= 0; j = cb.next[j]) {
                double cmax = col_max(j);
                if (cmax < singular_tol) { tiny_cols.push_back(j); continue; }
                for (const auto& e : acol[j]) {
                    int i = e.first;
                    if (row_done[i]) continue;
                    double v = std::fabs(e.second);
                    if (v < u * cmax) continue;
                    double cost = double(rcnt[i] - 1) * double(k - 1);
                    if (cost < best_cost || (cost == best_cost && v > best_abs)) {
                        best_cost = cost; best_abs = v; best_r = i; best_c = j;
                    }
                }
                if (best_r >= 0 && ++searched >= 4) break;
            }
            if (best_r >= 0 && (searched >= 4 || best_cost <= double(k - 1) * (k - 1))) break;
            for (int i = rb.head[k]; i >= 0; i = rb.next[i]) {
                for (const auto& e : arow[i]) {
                    int j = e.first;
                    if (col_done[j]) continue;
                    double cmax = col_max(j);
                    if (cmax < singular_tol) continue;
                    double v = std::fabs(e.second);
                    if (v < u * cmax) continue;
                    double cost = double(k - 1) * double(ccnt[j] - 1);
                    if (cost < best_cost || (cost == best_cost && v > best_abs)) {
                        best_cost = cost; best_abs = v; best_r = i; best_c = j;
                    }
                }
                if (best_r >= 0 && ++searched >= 4) break;
            }
            // Every unsearched candidate has row and column count >= k+1.
            if (best_r >= 0 && (searched >= 4 || best_cost <= double(k) * k)) break;
        }
        if (best_r < 0) {
            if (tiny_cols.empty()) break;           // nothing pivotable remains
            for (int j : tiny_cols) if (!col_done[j]) drop_singular_col(j);
            continue;
        }

        // ---- eliminate with pivot (r, c) ----
        const int r = best_r, c = best_c;
        const double piv = find_val(arow[r], c);
        row_of_[step] = r; slot_of_[step] = c; diag_[step] = piv;
        t_of_row_[r] = step; t_of_slot_[c] = step;

        // Pivot row leaves the active matrix; its off-pivot part becomes a U row.
        prow.clear();
        for (const auto& e : arow[r])
            if (e.first != c && !col_done[e.first]) prow.push_back(e);
        urow_slot[step] = prow;
        std::vector<std::pair<int,double>>().swap(arow[r]);
        rb.remove(r); row_done[r] = 1;

        // Multipliers l_i = a_ic / pivot, straight from the column copy.
        Eta eta; eta.pivot_row = r;
        for (const auto& e : acol[c])
            if (e.first != r && !row_done[e.first]) eta.entries.emplace_back(e.first, e.second / piv);
        std::vector<std::pair<int,double>>().swap(acol[c]);
        cb.remove(c); col_done[c] = 1;

        // a_ij -= l_i * a_rj on the row copy; the pass drops column c and any
        // stale entries as it goes ...
        for (const auto& me : eta.entries) {
            const int i = me.first; const double l = me.second;
            auto& ri = arow[i];
            for (size_t k = 0; k < ri.size(); ) {          // index the row, dropping done columns in place
                if (col_done[ri[k].first]) { ri[k] = ri.back(); ri.pop_back(); continue; }
                pos_in_row[ri[k].first] = (int)k; ++k;
            }
            for (const auto& e : prow) {
                int j = e.first;
                if (pos_in_row[j] >= 0) ri[pos_in_row[j]].second -= l * e.second;
                else ri.emplace_back(j, -l * e.second);
            }
            for (const auto& e : ri) pos_in_row[e.first] = -1;
            rcnt[i] = (int)ri.size();
            rb.move(i, rcnt[i]);
        }
        // ... and, with the identical arithmetic, on the column copy, which
        // drops row r as it goes.
        for (const auto& pe : prow) {
            const int j = pe.first; const double a_rj = pe.second;
            auto& cj = acol[j];
            if (eta.entries.empty()) {                     // nothing to eliminate: row r just leaves column j
                ccnt[j]--;
                cb.move(j, ccnt[j]);
                cmax_valid[j] = 0;
                continue;
            }
            for (size_t k = 0; k < cj.size(); ) {          // index the column, dropping removed rows in place
                if (row_done[cj[k].first]) { cj[k] = cj.back(); cj.pop_back(); continue; }
                pos_in_col[cj[k].first] = (int)k; ++k;
            }
            for (const auto& me : eta.entries) {
                int i = me.first;
                if (pos_in_col[i] >= 0) cj[pos_in_col[i]].second -= me.second * a_rj;
                else cj.emplace_back(i, -me.second * a_rj);
            }
            for (const auto& e : cj) pos_in_col[e.first] = -1;
            ccnt[j] = (int)cj.size();
            cb.move(j, ccnt[j]);
            cmax_valid[j] = 0;
        }
        if (!eta.entries.empty()) l_etas_.push_back(std::move(eta));
        ++step;
    }

    // ---- basis repair: pair each unpivoted slot with an unpivoted row ----
    std::vector<std::pair<int,int>> replaced;
    {
        std::vector<int> free_rows, free_slots;
        for (int i = 0; i < m; ++i) if (!row_done[i]) free_rows.push_back(i);
        for (int j = 0; j < m; ++j) if (t_of_slot_[j] < 0) free_slots.push_back(j);
        for (size_t k = 0; k < free_slots.size(); ++k) {
            int j = free_slots[k], i = free_rows[k];
            row_of_[step] = i; slot_of_[step] = j; diag_[step] = 1.0;
            t_of_row_[i] = step; t_of_slot_[j] = step;
            replaced.emplace_back(j, i);
            ++step;
        }
    }
    std::vector<char> slot_replaced(m, 0);
    for (const auto& p : replaced) slot_replaced[p.first] = 1;

    // ---- convert U rows from slot keys to U-index keys ----
    for (int t = 0; t < m; ++t) {
        for (const auto& e : urow_slot[t]) {
            if (slot_replaced[e.first]) continue;     // replaced column is now a unit vector
            int t2 = t_of_slot_[e.first];
            if (e.second == 0.0) continue;
            urow_[t].emplace_back(t2, e.second);
            ucol_[t2].emplace_back(t, e.second);
        }
    }
    // Row-wise copy of L for BTRAN's push form: lrow_[i] = (pivot row, l) of
    // every eta in which row i was eliminated.
    lrow_.assign(m, {});
    for (const auto& eta : l_etas_)
        for (const auto& en : eta.entries) lrow_[en.first].emplace_back(eta.pivot_row, en.second);
    lrow_order_ = row_of_;

    ord_.resize(m); pos_.resize(m);
    for (int t = 0; t < m; ++t) { ord_[t] = t; pos_[t] = t; }
    return replaced;
}

void BasisLU::ftran(std::vector<double>& y, std::vector<double>& x, std::vector<double>* spike) const {
    for (const auto& eta : l_etas_) {
        double yr = y[eta.pivot_row];
        if (yr == 0.0) continue;
        for (const auto& e : eta.entries) y[e.first] -= e.second * yr;
    }
    for (const auto& eta : r_etas_) {
        double s = 0.0;
        for (const auto& e : eta.entries) s += e.second * y[e.first];
        y[eta.pivot_row] -= s;
    }
    if (spike) *spike = y;
    x.assign(m_, 0.0);
    for (int k = m_ - 1; k >= 0; --k) {
        int t = ord_[k];
        double yt = y[row_of_[t]];
        if (yt == 0.0) continue;
        double xt = yt / diag_[t];
        x[slot_of_[t]] = xt;
        for (const auto& e : ucol_[t]) y[row_of_[e.first]] -= e.second * xt;
    }
}

// All three stages run in "push" form: once an entry is final it is skipped
// if zero, otherwise pushed along its row. Cost is O(m + nonzeros touched),
// not O(nnz(L) + nnz(U)), which matters because BTRAN runs every iteration
// with a unit (hence usually sparse) right-hand side.
void BasisLU::btran(std::vector<double>& e, std::vector<double>& y) const {
    // U^T w = e, forward in pivot order; w_t lives on row_of[t]. e is used as
    // the (slot-indexed) work vector.
    y.assign(m_, 0.0);
    for (int k = 0; k < m_; ++k) {
        int t = ord_[k];
        double s = e[slot_of_[t]];
        if (s == 0.0) continue;
        double wt = s / diag_[t];
        y[row_of_[t]] = wt;
        for (const auto& en : urow_[t]) e[slot_of_[en.first]] -= en.second * wt;
    }
    for (auto it = r_etas_.rbegin(); it != r_etas_.rend(); ++it) {
        double yp = y[it->pivot_row];
        if (yp == 0.0) continue;
        for (const auto& en : it->entries) y[en.first] -= en.second * yp;
    }
    // L^T: rows in reverse elimination order; a row's value is final once
    // every later pivot has pushed into it.
    for (int t = (int)lrow_order_.size() - 1; t >= 0; --t) {
        int i = lrow_order_[t];
        double yi = y[i];
        if (yi == 0.0) continue;
        for (const auto& en : lrow_[i]) y[en.first] -= en.second * yi;
    }
}

bool BasisLU::update(int slot, const std::vector<double>& spike, double alpha_pivot) {
    const int t = t_of_slot_[slot];
    const int p = pos_[t];
    const double old_diag = diag_[t];

    // 1. Drop the old column t of U.
    for (const auto& e : ucol_[t]) erase_key(urow_[e.first], t);
    ucol_[t].clear();

    // 2. Row t's off-diagonal part moves into the work vector (U-index keyed).
    std::vector<double>& w = work_;
    std::vector<int> touched;
    for (const auto& e : urow_[t]) {
        erase_key(ucol_[e.first], t);
        w[e.first] = e.second; touched.push_back(e.first);
    }
    urow_[t].clear();

    // 3. Spike becomes the new column t. Every other row index now precedes t.
    double d0 = 0.0;
    for (int i = 0; i < m_; ++i) {
        double v = spike[i];
        if (v == 0.0) continue;
        int s = t_of_row_[i];
        if (s == t) { d0 = v; continue; }
        if (std::fabs(v) < 1e-14) continue;
        ucol_[t].emplace_back(s, v);
        urow_[s].emplace_back(t, v);
    }
    w[t] = d0; touched.push_back(t);

    // 4. Eliminate row t against the rows that now precede it.
    Eta eta; eta.pivot_row = row_of_[t];
    for (int k = p + 1; k < m_; ++k) {
        int s = ord_[k];
        double ws = w[s];
        if (ws == 0.0) continue;
        w[s] = 0.0;
        double mult = ws / diag_[s];
        eta.entries.emplace_back(row_of_[s], mult);
        for (const auto& e : urow_[s]) {
            if (w[e.first] == 0.0) touched.push_back(e.first);
            w[e.first] -= mult * e.second;
        }
    }
    double new_diag = w[t];
    for (int s : touched) w[s] = 0.0;

    // 5. Move t to the end of the pivot order.
    ord_.erase(ord_.begin() + p);
    ord_.push_back(t);
    for (int k = p; k < m_; ++k) pos_[ord_[k]] = k;

    diag_[t] = new_diag;
    if (!eta.entries.empty()) r_etas_.push_back(std::move(eta));
    else r_etas_.push_back(Eta{row_of_[t], {}});   // keep num_updates() honest
    if (!(std::fabs(new_diag) > singular_tol)) return false;
    // |det B_new| = |alpha_r| |det B_old| and every other diagonal of U is
    // unchanged, so |new_diag| must equal |alpha_r * old_diag|. A mismatch
    // means the update lost accuracy: refactorize rather than keep it.
    if (alpha_pivot != 0.0) {
        double expect = std::fabs(alpha_pivot * old_diag);
        if (std::fabs(std::fabs(new_diag) - expect) > 1e-8 * (expect + 1e-12) * 1e2) return false;
    }
    return true;
}

long long BasisLU::fill() const {
    long long f = 0;
    for (const auto& e : l_etas_) f += (long long)e.entries.size();
    for (const auto& c : ucol_) f += (long long)c.size();
    return f + m_;
}
