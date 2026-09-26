// pdlp_gpu_solve.cu
// ==================
// The engine that actually gets compiled into the GPU race, when CMake
// finds a CUDA compiler (see CMakeLists.txt's check_language(CUDA) block
// and the SOVEREIGN_WITH_CUDA guard in include/pdlp_gpu.hpp). This is a
// SEPARATE file from gpu/pdlp_spmv_kernel.cu on purpose: that file is kept
// exactly as it was -- the original, documented kernel design, referenced
// by name in the deck and README -- while this file is the complete,
// buildable host+device implementation that CMake wires into the binary.
// The three kernels below are the same design (warp-per-row CSR SpMV,
// kernel-fused elementwise PDHG updates) reproduced here so this
// translation unit is self-contained and there is exactly one definition
// of each __global__ symbol in the link.
//
// Honesty note (read this before quoting a number from this file): this
// code has been written to compile with nvcc and to mirror src/pdlp.cpp's
// solve_pdhg() exactly, iteration for iteration, restart rule for restart
// rule -- but it has NOT been compiled or run anywhere in this project's
// development, because no environment used so far (this sandbox, and the
// Windows/MSYS2 toolchain used for the CPU build) has an NVIDIA GPU and
// CUDA toolkit available. It is real, complete, structurally sound CUDA
// C++ -- not a stub -- but "compiles" and "has been compiled" are two
// different claims, and only the second one has actual evidence behind it
// once you build this on real GPU hardware.

#include "pdlp_gpu.hpp"
#ifdef SOVEREIGN_WITH_CUDA

#include "verify.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#define SOVEREIGN_CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") + std::to_string(__LINE__) + \
                                  " -- " + cudaGetErrorString(_e)); \
    } \
} while (0)

// ---------------------------------------------------------------- kernels

// One warp computes one row of y = A * x, A in CSR -- identical design to
// gpu/pdlp_spmv_kernel.cu's spmv_csr_warp_per_row (spec Part 8.4).
__global__ void sov_spmv_csr_warp_per_row(
    int rows, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_id >= rows) return;
    int row_start = row_ptr[warp_id], row_end = row_ptr[warp_id + 1];
    double sum = 0.0;
    for (int i = row_start + lane; i < row_end; i += 32) sum += values[i] * x[col_idx[i]];
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[warp_id] = sum;
}

// x_new = clip(x - tau*(c - ATy), l, u) -- fused into one kernel launch.
__global__ void sov_pdhg_primal_update(
    int n, const double* __restrict__ x, const double* __restrict__ c,
    const double* __restrict__ ATy, const double* __restrict__ l, const double* __restrict__ u,
    double tau, double* __restrict__ x_new)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double v = x[j] - tau * (c[j] - ATy[j]);
    x_new[j] = v < l[j] ? l[j] : (v > u[j] ? u[j] : v);
}

// y_new = clip_ineq(y + sigma*(b - Adx))
__global__ void sov_pdhg_dual_update(
    int m, const double* __restrict__ y, const double* __restrict__ b,
    const double* __restrict__ Adx, const unsigned char* __restrict__ is_ineq,
    double sigma, double* __restrict__ y_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double v = y[i] + sigma * (b[i] - Adx[i]);
    if (is_ineq[i] && v < 0.0) v = 0.0;
    y_new[i] = v;
}

// two_xnew_minus_x = 2*x_new - x -- the PDHG extrapolation step, one launch.
__global__ void sov_extrapolate(int n, const double* __restrict__ x_new, const double* __restrict__ x,
                                 double* __restrict__ out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    out[j] = 2.0 * x_new[j] - x[j];
}

// running_sum += value -- used for the ergodic (restart-averaged) iterate.
__global__ void sov_accumulate(int n, double* __restrict__ running_sum, const double* __restrict__ value)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    running_sum[j] += value[j];
}

// -------------------------------------------------------------- host side

namespace {

struct DeviceCSR {
    int rows = 0;
    int* row_ptr = nullptr;
    int* col_idx = nullptr;
    double* values = nullptr;

    void upload(const CSR& h) {
        rows = h.rows;
        SOVEREIGN_CUDA_CHECK(cudaMalloc(&row_ptr, h.indptr.size() * sizeof(int)));
        SOVEREIGN_CUDA_CHECK(cudaMalloc(&col_idx, h.indices.size() * sizeof(int)));
        SOVEREIGN_CUDA_CHECK(cudaMalloc(&values, h.data.size() * sizeof(double)));
        SOVEREIGN_CUDA_CHECK(cudaMemcpy(row_ptr, h.indptr.data(), h.indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        SOVEREIGN_CUDA_CHECK(cudaMemcpy(col_idx, h.indices.data(), h.indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        SOVEREIGN_CUDA_CHECK(cudaMemcpy(values, h.data.data(), h.data.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    void free() {
        if (row_ptr) cudaFree(row_ptr);
        if (col_idx) cudaFree(col_idx);
        if (values) cudaFree(values);
        row_ptr = col_idx = nullptr; values = nullptr;
    }
};

double* dalloc(size_t n) { double* p; SOVEREIGN_CUDA_CHECK(cudaMalloc(&p, n * sizeof(double))); return p; }

void spmv_gpu(const DeviceCSR& A, const double* x, double* y) {
    const int threads = 128, warps_per_block = threads / 32;
    const int blocks = (A.rows + warps_per_block - 1) / warps_per_block;
    sov_spmv_csr_warp_per_row<<<blocks, threads>>>(A.rows, A.row_ptr, A.col_idx, A.values, x, y);
    SOVEREIGN_CUDA_CHECK(cudaGetLastError());
}

int blocks_for(int n, int threads = 256) { return (n + threads - 1) / threads; }

// Same math as src/pdlp.cpp's file-local unscale(), reproduced here since
// that one has internal linkage and isn't visible outside pdlp.cpp.
void unscale_host(const std::vector<double>& Dr, const std::vector<double>& Dc,
                   const std::vector<double>& xs, const std::vector<double>& ys,
                   std::vector<double>& x, std::vector<double>& y) {
    x.resize(xs.size()); for (size_t j = 0; j < xs.size(); ++j) x[j] = Dc[j] * xs[j];
    y.resize(ys.size()); for (size_t i = 0; i < ys.size(); ++i) y[i] = Dr[i] * ys[i];
}

} // namespace

PdlpResult solve_pdhg_gpu(const RangedLP& unscaled_form, const RangedLP& scaled_form,
                           const std::vector<double>& Dr, const std::vector<double>& Dc,
                           double eta, int max_iterations, int check_every, double tol,
                           std::atomic<bool>* stop_flag) {
    const int m2 = scaled_form.m(), n2 = scaled_form.n();
    CSR A_h = to_csr(scaled_form.A);
    CSR AT_h = to_csc_as_transposed_csr(scaled_form.A);
    const auto& c_h = scaled_form.c; const auto& rL_h = scaled_form.rL; const auto& rU_h = scaled_form.rU;
    const auto& l_h = scaled_form.l; const auto& u_h = scaled_form.u;
    std::vector<unsigned char> is_ineq_h(m2);
    for (int i = 0; i < m2; ++i) is_ineq_h[i] = (rL_h[i] != rU_h[i]) ? 1 : 0;
    const auto& b_h = rL_h;

    // ---- device-resident data: uploaded once, kept for the whole solve ----
    DeviceCSR A, AT; A.upload(A_h); AT.upload(AT_h);
    double* c = dalloc(n2); SOVEREIGN_CUDA_CHECK(cudaMemcpy(c, c_h.data(), n2 * sizeof(double), cudaMemcpyHostToDevice));
    double* l = dalloc(n2); SOVEREIGN_CUDA_CHECK(cudaMemcpy(l, l_h.data(), n2 * sizeof(double), cudaMemcpyHostToDevice));
    double* u = dalloc(n2); SOVEREIGN_CUDA_CHECK(cudaMemcpy(u, u_h.data(), n2 * sizeof(double), cudaMemcpyHostToDevice));
    double* b = dalloc(m2); SOVEREIGN_CUDA_CHECK(cudaMemcpy(b, b_h.data(), m2 * sizeof(double), cudaMemcpyHostToDevice));
    unsigned char* is_ineq; SOVEREIGN_CUDA_CHECK(cudaMalloc(&is_ineq, m2));
    SOVEREIGN_CUDA_CHECK(cudaMemcpy(is_ineq, is_ineq_h.data(), m2, cudaMemcpyHostToDevice));

    double *x = dalloc(n2), *y = dalloc(m2), *x_new = dalloc(n2), *y_new = dalloc(m2);
    double *ATy = dalloc(n2), *extrap = dalloc(n2), *Adx = dalloc(m2);
    double *x_sum = dalloc(n2), *y_sum = dalloc(m2);
    SOVEREIGN_CUDA_CHECK(cudaMemset(x_sum, 0, n2 * sizeof(double)));
    SOVEREIGN_CUDA_CHECK(cudaMemset(y_sum, 0, m2 * sizeof(double)));

    // x0[j] = clip(0, l[j], u[j]) -- identical init to the CPU engine.
    std::vector<double> x0_h(n2);
    for (int j = 0; j < n2; ++j) x0_h[j] = std::min(std::max(0.0, l_h[j]), u_h[j]);
    SOVEREIGN_CUDA_CHECK(cudaMemcpy(x, x0_h.data(), n2 * sizeof(double), cudaMemcpyHostToDevice));
    SOVEREIGN_CUDA_CHECK(cudaMemset(y, 0, m2 * sizeof(double)));

    double tau = eta, sigma = eta;
    int n_since_restart = 0, restarts = 0;
    double last_restart_score = std::numeric_limits<double>::infinity();

    std::vector<double> x_host(n2), y_host(m2), x_sum_host(n2), y_sum_host(m2);

    auto finish = [&](bool converged, const std::vector<double>& xu, const std::vector<double>& yu,
                       const KKTReport& rep, int iters) {
        A.free(); AT.free();
        cudaFree(c); cudaFree(l); cudaFree(u); cudaFree(b); cudaFree(is_ineq);
        cudaFree(x); cudaFree(y); cudaFree(x_new); cudaFree(y_new);
        cudaFree(ATy); cudaFree(extrap); cudaFree(Adx); cudaFree(x_sum); cudaFree(y_sum);
        PdlpResult res; res.x = xu; res.y = yu; res.iterations = iters; res.converged = converged;
        res.eps_P = rep.eps_P; res.eps_D = rep.eps_D; res.eps_G = rep.eps_G; res.restarts = restarts;
        return res;
    };

    for (int k = 1; k <= max_iterations; ++k) {
        if (stop_flag && stop_flag->load(std::memory_order_relaxed)) {
            SOVEREIGN_CUDA_CHECK(cudaMemcpy(x_host.data(), x, n2 * sizeof(double), cudaMemcpyDeviceToHost));
            SOVEREIGN_CUDA_CHECK(cudaMemcpy(y_host.data(), y, m2 * sizeof(double), cudaMemcpyDeviceToHost));
            std::vector<double> xu, yu; unscale_host(Dr, Dc, x_host, y_host, xu, yu);
            KKTReport rep = verify(unscaled_form, xu, yu);
            return finish(false, xu, yu, rep, k);
        }

        spmv_gpu(AT, y, ATy);                                                   // ATy = AT * y
        sov_pdhg_primal_update<<<blocks_for(n2), 256>>>(n2, x, c, ATy, l, u, tau, x_new);
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());
        sov_extrapolate<<<blocks_for(n2), 256>>>(n2, x_new, x, extrap);
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());
        spmv_gpu(A, extrap, Adx);                                               // Adx = A * (2*x_new - x)
        sov_pdhg_dual_update<<<blocks_for(m2), 256>>>(m2, y, b, Adx, is_ineq, sigma, y_new);
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());

        std::swap(x, x_new); std::swap(y, y_new);                               // x=x_new, y=y_new (pointer swap, no copy)
        sov_accumulate<<<blocks_for(n2), 256>>>(n2, x_sum, x);
        sov_accumulate<<<blocks_for(m2), 256>>>(m2, y_sum, y);
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());
        n_since_restart++;

        if (k % check_every == 0) {
            // The one sync point per check_every iterations -- same cadence
            // the CPU engine uses, so both engines call verify() equally
            // often, on the same schedule, for a fair race.
            SOVEREIGN_CUDA_CHECK(cudaMemcpy(x_host.data(), x, n2 * sizeof(double), cudaMemcpyDeviceToHost));
            SOVEREIGN_CUDA_CHECK(cudaMemcpy(y_host.data(), y, m2 * sizeof(double), cudaMemcpyDeviceToHost));
            SOVEREIGN_CUDA_CHECK(cudaMemcpy(x_sum_host.data(), x_sum, n2 * sizeof(double), cudaMemcpyDeviceToHost));
            SOVEREIGN_CUDA_CHECK(cudaMemcpy(y_sum_host.data(), y_sum, m2 * sizeof(double), cudaMemcpyDeviceToHost));

            std::vector<double> x_avg(n2), y_avg(m2);
            for (int j = 0; j < n2; ++j) x_avg[j] = x_sum_host[j] / n_since_restart;
            for (int i = 0; i < m2; ++i) y_avg[i] = y_sum_host[i] / n_since_restart;

            std::vector<double> xu_cur, yu_cur, xu_avg, yu_avg;
            unscale_host(Dr, Dc, x_host, y_host, xu_cur, yu_cur);
            unscale_host(Dr, Dc, x_avg, y_avg, xu_avg, yu_avg);
            KKTReport rep_cur = verify(unscaled_form, xu_cur, yu_cur);
            KKTReport rep_avg = verify(unscaled_form, xu_avg, yu_avg);
            double score_cur = rep_cur.eps_P + rep_cur.eps_D + rep_cur.eps_G;
            double score_avg = rep_avg.eps_P + rep_avg.eps_D + rep_avg.eps_G;

            bool use_avg = score_avg <= score_cur;
            const auto& cand_xu = use_avg ? xu_avg : xu_cur;
            const auto& cand_yu = use_avg ? yu_avg : yu_cur;
            const auto& cand_x = use_avg ? x_avg : x_host;
            const auto& cand_y = use_avg ? y_avg : y_host;
            double cand_score = use_avg ? score_avg : score_cur;
            const KKTReport& cand_rep = use_avg ? rep_avg : rep_cur;

            if (std::max({cand_rep.eps_P, cand_rep.eps_D, cand_rep.eps_G}) <= tol) {
                return finish(true, cand_xu, cand_yu, cand_rep, k);
            }

            bool sufficient_decay = cand_score <= 0.2 * last_restart_score;
            bool artificial = n_since_restart >= std::max(64, (int)(0.36 * k));
            if (sufficient_decay || artificial) {
                SOVEREIGN_CUDA_CHECK(cudaMemcpy(x, cand_x.data(), n2 * sizeof(double), cudaMemcpyHostToDevice));
                SOVEREIGN_CUDA_CHECK(cudaMemcpy(y, cand_y.data(), m2 * sizeof(double), cudaMemcpyHostToDevice));
                SOVEREIGN_CUDA_CHECK(cudaMemset(x_sum, 0, n2 * sizeof(double)));
                SOVEREIGN_CUDA_CHECK(cudaMemset(y_sum, 0, m2 * sizeof(double)));
                n_since_restart = 0;
                last_restart_score = cand_score;
                restarts++;
            }
        }
    }

    // max_iterations reached without converging -- same tail tie-break as
    // the CPU engine: report whichever of {current, running-average} has
    // the smaller total KKT residual.
    SOVEREIGN_CUDA_CHECK(cudaMemcpy(x_host.data(), x, n2 * sizeof(double), cudaMemcpyDeviceToHost));
    SOVEREIGN_CUDA_CHECK(cudaMemcpy(y_host.data(), y, m2 * sizeof(double), cudaMemcpyDeviceToHost));
    SOVEREIGN_CUDA_CHECK(cudaMemcpy(x_sum_host.data(), x_sum, n2 * sizeof(double), cudaMemcpyDeviceToHost));
    SOVEREIGN_CUDA_CHECK(cudaMemcpy(y_sum_host.data(), y_sum, m2 * sizeof(double), cudaMemcpyDeviceToHost));
    int denom = std::max(1, n_since_restart);
    std::vector<double> x_avg(n2), y_avg(m2);
    for (int j = 0; j < n2; ++j) x_avg[j] = x_sum_host[j] / denom;
    for (int i = 0; i < m2; ++i) y_avg[i] = y_sum_host[i] / denom;
    std::vector<double> xu_cur, yu_cur, xu_avg, yu_avg;
    unscale_host(Dr, Dc, x_host, y_host, xu_cur, yu_cur);
    unscale_host(Dr, Dc, x_avg, y_avg, xu_avg, yu_avg);
    KKTReport rep_cur = verify(unscaled_form, xu_cur, yu_cur);
    KKTReport rep_avg = verify(unscaled_form, xu_avg, yu_avg);
    if ((rep_avg.eps_P + rep_avg.eps_D + rep_avg.eps_G) <= (rep_cur.eps_P + rep_cur.eps_D + rep_cur.eps_G))
        return finish(false, xu_avg, yu_avg, rep_avg, max_iterations);
    return finish(false, xu_cur, yu_cur, rep_cur, max_iterations);
}

#endif // SOVEREIGN_WITH_CUDA
