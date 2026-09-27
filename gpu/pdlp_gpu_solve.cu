// pdlp_gpu_solve.cu
// ==================
// CUDA backend for the restarted PDLP engine. The algorithm itself -- step
// sizes, primal weight, averaging, restarts, termination -- is NOT in this
// file: it is include/pdlp_algo.hpp's pdlp_run(), the same template the CPU
// backend (src/pdlp.cpp) instantiates. This file only supplies the
// arithmetic on the device, with hand-written kernels (no cuBLAS/cuSPARSE,
// per the LP pipeline's Step 3 sovereignty note):
//   * warp-per-row CSR SpMV (A x and A^T y, A^T stored as its own CSR so
//     both products are row-parallel and coalesced)
//   * fused PDHG primal and dual updates
//   * fused reductions for the adaptive step size (|dx|^2, |dy|^2 and
//     (y'-y)^T A (x'-x) in one pass)
// A, the iterates and A x / A^T y stay resident in VRAM; the host only reads
// three scalars per step (the step-size test) and the iterates every
// `check_every` accepted steps, for the same independent verify() the CPU
// engine uses.
//
// gpu/pdlp_spmv_kernel.cu is kept separately as the original documented
// kernel design; this is the file CMake compiles when it finds nvcc.

#include "pdlp_gpu.hpp"
#ifdef SOVEREIGN_WITH_CUDA

#include "pdlp_algo.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>

#define SOVEREIGN_CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") + std::to_string(__LINE__) + \
                                  " -- " + cudaGetErrorString(_e)); \
    } \
} while (0)

// ---------------------------------------------------------------- kernels

// One warp per row of y = A x, A in CSR.
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

// One thread per row -- for matrices whose rows are short (e.g. A^T of a
// transportation LP: 2 nonzeros per row), where a warp per row would leave
// 30 of 32 lanes idle.
__global__ void sov_spmv_csr_thread_per_row(
    int rows, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    double sum = 0.0;
    for (int i = row_ptr[r]; i < row_ptr[r + 1]; ++i) sum += values[i] * x[col_idx[i]];
    y[r] = sum;
}

// xn = clip(x - tau (c - A^T y), l, u)
__global__ void sov_primal_step(int n, const double* __restrict__ x, const double* __restrict__ c,
                                const double* __restrict__ aty, const double* __restrict__ l,
                                const double* __restrict__ u, double tau, double* __restrict__ xn)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double v = x[j] - tau * (c[j] - aty[j]);
    xn[j] = v < l[j] ? l[j] : (v > u[j] ? u[j] : v);
}

// yn = proj(y + sigma (b - (2 A xn - A x)))   (inequality rows: y >= 0)
__global__ void sov_dual_step(int m, const double* __restrict__ y, const double* __restrict__ b,
                              const double* __restrict__ axn, const double* __restrict__ ax,
                              const unsigned char* __restrict__ is_ineq, double sigma, double* __restrict__ yn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double v = y[i] + sigma * (b[i] - (2.0 * axn[i] - ax[i]));
    if (is_ineq[i] && v < 0.0) v = 0.0;
    yn[i] = v;
}

__device__ double sov_block_sum(double v) {
    __shared__ double warp_sums[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    if (lane == 0) warp_sums[w] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? warp_sums[lane] : 0.0;
    if (w == 0) for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;   // valid in thread 0
}

// Reductions write one partial sum per block (no atomics); the host adds the
// partials in block order, so results are bit-for-bit reproducible run to run.
// part[b] = block b's sum of (xn - x)^2
__global__ void sov_primal_stats(int n, const double* __restrict__ x, const double* __restrict__ xn, double* part)
{
    double s = 0.0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        double d = xn[j] - x[j]; s += d * d;
    }
    s = sov_block_sum(s);
    if (threadIdx.x == 0) part[blockIdx.x] = s;
}

// part1[b] = block b's sum of (yn - y)^2,  part2[b] = of (yn - y)(axn - ax)
__global__ void sov_dual_stats(int m, const double* __restrict__ y, const double* __restrict__ yn,
                               const double* __restrict__ ax, const double* __restrict__ axn,
                               double* part1, double* part2)
{
    double s1 = 0.0, s2 = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
        double d = yn[i] - y[i]; s1 += d * d; s2 += d * (axn[i] - ax[i]);
    }
    s1 = sov_block_sum(s1);
    __syncthreads();
    s2 = sov_block_sum(s2);
    if (threadIdx.x == 0) { part1[blockIdx.x] = s1; part2[blockIdx.x] = s2; }
}

// y += a x
__global__ void sov_axpy(int n, double a, const double* __restrict__ x, double* __restrict__ y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) y[j] += a * x[j];
}

// -------------------------------------------------------------- host side

namespace {

int blocks_for(int n, int threads = 256) { return (n + threads - 1) / threads; }
constexpr int kMaxBlocks = 256;
int reduce_blocks(int n) { return std::min(kMaxBlocks, std::max(1, blocks_for(n))); }

// Owning device vector; move-only so pdlp_run's swaps are pointer swaps.
struct DVec {
    double* p = nullptr;
    int n = 0;
    DVec() = default;
    explicit DVec(int len) : n(len) {
        SOVEREIGN_CUDA_CHECK(cudaMalloc(&p, std::max(1, n) * sizeof(double)));
        SOVEREIGN_CUDA_CHECK(cudaMemset(p, 0, std::max(1, n) * sizeof(double)));
    }
    DVec(DVec&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DVec& operator=(DVec&& o) noexcept { std::swap(p, o.p); std::swap(n, o.n); return *this; }
    DVec(const DVec&) = delete;
    DVec& operator=(const DVec&) = delete;
    ~DVec() { if (p) cudaFree(p); }
};

struct DeviceCSR {
    int rows = 0;
    int *row_ptr = nullptr, *col_idx = nullptr;
    double* values = nullptr;
    bool short_rows = false;     // average row length <= 8: thread-per-row kernel
    explicit DeviceCSR(const CSR& h) : rows(h.rows) {
        short_rows = h.rows > 0 && h.indices.size() <= 8 * (size_t)h.rows;
        SOVEREIGN_CUDA_CHECK(cudaMalloc(&row_ptr, h.indptr.size() * sizeof(int)));
        SOVEREIGN_CUDA_CHECK(cudaMalloc(&col_idx, std::max<size_t>(1, h.indices.size()) * sizeof(int)));
        SOVEREIGN_CUDA_CHECK(cudaMalloc(&values, std::max<size_t>(1, h.data.size()) * sizeof(double)));
        SOVEREIGN_CUDA_CHECK(cudaMemcpy(row_ptr, h.indptr.data(), h.indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (!h.indices.empty()) {
            SOVEREIGN_CUDA_CHECK(cudaMemcpy(col_idx, h.indices.data(), h.indices.size() * sizeof(int), cudaMemcpyHostToDevice));
            SOVEREIGN_CUDA_CHECK(cudaMemcpy(values, h.data.data(), h.data.size() * sizeof(double), cudaMemcpyHostToDevice));
        }
    }
    ~DeviceCSR() { cudaFree(row_ptr); cudaFree(col_idx); cudaFree(values); }
    DeviceCSR(const DeviceCSR&) = delete;
    DeviceCSR& operator=(const DeviceCSR&) = delete;
};

DVec upload(const std::vector<double>& h) {
    DVec d((int)h.size());
    if (!h.empty()) SOVEREIGN_CUDA_CHECK(cudaMemcpy(d.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
    return d;
}

struct GpuBackend {
    using Vec = DVec;
    int n, m;
    DeviceCSR A, AT;
    DVec c, l, u, b, stats;
    unsigned char* is_ineq = nullptr;

    explicit GpuBackend(const RangedLP& s)
        : n(s.n()), m(s.m()), A(to_csr(s.A)), AT(to_csc_as_transposed_csr(s.A)),
          c(upload(s.c)), l(upload(s.l)), u(upload(s.u)), b(upload(s.rL)), stats(3 * kMaxBlocks) {
        std::vector<unsigned char> ineq(m);
        for (int i = 0; i < m; ++i) ineq[i] = s.rL[i] != s.rU[i];
        SOVEREIGN_CUDA_CHECK(cudaMalloc(&is_ineq, std::max(1, m)));
        if (m) SOVEREIGN_CUDA_CHECK(cudaMemcpy(is_ineq, ineq.data(), m, cudaMemcpyHostToDevice));
    }
    ~GpuBackend() { cudaFree(is_ineq); }

    Vec vec_n() const { return DVec(n); }
    Vec vec_m() const { return DVec(m); }

    static void spmv(const DeviceCSR& M, const DVec& x, DVec& y) {
        if (M.rows == 0) return;
        if (M.short_rows) {
            sov_spmv_csr_thread_per_row<<<blocks_for(M.rows), 256>>>(M.rows, M.row_ptr, M.col_idx, M.values, x.p, y.p);
        } else {
            const int threads = 128, warps = threads / 32;
            sov_spmv_csr_warp_per_row<<<(M.rows + warps - 1) / warps, threads>>>(M.rows, M.row_ptr, M.col_idx, M.values, x.p, y.p);
        }
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());
    }
    void Ax(const Vec& x, Vec& out) const { spmv(A, x, out); }
    void ATy(const Vec& y, Vec& out) const { spmv(AT, y, out); }

    void primal_step(const Vec& x, const Vec& aty, double tau, Vec& xn) const {
        if (n) sov_primal_step<<<blocks_for(n), 256>>>(n, x.p, c.p, aty.p, l.p, u.p, tau, xn.p);
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());
    }
    void dual_step(const Vec& y, const Vec& axn, const Vec& ax, double sigma, Vec& yn) const {
        if (m) sov_dual_step<<<blocks_for(m), 256>>>(m, y.p, b.p, axn.p, ax.p, is_ineq, sigma, yn.p);
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());
    }
    void step_stats(const Vec& x, const Vec& xn, const Vec& y, const Vec& yn, const Vec& ax, const Vec& axn,
                    double& dx2, double& dy2, double& inter) const {
        const int bn = n ? reduce_blocks(n) : 0, bm = m ? reduce_blocks(m) : 0;
        if (bn) sov_primal_stats<<<bn, 256>>>(n, x.p, xn.p, stats.p);
        if (bm) sov_dual_stats<<<bm, 256>>>(m, y.p, yn.p, ax.p, axn.p, stats.p + kMaxBlocks, stats.p + 2 * kMaxBlocks);
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());
        double h[3 * kMaxBlocks];
        SOVEREIGN_CUDA_CHECK(cudaMemcpy(h, stats.p, 3 * kMaxBlocks * sizeof(double), cudaMemcpyDeviceToHost));
        dx2 = dy2 = inter = 0.0;
        for (int k = 0; k < bn; ++k) dx2 += h[k];
        for (int k = 0; k < bm; ++k) { dy2 += h[kMaxBlocks + k]; inter += h[2 * kMaxBlocks + k]; }
    }
    void axpy(double a, const Vec& x, Vec& y) const {
        if (x.n) sov_axpy<<<blocks_for(x.n), 256>>>(x.n, a, x.p, y.p);
        SOVEREIGN_CUDA_CHECK(cudaGetLastError());
    }
    void zero(Vec& v) const { SOVEREIGN_CUDA_CHECK(cudaMemset(v.p, 0, std::max(1, v.n) * sizeof(double))); }
    void swap(Vec& a, Vec& b2) const { std::swap(a.p, b2.p); std::swap(a.n, b2.n); }
    void to_host(const Vec& v, std::vector<double>& h) const {
        h.resize(v.n);
        if (v.n) SOVEREIGN_CUDA_CHECK(cudaMemcpy(h.data(), v.p, v.n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    void from_host(const std::vector<double>& h, Vec& v) const {
        if (!h.empty()) SOVEREIGN_CUDA_CHECK(cudaMemcpy(v.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
};

} // namespace

PdlpResult solve_pdhg_gpu(const RangedLP& unscaled_form, const RangedLP& scaled_form,
                           const std::vector<double>& Dr, const std::vector<double>& Dc,
                           double /*eta: adaptive*/, int max_iterations, int check_every, double tol,
                           std::atomic<bool>* stop_flag) {
    GpuBackend B(scaled_form);
    return pdlp_run(B, unscaled_form, scaled_form, Dr, Dc, max_iterations, check_every, tol, stop_flag);
}

#endif // SOVEREIGN_WITH_CUDA
