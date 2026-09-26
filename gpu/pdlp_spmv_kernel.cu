// pdlp_spmv_kernel.cu
// ====================
// The GPU half of the PDLP engine: a real CUDA kernel, written to compile
// with nvcc, but NOT built or run as part of today's POC -- this sandbox
// has no CUDA toolkit and no GPU attached (checked before writing this
// file). It is included honestly, as designed-not-yet-tested code, so the
// architecture in the master plan doc has a concrete artifact behind it,
// not just a diagram.
//
// It mirrors src/pdlp.cpp's solve_pdhg() loop exactly:
//   x_new = clip(x - tau*(c - A^T y), l, u)
//   y_new = clip_ineq(y + sigma*(b - A*(2*x_new - x)))
// Both lines are one sparse matrix-vector product each -- vector-CSR,
// one warp per row (spec Part 8.4), which is what the kernel below does.
// Porting the CPU loop to call this kernel instead of matvec()/matvec_T()
// is the ONLY change needed to move PDLP onto the GPU; the surrounding
// restart/verify logic in solve_pdhg() is untouched.

#include <cuda_runtime.h>

// One warp (32 threads) computes one row of y = A * x, A in CSR.
// This is the exact pattern named in the master doc's Technical Approach
// section ("Warp-level SpMV (CSR)") and in spec Part 8.4.
__global__ void spmv_csr_warp_per_row(
    int rows,
    const int*    __restrict__ row_ptr,   // size rows+1
    const int*    __restrict__ col_idx,   // size nnz
    const double* __restrict__ values,    // size nnz
    const double* __restrict__ x,         // size cols
    double*       __restrict__ y)         // size rows, output
{
    const int warp_id  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane      = threadIdx.x % 32;
    if (warp_id >= rows) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    double sum = 0.0;
    for (int i = row_start + lane; i < row_end; i += 32)
        sum += values[i] * x[col_idx[i]];

    // Warp-level reduction: __shfl_down_sync combines the 32 partial sums
    // in registers, no shared memory or block-wide __syncthreads() needed
    // for this step -- the technique named in the master doc's
    // "Synchronization" section.
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) y[warp_id] = sum;
}

// Elementwise PDHG update kernels -- kernel-fused (one launch does clip +
// subtract + scale in a single pass) per the master doc's "kernel fusion"
// point, instead of three separate elementwise kernels.
__global__ void pdhg_primal_update(
    int n, const double* __restrict__ x, const double* __restrict__ c,
    const double* __restrict__ ATy, const double* __restrict__ l,
    const double* __restrict__ u, double tau, double* __restrict__ x_new)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double v = x[j] - tau * (c[j] - ATy[j]);
    x_new[j] = v < l[j] ? l[j] : (v > u[j] ? u[j] : v);
}

__global__ void pdhg_dual_update(
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

// Host-side launch wrapper -- what src/pdlp.cpp's solve_pdhg() would call
// instead of matvec()/matvec_T() once ported. Left as a stub: needs the
// CUDA memory-residency plumbing (allocate once, keep A/x/y resident for
// the whole solve, per the master doc's "data stays on GPU" point) that
// this environment cannot exercise without a GPU. Not compiled here.
//
// void spmv_gpu(int rows, const int* row_ptr, const int* col_idx,
//               const double* values, const double* x, double* y) {
//     int threads_per_block = 128;
//     int warps_per_block = threads_per_block / 32;
//     int blocks = (rows + warps_per_block - 1) / warps_per_block;
//     spmv_csr_warp_per_row<<<blocks, threads_per_block>>>(rows, row_ptr, col_idx, values, x, y);
// }
