// cutlass_gemm.cu — Cutlass GEMM replacing cuBLAS/cuBLASLt
//
// Three layout combinations (TN, NT, NN) × two TF32 tile sizes + SIMT fallback.
// Batched variants for attention (TN, NN).
// Operator caching: each unique (M,N,K,lda,ldb,ldc) gets initialize() once.
// Subsequent calls use update() (pointer swap) + operator().

#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

using Element     = float;
using Accumulator = float;
using Compute     = float;
using LayoutRM    = cutlass::layout::RowMajor;
using LayoutCM    = cutlass::layout::ColumnMajor;

using EpilogueTF32 = cutlass::epilogue::thread::LinearCombination<
    Element, 4, Accumulator, Compute>;

using EpilogueSIMT = cutlass::epilogue::thread::LinearCombination<
    Element, 1, Accumulator, Compute>;

// ---------------------------------------------------------------------------
// Swizzle + common arch
// ---------------------------------------------------------------------------

using Swizzle        = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using BatchedSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
using Arch    = cutlass::arch::Sm80;

// ---------------------------------------------------------------------------
// GEMM kernel type definitions
//   Naming: Gemm{LayoutA}{LayoutB}_{OpClass}_{Tile}
//   TN = A RowMajor (cuBLAS OP_T on col-major), B ColumnMajor (cuBLAS OP_N)
//   NT = A ColumnMajor, B RowMajor
//   NN = A ColumnMajor, B ColumnMajor
// ---------------------------------------------------------------------------

// ---- TN (OP_T, OP_N): most calls ----

using GemmTN_TF32_Large = cutlass::gemm::device::Gemm<
    Element, LayoutRM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTF32, Swizzle, 3>;

using GemmTN_TF32_Small = cutlass::gemm::device::Gemm<
    Element, LayoutRM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTF32, Swizzle, 3>;

// TF32 TensorOp with align1 epilogue (handles M not divisible by 4, K must be div4)
using GemmTN_TF32_Align1 = cutlass::gemm::device::Gemm<
    Element, LayoutRM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueSIMT, Swizzle, 3>;

// SIMT fallback (no alignment requirements, for K not div4)
using GemmTN_SIMT = cutlass::gemm::device::Gemm<
    Element, LayoutRM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassSimt, Arch,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueSIMT, Swizzle, 4>;

// ---- NT (OP_N, OP_T): alignment matmul ----

using GemmNT_TF32_Large = cutlass::gemm::device::Gemm<
    Element, LayoutCM, Element, LayoutRM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTF32, Swizzle, 3>;

using GemmNT_TF32_Small = cutlass::gemm::device::Gemm<
    Element, LayoutCM, Element, LayoutRM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTF32, Swizzle, 3>;

using GemmNT_TF32_Align1 = cutlass::gemm::device::Gemm<
    Element, LayoutCM, Element, LayoutRM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueSIMT, Swizzle, 3>;

using GemmNT_SIMT = cutlass::gemm::device::Gemm<
    Element, LayoutCM, Element, LayoutRM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassSimt, Arch,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueSIMT, Swizzle, 4>;

// ---- NN (OP_N, OP_N): conv_transpose1d GEMM, G2P gate_up/down ----

using GemmNN_TF32_Large = cutlass::gemm::device::Gemm<
    Element, LayoutCM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTF32, Swizzle, 3>;

using GemmNN_TF32_Small = cutlass::gemm::device::Gemm<
    Element, LayoutCM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTF32, Swizzle, 3>;

using GemmNN_TF32_Align1 = cutlass::gemm::device::Gemm<
    Element, LayoutCM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueSIMT, Swizzle, 3>;

using GemmNN_SIMT = cutlass::gemm::device::Gemm<
    Element, LayoutCM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassSimt, Arch,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueSIMT, Swizzle, 4>;

// ---- Batched TN (attention QK^T) ----

using GemmBatchedTN_TF32 = cutlass::gemm::device::GemmBatched<
    Element, LayoutRM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTF32, BatchedSwizzle, 3>;

using GemmBatchedTN_SIMT = cutlass::gemm::device::GemmBatched<
    Element, LayoutRM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassSimt, Arch,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueSIMT, BatchedSwizzle, 4>;

// ---- Batched NN (attention scores*V) ----

using GemmBatchedNN_TF32 = cutlass::gemm::device::GemmBatched<
    Element, LayoutCM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTF32, BatchedSwizzle, 3>;

using GemmBatchedNN_SIMT = cutlass::gemm::device::GemmBatched<
    Element, LayoutCM, Element, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassSimt, Arch,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueSIMT, BatchedSwizzle, 4>;

// ---------------------------------------------------------------------------
// Operator caches
// ---------------------------------------------------------------------------

struct GemmKey {
    int M, N, K, lda, ldb, ldc;
    bool operator==(const GemmKey& o) const {
        return M == o.M && N == o.N && K == o.K &&
               lda == o.lda && ldb == o.ldb && ldc == o.ldc;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const {
        size_t h = 0;
        auto mix = [&](int v) { h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2); };
        mix(k.M); mix(k.N); mix(k.K); mix(k.lda); mix(k.ldb); mix(k.ldc);
        return h;
    }
};

// TN caches
static std::unordered_map<GemmKey, GemmTN_TF32_Large, GemmKeyHash>  s_tn_large;
static std::unordered_map<GemmKey, GemmTN_TF32_Small, GemmKeyHash>  s_tn_small;
static std::unordered_map<GemmKey, GemmTN_TF32_Align1, GemmKeyHash> s_tn_align1;
static std::unordered_map<GemmKey, GemmTN_SIMT, GemmKeyHash>        s_tn_simt;

// NT caches
static std::unordered_map<GemmKey, GemmNT_TF32_Large, GemmKeyHash>  s_nt_large;
static std::unordered_map<GemmKey, GemmNT_TF32_Small, GemmKeyHash>  s_nt_small;
static std::unordered_map<GemmKey, GemmNT_TF32_Align1, GemmKeyHash> s_nt_align1;
static std::unordered_map<GemmKey, GemmNT_SIMT, GemmKeyHash>        s_nt_simt;

// NN caches
static std::unordered_map<GemmKey, GemmNN_TF32_Large, GemmKeyHash>  s_nn_large;
static std::unordered_map<GemmKey, GemmNN_TF32_Small, GemmKeyHash>  s_nn_small;
static std::unordered_map<GemmKey, GemmNN_TF32_Align1, GemmKeyHash> s_nn_align1;
static std::unordered_map<GemmKey, GemmNN_SIMT, GemmKeyHash>        s_nn_simt;

// Batched caches (keyed by M,N,K + batch as lda, stride info as ldb,ldc)
struct BatchedKey {
    int M, N, K, batch;
    long long strideA, strideB, strideC;
    bool operator==(const BatchedKey& o) const {
        return M == o.M && N == o.N && K == o.K && batch == o.batch &&
               strideA == o.strideA && strideB == o.strideB && strideC == o.strideC;
    }
};

struct BatchedKeyHash {
    size_t operator()(const BatchedKey& k) const {
        size_t h = 0;
        auto mix = [&](long long v) { h ^= std::hash<long long>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2); };
        mix(k.M); mix(k.N); mix(k.K); mix(k.batch);
        mix(k.strideA); mix(k.strideB); mix(k.strideC);
        return h;
    }
};

static std::unordered_map<BatchedKey, GemmBatchedTN_TF32, BatchedKeyHash> s_batched_tn_tf32;
static std::unordered_map<BatchedKey, GemmBatchedTN_SIMT, BatchedKeyHash> s_batched_tn_simt;
static std::unordered_map<BatchedKey, GemmBatchedNN_TF32, BatchedKeyHash> s_batched_nn_tf32;
static std::unordered_map<BatchedKey, GemmBatchedNN_SIMT, BatchedKeyHash> s_batched_nn_simt;

// ---------------------------------------------------------------------------
// Tile selection threshold
// ---------------------------------------------------------------------------

static constexpr int SM_COUNT = 70;  // RTX 5070 Ti

// ---------------------------------------------------------------------------
// Templated dispatch: cache hit → update, else initialize → cache
// ---------------------------------------------------------------------------

template <typename GemmOp, typename CacheMap, typename KeyType>
static int dispatch_gemm(CacheMap& cache,
                          const KeyType& key,
                          typename GemmOp::Arguments& arguments,
                          float* workspace, size_t workspace_bytes,
                          cudaStream_t stream) {
    auto it = cache.find(key);
    if (it != cache.end()) {
        it->second.update(arguments);
        cutlass::Status status = it->second(stream);
        return (status == cutlass::Status::kSuccess) ? 0 : -4;
    }

    GemmOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return -1;

    size_t needed = gemm_op.get_workspace_size(arguments);
    if (needed > workspace_bytes) return -2;

    status = gemm_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -3;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -4;

    cache[key] = gemm_op;
    return 0;
}

// ---------------------------------------------------------------------------
// TF32 alignment check: M, K must be divisible by 4 for TensorOp alignment.
// N can be any value (the epilogue handles unaligned N).
// For LayoutA=RowMajor (TN): alignment of A is on K dimension, B on K.
// For LayoutA=ColMajor (NT/NN): alignment of A is on M dimension, B on K.
// Conservative: require M % 4 == 0 && K % 4 == 0 for all layouts.
// ---------------------------------------------------------------------------

// TF32 align4 epilogue: M and K must be divisible by 4
static bool tf32_align4(int M, int K) {
    return (M % 4 == 0) && (K % 4 == 0);
}

// TF32 align1 epilogue: only K must be divisible by 4 (handles unaligned M)
static bool tf32_align1(int K) {
    return (K % 4 == 0);
}

// ---------------------------------------------------------------------------
// cutlass_gemm_tn: C = alpha * A^T * B + beta * C
//   cuBLAS convention: A stored [K, M] col-major → RowMajor [M, K]
//                      B stored [K, N] col-major → ColumnMajor [K, N]
//                      C stored [M, N] col-major → ColumnMajor [M, N]
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_tn(int M, int N, int K,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float* C, int ldc,
                     float alpha, float beta,
                     float* workspace, size_t workspace_bytes,
                     cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    GemmKey key{M, N, K, lda, ldb, ldc};

    // Tier 1-2: TF32 with align4 epilogue (M and K div4)
    if (tf32_align4(M, K)) {
        int ctas = ((M + 127) / 128) * ((N + 127) / 128);
        if (ctas >= SM_COUNT) {
            typename GemmTN_TF32_Large::Arguments args(problem,
                {(const Element*)A, LayoutRM(lda)}, {(const Element*)B, LayoutCM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmTN_TF32_Large>(s_tn_large, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
        {
            typename GemmTN_TF32_Small::Arguments args(problem,
                {(const Element*)A, LayoutRM(lda)}, {(const Element*)B, LayoutCM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmTN_TF32_Small>(s_tn_small, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
    }
    // Tier 3: TF32 with align1 epilogue (K div4, M may not be)
    if (tf32_align1(K)) {
        typename GemmTN_TF32_Align1::Arguments args(problem,
            {(const Element*)A, LayoutRM(lda)}, {(const Element*)B, LayoutCM(ldb)},
            {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
        if (dispatch_gemm<GemmTN_TF32_Align1>(s_tn_align1, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }
    // Tier 4: SIMT (any alignment)
    typename GemmTN_SIMT::Arguments args(problem,
        {(const Element*)A, LayoutRM(lda)}, {(const Element*)B, LayoutCM(ldb)},
        {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
    return dispatch_gemm<GemmTN_SIMT>(s_tn_simt, key, args, workspace, workspace_bytes, stream);
}

// ---------------------------------------------------------------------------
// cutlass_gemm_nt: C = alpha * A * B^T + beta * C
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_nt(int M, int N, int K,
                     const float* A, int lda, const float* B, int ldb,
                     float* C, int ldc, float alpha, float beta,
                     float* workspace, size_t workspace_bytes, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    GemmKey key{M, N, K, lda, ldb, ldc};

    if (tf32_align4(M, K)) {
        int ctas = ((M + 127) / 128) * ((N + 127) / 128);
        if (ctas >= SM_COUNT) {
            typename GemmNT_TF32_Large::Arguments args(problem,
                {(const Element*)A, LayoutCM(lda)}, {(const Element*)B, LayoutRM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmNT_TF32_Large>(s_nt_large, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
        {
            typename GemmNT_TF32_Small::Arguments args(problem,
                {(const Element*)A, LayoutCM(lda)}, {(const Element*)B, LayoutRM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmNT_TF32_Small>(s_nt_small, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
    }
    if (tf32_align1(K)) {
        typename GemmNT_TF32_Align1::Arguments args(problem,
            {(const Element*)A, LayoutCM(lda)}, {(const Element*)B, LayoutRM(ldb)},
            {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
        if (dispatch_gemm<GemmNT_TF32_Align1>(s_nt_align1, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }
    typename GemmNT_SIMT::Arguments args(problem,
        {(const Element*)A, LayoutCM(lda)}, {(const Element*)B, LayoutRM(ldb)},
        {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
    return dispatch_gemm<GemmNT_SIMT>(s_nt_simt, key, args, workspace, workspace_bytes, stream);
}

// ---------------------------------------------------------------------------
// cutlass_gemm_nn: C = alpha * A * B + beta * C
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_nn(int M, int N, int K,
                     const float* A, int lda, const float* B, int ldb,
                     float* C, int ldc, float alpha, float beta,
                     float* workspace, size_t workspace_bytes, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    GemmKey key{M, N, K, lda, ldb, ldc};

    if (tf32_align4(M, K)) {
        int ctas = ((M + 127) / 128) * ((N + 127) / 128);
        if (ctas >= SM_COUNT) {
            typename GemmNN_TF32_Large::Arguments args(problem,
                {(const Element*)A, LayoutCM(lda)}, {(const Element*)B, LayoutCM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmNN_TF32_Large>(s_nn_large, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
        {
            typename GemmNN_TF32_Small::Arguments args(problem,
                {(const Element*)A, LayoutCM(lda)}, {(const Element*)B, LayoutCM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmNN_TF32_Small>(s_nn_small, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
    }
    if (tf32_align1(K)) {
        typename GemmNN_TF32_Align1::Arguments args(problem,
            {(const Element*)A, LayoutCM(lda)}, {(const Element*)B, LayoutCM(ldb)},
            {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
        if (dispatch_gemm<GemmNN_TF32_Align1>(s_nn_align1, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }
    typename GemmNN_SIMT::Arguments args(problem,
        {(const Element*)A, LayoutCM(lda)}, {(const Element*)B, LayoutCM(ldb)},
        {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
    return dispatch_gemm<GemmNN_SIMT>(s_nn_simt, key, args, workspace, workspace_bytes, stream);
}

// ---------------------------------------------------------------------------
// cutlass_gemm_tn_bias: D = A^T * B + bias  (bias broadcast across columns)
//   bias: [M] vector, fused into epilogue via stride-0 C source.
//   Same as cutlass_gemm_tn but with separate C (bias) and D (output).
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_tn_bias(int M, int N, int K,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float* D, int ldd,
                          const float* bias,
                          float* workspace, size_t workspace_bytes,
                          cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    // Use ldd=0 in key to distinguish bias-fused from regular GEMM
    GemmKey key{M, N, K, lda, ldb, -(ldd + 1)};

    if (tf32_align4(M, K)) {
        int ctas = ((M + 127) / 128) * ((N + 127) / 128);
        if (ctas >= SM_COUNT) {
            typename GemmTN_TF32_Large::Arguments args(problem,
                {(const Element*)A, LayoutRM(lda)}, {(const Element*)B, LayoutCM(ldb)},
                {const_cast<Element*>(bias), LayoutCM(0)},
                {D, LayoutCM(ldd)}, {1.0f, 1.0f});
            if (dispatch_gemm<GemmTN_TF32_Large>(s_tn_large, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
        {
            typename GemmTN_TF32_Small::Arguments args(problem,
                {(const Element*)A, LayoutRM(lda)}, {(const Element*)B, LayoutCM(ldb)},
                {const_cast<Element*>(bias), LayoutCM(0)},
                {D, LayoutCM(ldd)}, {1.0f, 1.0f});
            if (dispatch_gemm<GemmTN_TF32_Small>(s_tn_small, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
    }
    if (tf32_align1(K)) {
        typename GemmTN_TF32_Align1::Arguments args(problem,
            {(const Element*)A, LayoutRM(lda)}, {(const Element*)B, LayoutCM(ldb)},
            {const_cast<Element*>(bias), LayoutCM(0)},
            {D, LayoutCM(ldd)}, {1.0f, 1.0f});
        if (dispatch_gemm<GemmTN_TF32_Align1>(s_tn_align1, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }
    typename GemmTN_SIMT::Arguments args(problem,
        {(const Element*)A, LayoutRM(lda)}, {(const Element*)B, LayoutCM(ldb)},
        {const_cast<Element*>(bias), LayoutCM(0)},
        {D, LayoutCM(ldd)}, {1.0f, 1.0f});
    return dispatch_gemm<GemmTN_SIMT>(s_tn_simt, key, args, workspace, workspace_bytes, stream);
}

// ---------------------------------------------------------------------------
// cutlass_gemm_batched_tn: C_i = alpha * A_i^T * B_i + beta * C_i
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_batched_tn(int M, int N, int K,
                              const float* A, int lda, long long strideA,
                              const float* B, int ldb, long long strideB,
                              float* C, int ldc, long long strideC,
                              int batch_count,
                              float alpha, float beta,
                              float* workspace, size_t workspace_bytes,
                              cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    BatchedKey key{M, N, K, batch_count, strideA, strideB, strideC};

    if (tf32_align4(M, K)) {
        typename GemmBatchedTN_TF32::Arguments args(
            problem, {(const Element*)A, LayoutRM(lda)},
            strideA, {(const Element*)B, LayoutCM(ldb)},
            strideB, {C, LayoutCM(ldc)},
            strideC, {C, LayoutCM(ldc)},
            strideC, {alpha, beta}, batch_count);
        if (dispatch_gemm<GemmBatchedTN_TF32>(s_batched_tn_tf32, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }

    // Fallback: loop of single GEMMs (uses TF32 align1 or SIMT per-element)
    for (int b = 0; b < batch_count; b++) {
        int rc = cutlass_gemm_tn(M, N, K,
            A + b * strideA, lda, B + b * strideB, ldb,
            C + b * strideC, ldc, alpha, beta,
            workspace, workspace_bytes, stream);
        if (rc != 0) return rc;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// cutlass_gemm_batched_nn: C_i = alpha * A_i * B_i + beta * C_i
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_batched_nn(int M, int N, int K,
                              const float* A, int lda, long long strideA,
                              const float* B, int ldb, long long strideB,
                              float* C, int ldc, long long strideC,
                              int batch_count,
                              float alpha, float beta,
                              float* workspace, size_t workspace_bytes,
                              cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    BatchedKey key{M, N, K, batch_count, strideA, strideB, strideC};

    if (tf32_align4(M, K)) {
        typename GemmBatchedNN_TF32::Arguments args(
            problem, {(const Element*)A, LayoutCM(lda)},
            strideA, {(const Element*)B, LayoutCM(ldb)},
            strideB, {C, LayoutCM(ldc)},
            strideC, {C, LayoutCM(ldc)},
            strideC, {alpha, beta}, batch_count);
        if (dispatch_gemm<GemmBatchedNN_TF32>(s_batched_nn_tf32, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }

    // Fallback: loop of single GEMMs (uses TF32 align1 or SIMT per-element)
    for (int b = 0; b < batch_count; b++) {
        int rc = cutlass_gemm_nn(M, N, K,
            A + b * strideA, lda, B + b * strideB, ldb,
            C + b * strideC, ldc, alpha, beta,
            workspace, workspace_bytes, stream);
        if (rc != 0) return rc;
    }
    return 0;
}
