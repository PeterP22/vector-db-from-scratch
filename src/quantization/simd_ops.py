"""SIMD-optimized operations for quantized vectors.

In production, these operations would be implemented in C/C++ using CPU
SIMD intrinsics (AVX2, AVX-512, NEON). Here we use NumPy's vectorized
operations which internally use SIMD when available.

Key Concepts:
1. SIMD (Single Instruction Multiple Data): Process multiple data points in parallel
2. Uint8 operations: Modern CPUs can process 32-64 uint8 values in one instruction
3. Distance computation: Vectorized operations are 4-10x faster than loops
4. Quantized data benefits: Smaller data types → more values per SIMD register

SIMD Width Examples:
- AVX2 (256-bit): 32 uint8 values at once
- AVX-512 (512-bit): 64 uint8 values at once
- ARM NEON (128-bit): 16 uint8 values at once

Production Examples:
- Qdrant: Hand-coded SIMD for distance computation
- Elasticsearch: AVX2-optimized quantized distance
- pgvector: AVX-512 support for binary distances
"""

import numpy as np
from typing import List
import time

from ..core.vector import Vector


class SIMDDistanceComputer:
    """SIMD-optimized distance computations for quantized vectors.

    Note: This uses NumPy's vectorized operations which are SIMD-optimized
    internally. For true control, you'd implement in C++ with intrinsics.
    """

    @staticmethod
    def l2_distance_batch_simd(
        query: np.ndarray,
        database: np.ndarray
    ) -> np.ndarray:
        """Compute L2 distances from query to all database vectors using SIMD.

        This is the vectorized (SIMD) version that processes all vectors
        in parallel using NumPy's optimized routines.

        Args:
            query: Query vector (uint8 or float32), shape (d,)
            database: Database vectors, shape (n, d)

        Returns:
            Array of distances, shape (n,)

        Note:
            NumPy internally uses SIMD (AVX2/AVX-512) for these operations
            when arrays are properly aligned and have compatible types.
        """
        # Compute differences: (n, d)
        diff = database.astype(np.int16) - query.astype(np.int16)

        # Squared differences and sum
        squared_diff = diff ** 2
        distances = np.sqrt(np.sum(squared_diff, axis=1))

        return distances

    @staticmethod
    def l2_distance_batch_naive(
        query: np.ndarray,
        database: List[np.ndarray]
    ) -> List[float]:
        """Compute L2 distances using naive Python loop (no SIMD).

        This is the non-vectorized version for comparison.

        Args:
            query: Query vector
            database: List of database vectors

        Returns:
            List of distances
        """
        distances = []
        for db_vec in database:
            diff = db_vec.astype(np.int16) - query.astype(np.int16)
            dist = float(np.sqrt(np.sum(diff ** 2)))
            distances.append(dist)
        return distances

    @staticmethod
    def dot_product_batch_simd(
        query: np.ndarray,
        database: np.ndarray
    ) -> np.ndarray:
        """Compute dot products using SIMD-optimized matrix multiplication.

        Args:
            query: Query vector, shape (d,)
            database: Database vectors, shape (n, d)

        Returns:
            Array of dot products, shape (n,)
        """
        # NumPy's dot/matmul use highly optimized BLAS routines (often with SIMD)
        return np.dot(database, query)

    @staticmethod
    def cosine_similarity_batch_simd(
        query: np.ndarray,
        database: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarities using SIMD operations.

        Args:
            query: Query vector, shape (d,)
            database: Database vectors, shape (n, d)

        Returns:
            Array of cosine similarities, shape (n,)
        """
        # Dot products
        dots = np.dot(database, query)

        # Norms
        query_norm = np.linalg.norm(query)
        db_norms = np.linalg.norm(database, axis=1)

        # Cosine similarity
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = dots / (query_norm * db_norms)
            similarities = np.nan_to_num(similarities, nan=0.0)

        return similarities


def benchmark_simd_speedup():
    """Benchmark SIMD vs naive implementations."""
    print("\n" + "=" * 60)
    print("SIMD Speedup Demonstration")
    print("=" * 60)

    computer = SIMDDistanceComputer()

    test_cases = [
        {"n": 1000, "d": 128, "name": "1K vectors @ 128D"},
        {"n": 10000, "d": 128, "name": "10K vectors @ 128D"},
        {"n": 10000, "d": 512, "name": "10K vectors @ 512D"},
    ]

    for test in test_cases:
        print(f"\n{test['name']}:")

        # Generate quantized data (uint8)
        query = np.random.randint(0, 256, test["d"], dtype=np.uint8)
        database = np.random.randint(0, 256, (test["n"], test["d"]), dtype=np.uint8)

        # SIMD version (vectorized)
        start = time.time()
        simd_dists = computer.l2_distance_batch_simd(query, database)
        simd_time = time.time() - start

        # Naive version (loop)
        database_list = [database[i] for i in range(test["n"])]
        start = time.time()
        naive_dists = computer.l2_distance_batch_naive(query, database_list)
        naive_time = time.time() - start

        # Verify results match
        max_diff = np.max(np.abs(simd_dists - naive_dists))

        speedup = naive_time / simd_time

        print(f"  SIMD (vectorized): {simd_time:.4f}s")
        print(f"  Naive (loop): {naive_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Max difference: {max_diff:.6f}")

        # QPS calculation
        qps_simd = test["n"] / simd_time
        qps_naive = test["n"] / naive_time

        print(f"  Throughput (SIMD): {qps_simd:,.0f} vectors/sec")
        print(f"  Throughput (naive): {qps_naive:,.0f} vectors/sec")


def demonstrate_quantization_simd_benefit():
    """Show why quantization + SIMD is powerful."""
    print("\n" + "=" * 60)
    print("Quantization + SIMD Benefits")
    print("=" * 60)

    dimension = 128
    n_vectors = 10000

    print(f"\nDataset: {n_vectors:,} vectors @ {dimension}D")

    # Float32 data
    query_f32 = np.random.randn(dimension).astype(np.float32)
    database_f32 = np.random.randn(n_vectors, dimension).astype(np.float32)

    # Uint8 data (quantized)
    query_u8 = (np.random.rand(dimension) * 255).astype(np.uint8)
    database_u8 = (np.random.rand(n_vectors, dimension) * 255).astype(np.uint8)

    computer = SIMDDistanceComputer()

    # Benchmark float32
    start = time.time()
    dists_f32 = computer.l2_distance_batch_simd(query_f32, database_f32)
    time_f32 = time.time() - start

    # Benchmark uint8
    start = time.time()
    dists_u8 = computer.l2_distance_batch_simd(query_u8, database_u8)
    time_u8 = time.time() - start

    speedup = time_f32 / time_u8

    print(f"\nFloat32 (no quantization):")
    print(f"  Time: {time_f32:.4f}s")
    print(f"  Memory: {database_f32.nbytes / 1024 / 1024:.2f} MB")

    print(f"\nUint8 (quantized):")
    print(f"  Time: {time_u8:.4f}s")
    print(f"  Memory: {database_u8.nbytes / 1024 / 1024:.2f} MB")

    print(f"\nBenefits:")
    print(f"  Speed improvement: {speedup:.2f}x")
    print(f"  Memory reduction: {database_f32.nbytes / database_u8.nbytes:.2f}x")
    print(f"  Combined benefit: {speedup * (database_f32.nbytes / database_u8.nbytes):.2f}x")

    print(f"\nWhy quantization + SIMD is powerful:")
    print(f"  1. Uint8 fits 4x more values per cache line")
    print(f"  2. SIMD registers process 4x more uint8 than float32")
    print(f"  3. Integer operations are faster than floating point")
    print(f"  4. Less memory bandwidth required")


def explain_simd():
    """Explain SIMD concepts."""
    print("\n" + "=" * 60)
    print("SIMD Explanation")
    print("=" * 60)

    print("""
SIMD (Single Instruction Multiple Data):
- Process multiple data elements in parallel with one instruction
- Modern CPUs have SIMD instruction sets: AVX2, AVX-512, NEON

Example - Adding 8 numbers:
  Scalar (no SIMD): 8 separate add operations
  SIMD (AVX2):      1 operation on 8-element vector

For uint8 distance computation:
  - AVX2 (256-bit): Process 32 uint8 values at once
  - AVX-512 (512-bit): Process 64 uint8 values at once
  - This gives 4-8x speedup over scalar code

Why quantization helps SIMD:
  - float32: 32 bits per value → 8 values per 256-bit register
  - uint8:   8 bits per value  → 32 values per 256-bit register
  - 4x more values processed per instruction!

Production implementation (C++ with AVX2):
  __m256i a = _mm256_load_si256(vec_a);  // Load 32 uint8
  __m256i b = _mm256_load_si256(vec_b);  // Load 32 uint8
  __m256i diff = _mm256_sub_epi8(a, b);  // 32 subtractions at once
  // ... continue with distance calculation

Real-world speedups:
  - Qdrant: 4-6x with SIMD + scalar quantization
  - Elasticsearch: 5-10x with AVX2 optimizations
  - pgvector: Up to 32x with AVX-512 + binary quantization
""")


def main():
    """Run all SIMD demonstrations."""
    print("\n" + "=" * 60)
    print("SIMD Operations for Quantized Vectors")
    print("=" * 60)

    # Explain SIMD
    explain_simd()

    # Benchmark speedup
    benchmark_simd_speedup()

    # Show quantization + SIMD benefits
    demonstrate_quantization_simd_benefit()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("\n1. SIMD processes multiple values in parallel (e.g., 32 uint8 at once)")
    print("2. Quantization enables SIMD: more values fit in SIMD registers")
    print("3. Combined speedup: 4-10x from SIMD + quantization")
    print("4. Production systems (Qdrant, pgvector) use hand-coded SIMD")
    print("5. NumPy internally uses SIMD, but C++ gives full control")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
