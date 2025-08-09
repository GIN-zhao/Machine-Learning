#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define TILE_WIDTH 16

__global__ void matrixMulShared(float *C, float *A, float *B, int M, int N,
                                int K) {
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float Cvalue = 0.0f;

  for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    if (row < M && (t * TILE_WIDTH + tx) < K) {
      As[ty][tx] = A[row * K + (t * TILE_WIDTH + tx)];
    } else {
      As[ty][tx] = 0.0f;
    }

    if ((t * TILE_WIDTH + ty) < K && col < N) {
      Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; ++i) {
      Cvalue += As[ty][i] * Bs[i][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = Cvalue;
  }
}

void printMatrix(const std::vector<float> &matrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main() {
  int M = 256;
  int N = 256;
  int K = 256;

  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::vector<float> h_C(M * N);

  for (int i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixMulShared<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, M, N, K);

  cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  std::cout << "Matrix multiplication with shared memory completed."
            << std::endl;

  return 0;
}
