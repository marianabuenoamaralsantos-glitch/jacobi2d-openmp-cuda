#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

__global__ void atualizar(const double* atual, double* proxima,
                          std::size_t n) {
    extern __shared__ double tile[];

    const unsigned largura = blockDim.x + 2;
    const unsigned altura = blockDim.y + 2;
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned threads = blockDim.x * blockDim.y;
    const std::size_t origem_j = blockIdx.x * blockDim.x;
    const std::size_t origem_i = blockIdx.y * blockDim.y;

    for (unsigned p = tid; p < largura * altura; p += threads) {
        const std::size_t i = origem_i + p / largura;
        const std::size_t j = origem_j + p % largura;
        tile[p] = (i < n && j < n) ? atual[i * n + j] : 0.0;
    }
    __syncthreads();

    const std::size_t i = origem_i + threadIdx.y + 1;
    const std::size_t j = origem_j + threadIdx.x + 1;
    const unsigned p = (threadIdx.y + 1) * largura + threadIdx.x + 1;

    if (i < n - 1 && j < n - 1) {
        proxima[i * n + j] = 0.25 * (
            tile[p - largura] + tile[p + largura]
            + tile[p - 1] + tile[p + 1]);
    }
}

void jacobi_cuda(std::vector<double>& atual, std::size_t n, int iteracoes) {
    if (n < 3 || iteracoes <= 0) {
        return;
    }

    thrust::device_vector<double> entrada(atual.begin(), atual.end());
    thrust::device_vector<double> saida = entrada;

    const dim3 bloco(BLOCO_X, BLOCO_Y);
    const unsigned grade_x = (n - 2 + bloco.x - 1) / bloco.x;
    const unsigned grade_y = (n - 2 + bloco.y - 1) / bloco.y;
    const dim3 grade(grade_x, grade_y);

    for (int passo = 0; passo < iteracoes; ++passo) {
        atualizar<<<grade, bloco,
            (bloco.x + 2) * (bloco.y + 2) * sizeof(double)>>>(thrust::raw_pointer_cast(entrada.data()),
                                   thrust::raw_pointer_cast(saida.data()), n);
        const cudaError_t erro = cudaGetLastError();
        if (erro != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(erro));
        }
        entrada.swap(saida);
    }

    thrust::copy(entrada.begin(), entrada.end(), atual.begin());
}
