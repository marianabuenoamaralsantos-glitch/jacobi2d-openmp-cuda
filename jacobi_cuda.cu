#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

__global__ void atualizar(const double* atual, double* proxima, std::size_t n) {
    const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const std::size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < n - 1 && j < n - 1) {
        const std::size_t k = i * n + j;
        proxima[k] = 0.25 * (atual[k - n] + atual[k + n]
                           + atual[k - 1] + atual[k + 1]);
    }
}

void jacobi_cuda(std::vector<double>& atual, std::size_t n, int iteracoes) {
    if (n < 3 || iteracoes <= 0) {
        return;
    }

    thrust::device_vector<double> entrada(atual.begin(), atual.end());
    thrust::device_vector<double> saida = entrada;

    const dim3 bloco(16, 16);
    const unsigned blocos = (n - 2 + bloco.x - 1) / bloco.x;
    const dim3 grade(blocos, blocos);

    for (int passo = 0; passo < iteracoes; ++passo) {
        atualizar<<<grade, bloco>>>(thrust::raw_pointer_cast(entrada.data()),
                                   thrust::raw_pointer_cast(saida.data()), n);
        const cudaError_t erro = cudaGetLastError();
        if (erro != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(erro));
        }
        entrada.swap(saida);
    }

    thrust::copy(entrada.begin(), entrada.end(), atual.begin());
}
