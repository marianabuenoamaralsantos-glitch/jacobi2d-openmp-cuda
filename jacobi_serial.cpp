#include <cstddef>
#include <iostream>
#include <vector>

void jacobi_serial(std::vector<double>& atual, std::size_t n, int iteracoes) {
    std::vector<double> proxima = atual;

    for (int passo = 0; passo < iteracoes; ++passo) {
        for (std::size_t i = 1; i < n - 1; ++i) {
            for (std::size_t j = 1; j < n - 1; ++j) {
                const std::size_t k = i * n + j;
                proxima[k] = 0.25 * (atual[k - n] + atual[k + n]
                                   + atual[k - 1] + atual[k + 1]);
            }
        }
        atual.swap(proxima);
    }
}

void jacobi_openmp(std::vector<double>& atual, std::size_t n, int iteracoes);
#ifdef JACOBI_COM_CUDA
void jacobi_cuda(std::vector<double>& atual, std::size_t n, int iteracoes);
#endif

int main() {
    constexpr std::size_t n = 4;
    constexpr int iteracoes = 2;
    std::vector<double> matriz(n * n, 1.0);

    for (std::size_t i = 1; i < n - 1; ++i) {
        for (std::size_t j = 1; j < n - 1; ++j) {
            matriz[i * n + j] = 0.0;
        }
    }

    auto matriz_openmp = matriz;

#ifdef JACOBI_COM_CUDA
    auto matriz_cuda = matriz;
#endif

    jacobi_serial(matriz, n, iteracoes);

#ifdef JACOBI_COM_CUDA
    jacobi_cuda(matriz_cuda, n, iteracoes);
    if (matriz_cuda != matriz) {
        std::cerr << "FAIL: CUDA divergiu da referencia sequencial\n";
        return 1;
    }
    std::cout << "PASS: CUDA igual ao sequencial nos 16 elementos\n";
#endif
    jacobi_openmp(matriz_openmp, n, iteracoes);

    if (matriz_openmp != matriz) {
        std::cerr << "FAIL: OpenMP divergiu da referencia sequencial\n";
        return 1;
    }
    std::cout << "PASS: OpenMP igual ao sequencial nos 16 elementos\n";

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const bool borda = i == 0 || j == 0 || i == n - 1 || j == n - 1;
            const double esperado = borda ? 1.0 : 0.75;
            if (matriz[i * n + j] != esperado) {
                std::cerr << "Falha na posicao (" << i << ", " << j << ")\n";
                return 1;
            }
        }
    }

    std::cout << "PASS: Jacobi sequencial 4x4, 2 iteracoes\n";
    return 0;
}
