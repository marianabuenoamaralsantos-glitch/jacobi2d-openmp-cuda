#include <cstddef>
#include <vector>

void jacobi_openmp(std::vector<double>& atual, std::size_t n, int iteracoes) {
    std::vector<double> proxima = atual;

    for (int passo = 0; passo < iteracoes; ++passo) {
        #pragma omp parallel for schedule(static)
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
