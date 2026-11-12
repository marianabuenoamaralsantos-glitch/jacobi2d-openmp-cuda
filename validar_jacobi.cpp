#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <limits>
#include <vector>

void jacobi_serial(std::vector<double>&, std::size_t, int);
void jacobi_openmp(std::vector<double>&, std::size_t, int);
void jacobi_cuda(std::vector<double>&, std::size_t, int);

bool validar(const char* nome, const std::vector<double>& referencia,
             const std::vector<double>& resultado) {
    constexpr double atol = 1e-12;
    constexpr double rtol = 1e-10;
    double max_abs = 0.0;
    double max_rel = 0.0;
    bool passou = true;

    for (std::size_t k = 0; k < referencia.size(); ++k) {
        const double ref = std::abs(referencia[k]);
        const double erro = std::abs(resultado[k] - referencia[k]);
        if (!std::isfinite(erro)) {
            passou = false;
            max_abs = max_rel = std::numeric_limits<double>::infinity();
            continue;
        }

        const double relativo = ref > 0.0 ? erro / ref
            : (erro == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());

        max_abs = std::max(max_abs, erro);
        max_rel = std::max(max_rel, relativo);
        passou = passou && erro <= atol + rtol * ref;
    }

    std::cout << nome << ' ' << (passou ? "PASS" : "FAIL")
              << " | max_abs=" << max_abs << " | max_rel=" << max_rel << '\n';
    return passou;
}

int main(int argc, char** argv) {
    int tamanho = 0;
    int iteracoes = 0;
    try {
        if (argc == 3) {
            std::size_t fim_n = 0;
            std::size_t fim_iteracoes = 0;
            tamanho = std::stoi(argv[1], &fim_n);
            iteracoes = std::stoi(argv[2], &fim_iteracoes);
            if (argv[1][fim_n] != '\0' || argv[2][fim_iteracoes] != '\0') {
                tamanho = 0;
            }
        }
    } catch (const std::exception&) {
        tamanho = 0;
    }

    if (tamanho < 3 || iteracoes < 1) {
        std::cerr << "Uso: ./validar_jacobi N ITERACOES (N >= 3, ITERACOES >= 1)\n";
        return 1;
    }
    const auto n = static_cast<std::size_t>(tamanho);
    std::vector<double> entrada(n * n, 1.0);

    for (std::size_t i = 1; i < n - 1; ++i) {
        for (std::size_t j = 1; j < n - 1; ++j) {
            entrada[i * n + j] = ((3 * i + 7 * j) % 101) / 100.0;
        }
    }

    auto serial = entrada;
    auto openmp = entrada;
    auto cuda = entrada;

    const auto medir = [&](const char* nome, decltype(&jacobi_serial) executar,
                           std::vector<double>& matriz) {
        const auto inicio = std::chrono::steady_clock::now();
        executar(matriz, n, iteracoes);
        const auto fim = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> tempo = fim - inicio;
        std::cout << nome << " | tempo_ms=" << tempo.count() << '\n';
    };

    jacobi_serial(serial, n, iteracoes);
    jacobi_openmp(openmp, n, iteracoes);
    jacobi_cuda(cuda, n, iteracoes);

    for (int repeticao = 1; repeticao <= 5; ++repeticao) {
        serial = entrada;
        openmp = entrada;
        cuda = entrada;

        std::cout << "Repeticao " << repeticao << '\n';
        medir("Serial", jacobi_serial, serial);
        medir("OpenMP", jacobi_openmp, openmp);
        medir("CUDA", jacobi_cuda, cuda);
    }

    const bool ok_openmp = validar("OpenMP", serial, openmp);
    const bool ok_cuda = validar("CUDA", serial, cuda);
    return ok_openmp && ok_cuda ? 0 : 1;
}
