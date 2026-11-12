# Jacobi 2D com OpenMP e CUDA

Trabalho da disciplina MO652 — Computação de Alto Desempenho, UNICAMP.

**Autora:** Mariana Bueno Amaral dos Santos.

O projeto implementa o método iterativo de Jacobi 2D em três versões, sequencial em C++, paralela com OpenMP e paralela com CUDA. Cada elemento interno recebe a média aritmética dos quatro vizinhos da iteração anterior. As bordas permanecem constantes durante um número fixo de iterações.

## Estrutura do repositório

Os arquivos devem ficar na raiz do repositório.

```text
jacobi2d-openmp-cuda/
├── README.md
├── jacobi_serial.cpp
├── jacobi_openmp.cpp
├── jacobi_openmp_otimizado.cpp
├── jacobi_openmp_simd.cpp
├── jacobi_cuda.cu
├── jacobi_cuda_blocos.cu
├── jacobi_cuda_shared.cu
└── validar_jacobi.cpp
```

| Arquivo | Função |
|---|---|
| `jacobi_serial.cpp` | Referência sequencial e teste com resultado conhecido em matriz 4×4. |
| `jacobi_openmp.cpp` | OpenMP básica, com uma região paralela por iteração. |
| `jacobi_openmp_otimizado.cpp` | OpenMP com região paralela persistente entre as iterações. |
| `jacobi_openmp_simd.cpp` | Região persistente com diretiva SIMD no laço das colunas. |
| `jacobi_cuda.cu` | CUDA básica, com memória global e blocos 16×16. |
| `jacobi_cuda_blocos.cu` | CUDA com memória global e dimensões de bloco configuráveis na compilação. |
| `jacobi_cuda_shared.cu` | CUDA com memória compartilhada e dimensões de bloco configuráveis. |
| `validar_jacobi.cpp` | Recebe parâmetros, mede tempos e valida as versões paralelas contra a sequencial. |

## Requisitos e ambiente utilizado

A compilação CPU requer um compilador C++ com suporte a C++11 e OpenMP. A compilação e a execução CUDA requerem o CUDA Toolkit, um driver compatível e uma GPU NVIDIA. As bibliotecas Thrust usadas nos fontes acompanham o CUDA Toolkit.

Os experimentos foram realizados no Lovelace do CENAPAD-SP. O ambiente registrado foi AlmaLinux 8.10, GCC 8.5.0, módulo `cuda/12.8.1`, compilador `nvcc V12.8.93` e driver NVIDIA 570.124.06. O nó possui dois AMD EPYC 7662 de 64 núcleos e 502 GiB de RAM visíveis. A fila `testegpu` disponibilizou oito CPUs, 30 GB de RAM e uma instância MIG `4g.20gb` da NVIDIA A100 PCIe de 40 GB, com 20.096 MiB e 56 SMs.

Os comandos CUDA abaixo usam `-arch=sm_80`, conforme a A100 utilizada. Em outra GPU, ajuste a arquitetura de compilação ao dispositivo disponível.

Execute os comandos a partir da raiz do repositório, em Bash. Cada ligação deve selecionar exatamente uma variante OpenMP e uma variante CUDA, pois os fontes alternativos definem os mesmos nomes de função. Execute as compilações e os testes em sequência, pois os comandos reutilizam os nomes dos objetos e executáveis.

## Executar no Lovelace

No terminal de login do Lovelace, solicite uma sessão interativa com GPU.

```bash
qsub -I -q testegpu -l walltime=00:05:00
```

Quando a sessão iniciar, acesse a pasta onde colocou o repositório e carregue o módulo CUDA.

```bash
cd /caminho/para/jacobi2d-openmp-cuda
module load cuda/12.8.1
```

Substitua o caminho acima pelo diretório real. A sessão tem limite de cinco minutos. Se ela terminar, solicite outra sessão e carregue o módulo novamente antes de continuar. As instruções seguintes compilam e executam diretamente os fontes presentes neste repositório.

Em um computador próprio com CUDA configurado, use os comandos de compilação e execução abaixo no ambiente local.

## Teste inicial CPU com resultado conhecido

Este teste compara sequencial e OpenMP em uma matriz 4×4, com interior inicial zero, bordas iguais a 1 e duas iterações. O resultado esperado é 0,75 nas quatro posições internas e 1 nas bordas.

```bash
g++ -O3 -std=c++11 -Wall -Wextra -fopenmp \
    jacobi_serial.cpp jacobi_openmp.cpp -o teste_jacobi
OMP_NUM_THREADS=2 ./teste_jacobi
```

Saída esperada quando aprovado.

```text
PASS: OpenMP igual ao sequencial nos 16 elementos
PASS: Jacobi sequencial 4x4, 2 iteracoes
```

## Compilar o validador com as versões básicas

O arquivo sequencial contém também o `main` do teste pequeno. Para utilizá-lo como referência no validador, a compilação renomeia apenas esse `main` por meio da opção `-Dmain=main_teste_4x4`.

```bash
g++ -O3 -std=c++11 -Wall -Wextra -Dmain=main_teste_4x4 \
    -c jacobi_serial.cpp -o jacobi_serial_validacao.o

g++ -O3 -std=c++11 -Wall -Wextra -fopenmp \
    -c jacobi_openmp.cpp -o jacobi_openmp.o

g++ -O3 -std=c++11 -Wall -Wextra -fopenmp \
    -c validar_jacobi.cpp -o validar_jacobi.o

nvcc -O3 -arch=sm_80 -Xcompiler=-fopenmp \
    jacobi_cuda.cu jacobi_serial_validacao.o jacobi_openmp.o \
    validar_jacobi.o -o validar_jacobi
```

Execute o validador com o tamanho da matriz e a quantidade de iterações.

```bash
OMP_DYNAMIC=FALSE OMP_NUM_THREADS=2 ./validar_jacobi 1024 100
```

O programa recebe `N ITERACOES`, com `N >= 3` e `ITERACOES >= 1`. Escolha dimensões compatíveis com a memória disponível. Os testes de desempenho do trabalho utilizaram N até 4096.

A saída apresenta cinco grupos de tempos em milissegundos, seguidos por `OpenMP PASS` e `CUDA PASS` quando os resultados são aprovados. Cada linha de validação informa `max_abs` e `max_rel`. O código de saída é zero somente quando ambas as comparações passam.

## Compilar as otimizações OpenMP

Primeiro, compile os objetos sequencial e do validador conforme a seção anterior. Para testar a região paralela persistente, execute os comandos abaixo.

```bash
g++ -O3 -std=c++11 -Wall -Wextra -fopenmp \
    -c jacobi_openmp_otimizado.cpp -o jacobi_openmp.o

nvcc -O3 -arch=sm_80 -Xcompiler=-fopenmp \
    jacobi_cuda.cu jacobi_serial_validacao.o jacobi_openmp.o \
    validar_jacobi.o -o validar_jacobi

OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close \
    OMP_NUM_THREADS=2 ./validar_jacobi 4096 100
```

Para testar a região persistente com SIMD, recompile o objeto OpenMP e faça a ligação novamente.

```bash
g++ -O3 -std=c++11 -Wall -Wextra -fopenmp \
    -c jacobi_openmp_simd.cpp -o jacobi_openmp.o

nvcc -O3 -arch=sm_80 -Xcompiler=-fopenmp \
    jacobi_cuda.cu jacobi_serial_validacao.o jacobi_openmp.o \
    validar_jacobi.o -o validar_jacobi

OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close \
    OMP_NUM_THREADS=2 ./validar_jacobi 4096 100
```

## Compilar as configurações CUDA

Primeiro, compile os objetos sequencial e do validador conforme a seção das versões básicas. Os comandos abaixo selecionam a OpenMP básica para acompanhar a comparação CUDA.

```bash
g++ -O3 -std=c++11 -Wall -Wextra -fopenmp \
    -c jacobi_openmp.cpp -o jacobi_openmp.o
```

Para memória global com bloco 32×8, execute os comandos abaixo.

```bash
nvcc -O3 -arch=sm_80 -Xcompiler=-fopenmp \
    -DBLOCO_X=32 -DBLOCO_Y=8 jacobi_cuda_blocos.cu \
    jacobi_serial_validacao.o jacobi_openmp.o validar_jacobi.o \
    -o validar_jacobi_blocos

OMP_DYNAMIC=FALSE OMP_NUM_THREADS=2 ./validar_jacobi_blocos 4096 100
```

Para memória compartilhada com o mesmo bloco, execute os comandos abaixo.

```bash
nvcc -O3 -arch=sm_80 -Xcompiler=-fopenmp \
    -DBLOCO_X=32 -DBLOCO_Y=8 jacobi_cuda_shared.cu \
    jacobi_serial_validacao.o jacobi_openmp.o validar_jacobi.o \
    -o validar_jacobi_shared

OMP_DYNAMIC=FALSE OMP_NUM_THREADS=2 ./validar_jacobi_shared 131 21
OMP_DYNAMIC=FALSE OMP_NUM_THREADS=2 ./validar_jacobi_shared 4096 100
```

O teste 131×131 com 21 iterações verifica blocos parcialmente preenchidos e uma quantidade ímpar de trocas. Para avaliar outras geometrias, altere `BLOCO_X` e `BLOCO_Y` e recompile o executável correspondente. O trabalho comparou blocos 16×16, 32×8 e 32×16.

## Variar tamanhos e quantidade de threads

Depois de compilar o validador básico, o exemplo abaixo executa os quatro tamanhos usados na campanha inicial, com duas threads OpenMP.

```bash
for n in 512 1024 2048 4096; do
    echo "N=$n"
    OMP_DYNAMIC=FALSE OMP_NUM_THREADS=2 ./validar_jacobi "$n" 100
done
```

Para avaliar a escalabilidade da variante OpenMP atualmente ligada ao executável, mantenha N fixo e varie as threads.

```bash
for threads in 1 2 4 8; do
    echo "threads=$threads"
    OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close \
        OMP_NUM_THREADS="$threads" ./validar_jacobi 4096 100
done
```

Repita a compilação e esse teste para cada variante OpenMP. O validador sempre executa as três versões selecionadas na ligação. Os tempos identificados como `OpenMP` correspondem à variante que foi compilada em `jacobi_openmp.o`.

## Inicialização e validação numérica

As matrizes usam precisão `double`. As bordas são fixadas em 1 e o interior é inicializado por `((3*i + 7*j) % 101) / 100.0`. As três versões recebem cópias da mesma entrada e executam a mesma quantidade de iterações.

A validação compara todos os elementos da última repetição com a referência sequencial. Cada elemento é aprovado quando satisfaz a condição abaixo.

```text
abs(resultado - referencia) <= 1e-12 + 1e-10 * abs(referencia)
```

O programa informa o maior erro absoluto e o maior erro relativo entre todos os elementos, incluindo as bordas. Para referência zero, o erro relativo vale zero quando a diferença também é zero e infinito caso contrário. Diferenças não finitas reprovam a comparação.

## Medição de desempenho

Antes das cinco execuções cronometradas, cada versão realiza uma execução preliminar não contabilizada nos resultados. Esse procedimento reduz a influência dos custos de primeira utilização, como a inicialização do ambiente CUDA.

O cronômetro `steady_clock` mede a função completa. Na CPU, inclui a alocação e a cópia do segundo buffer. No CUDA, inclui alocações, transferências CPU/GPU, kernels e retorno à CPU. A cópia final disponibiliza o resultado no host antes do término da medição. A restauração da entrada, a compilação e a validação ficam fora do intervalo cronometrado.

A ordem das chamadas é sempre sequencial, OpenMP e CUDA. A validação é realizada após a quinta repetição. Para comparar tempos, o relatório utiliza a mediana das cinco observações. O speedup entre versões é a razão entre a mediana sequencial e a mediana paralela para o mesmo N. A escalabilidade OpenMP usa a execução de uma thread da mesma variante como referência.

Os resultados e a discussão experimental são apresentados no relatório entregue separadamente no Classroom.
