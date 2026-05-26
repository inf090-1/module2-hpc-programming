# 6. Cholesky com BLAS (kernels) + OpenMP tasks (grafo de dependências)

Nesta lição você implementa/entende um **Cholesky bloqueado (in-place)** com paralelismo via **OpenMP tasks**.
As operações pesadas do algoritmo usam chamadas de **BLAS**:
- `dtrsm` (triangular solve) para calcular os blocos do painel `L(i,k)`
- `dsyrk` para atualizar blocos diagonais do trailing submatrix
- `dgemm` para atualizar blocos fora da diagonal

A fatoração do bloco diagonal `L(k,k)` é feita com um Cholesky simples (loops) para manter o exemplo curto; o paralelismo principal e as atualizações ficam por conta de tarefas + BLAS.

## Arquivos
- `cholesky_blas.c`: versão completa (OpenMP tasks + BLAS).
- `CMakeLists.txt`: usa CMake para achar e linkar uma BLAS e habilitar OpenMP.

## Build com CMake

```bash
cd 05-openmp-advanced/3-cholesky-blas-cmake
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executável:
- `build/cholesky_blas`

## Execução
```bash
./build/cholesky_blas 512 64 0
```

- `n` = tamanho da matriz SPD
- `block` = tamanho do bloco (tile) usado para particionar a matriz
- `seed` = semente do gerador aleatório

O programa imprime um checksum (`diag_sum`) e um `residual_Frob` para validar o fator.

## Ideia das tarefas (visão rápida)
Para cada bloco-coluna `k`, o algoritmo executa em **fases** (com `#pragma omp taskwait` entre elas):
1. `factor(k,k)`: fatorar o bloco diagonal com loops.
2. `trsm(i,k)`: para cada `i>k`, calcular `L(i,k)` com `dtrsm` (tasks em paralelo).
3. `update(i,i)`: atualizar cada bloco diagonal com `dsyrk` (tasks em paralelo).
4. `update(i,j)`: atualizar blocos fora da diagonal com `dgemm` (para i>j, também em paralelo).

As fases com `taskwait` garantem que as atualizações necessárias terminem antes de avançar para o próximo `k`.

## Question
- Como o tamanho do `block` afeta:
  - o paralelismo criado por tasks?
  - o tamanho dos problemas enviados ao BLAS (eficiência do Level-3)?
- O speedup vem mais de tasks (CPU) ou do próprio paralelismo interno do BLAS (dependendo da lib)?
