#include <iostream>
#include <iomanip>
#include <omp.h>

int main() {
    long long num_steps = 1000000000;
    double step = 1.0 / (double)num_steps;
    double pi = 0.0;
    
    // Warm-up
    {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (long long i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
    }

    int num_runs = 5;
    double total_time = 0.0;
    pi = 0.0;

    for (int r = 0; r < num_runs; r++) {
        pi = 0.0;
        double start_time = omp_get_wtime();
        
        #pragma omp parallel
        {
            double sum = 0.0;
            #pragma omp for
            for (long long i = 0; i < num_steps; i++) {
                double x = (i + 0.5) * step;
                sum += 4.0 / (1.0 + x * x);
            }
            #pragma omp atomic
            pi += sum * step;
        }
        
        double run_time = omp_get_wtime() - start_time;
        total_time += run_time;
    }
    
    std::cout << "pi = " << std::setprecision(15) << pi << std::endl;
    std::cout << "Time: " << (total_time / num_runs) << " seconds" << std::endl;
    
    return 0;
}
