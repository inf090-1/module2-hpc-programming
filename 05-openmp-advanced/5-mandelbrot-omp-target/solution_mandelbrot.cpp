#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <climits>

// Multi-device Mandelbrot using OpenMP target offload.
//
// The image is split into vertical strips, one per device.
// Each device runs a single target region with a 1D flattened loop
// over its strip, giving maximum parallelism.

#ifndef WIDTH
#define WIDTH 4096
#endif
#ifndef HEIGHT
#define HEIGHT 4096
#endif
#ifndef MAX_ITER
#define MAX_ITER 5000
#endif

static inline int mandelbrot_iter(float cr, float ci, int max_iter) {
    float zr = 0.0f;
    float zi = 0.0f;
    int iter = 0;

    while (iter < max_iter) {
        float zr2 = zr * zr;
        float zi2 = zi * zi;
        if (zr2 + zi2 > 4.0f) break;

        float new_zr = zr2 - zi2 + cr;
        float new_zi = 2.0f * zr * zi + ci;
        zr = new_zr;
        zi = new_zi;
        ++iter;
    }

    return iter;
}

static void mandelbrot_cpu(int* image,
                           int W, int H,
                           int max_iter,
                           float xmin, float xmax,
                           float ymin, float ymax) {
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            float cr = xmin + (xmax - xmin) * (float)px / (float)(W - 1);
            float ci = ymin + (ymax - ymin) * (float)py / (float)(H - 1);
            image[py * W + px] = mandelbrot_iter(cr, ci, max_iter);
        }
    }
}

static void mandelbrot_omp_parallel_for(int* image,
                                        int W, int H,
                                        int max_iter,
                                        float xmin, float xmax,
                                        float ymin, float ymax) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            float cr = xmin + (xmax - xmin) * (float)px / (float)(W - 1);
            float ci = ymin + (ymax - ymin) * (float)py / (float)(H - 1);
            image[py * W + px] = mandelbrot_iter(cr, ci, max_iter);
        }
    }
}

static void write_ppm(const char* path,
                      const int* image,
                      int W, int H,
                      int max_iter) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::perror("fopen");
        return;
    }

    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            int iter = image[py * W + px];
            float t = (float)iter / (float)max_iter;

            int r = (int)(9.0f * (1.0f - t) * t * t * t * 255.0f);
            int g = (int)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);
            int b = (int)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);

            unsigned char rgb[3];
            rgb[0] = (unsigned char)r;
            rgb[1] = (unsigned char)g;
            rgb[2] = (unsigned char)b;
            std::fwrite(rgb, 1, 3, f);
        }
    }
    std::fclose(f);
}

int main(void) {
    const int W = WIDTH;
    const int H = HEIGHT;

    const float xmin = -2.5f;
    const float xmax = 1.0f;
    const float ymin = -1.5f;
    const float ymax = 1.5f;
    const int max_iter = MAX_ITER;

    std::vector<int> ref(W * H, -1);
    double t_cpu_serial0 = omp_get_wtime();
    mandelbrot_cpu(ref.data(), W, H, max_iter, xmin, xmax, ymin, ymax);
    double t_cpu_serial1 = omp_get_wtime();
    double t_cpu_serial = t_cpu_serial1 - t_cpu_serial0;

    std::vector<int> ref_omp(W * H, -1);
    double t_cpu_omp0 = omp_get_wtime();
    mandelbrot_omp_parallel_for(ref_omp.data(), W, H, max_iter, xmin, xmax, ymin, ymax);
    double t_cpu_omp1 = omp_get_wtime();
    double t_cpu_omp = t_cpu_omp1 - t_cpu_omp0;

    int device_count = omp_get_num_devices();
    if (device_count < 1) {
        device_count = 1;
    }

    std::vector<int> image(W * H, -1);
    std::vector<std::vector<int>> device_images((size_t)device_count,
                                                 std::vector<int>(W * H, 0));
    double t_target0 = omp_get_wtime();

    #pragma omp parallel num_threads(device_count)
    {
        int dev = omp_get_thread_num();
        if (dev < device_count) {
            int y_start = dev * H / device_count;
            int y_end = (dev + 1) * H / device_count;
            int* dev_image_ptr = device_images[(size_t)dev].data();

            #pragma omp target data device(dev) map(tofrom: dev_image_ptr[0:W * H])
            {
                #pragma omp target teams distribute parallel for device(dev) \
                    map(present, tofrom: dev_image_ptr[0:W * H]) \
                    firstprivate(W, H, xmin, xmax, ymin, ymax, max_iter, y_start, y_end) \
                    num_teams(512) thread_limit(256)
                for (int idx = 0; idx < (y_end - y_start) * W; ++idx) {
                    int py = y_start + idx / W;
                    int px = idx % W;
                    float cr = xmin + (xmax - xmin) * (float)px / (float)(W - 1);
                    float ci = ymin + (ymax - ymin) * (float)py / (float)(H - 1);

                    float zr = 0.0f;
                    float zi = 0.0f;
                    int iter = 0;
                    while (iter < max_iter) {
                        float zr2 = zr * zr;
                        float zi2 = zi * zi;
                        if (zr2 + zi2 > 4.0f) break;

                        float new_zr = zr2 - zi2 + cr;
                        float new_zi = 2.0f * zr * zi + ci;
                        zr = new_zr;
                        zi = new_zi;
                        ++iter;
                    }

                    dev_image_ptr[py * W + px] = iter;
                }
            }
        }
    }
    double t_target1 = omp_get_wtime();
    double t_target = t_target1 - t_target0;

    for (int dev = 0; dev < device_count; ++dev) {
        int y_start = dev * H / device_count;
        int y_end = (dev + 1) * H / device_count;
        for (int py = y_start; py < y_end; ++py) {
            int row = py * W;
            std::copy(device_images[(size_t)dev].begin() + row,
                      device_images[(size_t)dev].begin() + row + W,
                      image.begin() + row);
        }
    }

    int64_t checksum_image = 0;
    int64_t checksum_ref = 0;
    int max_abs_diff = 0;
    int min_image = INT_MAX, max_image = INT_MIN;
    int min_ref = INT_MAX, max_ref = INT_MIN;

    for (int i = 0; i < W * H; ++i) {
        int iv = image[i];
        int rv = ref[i];

        checksum_image += (int64_t)iv;
        checksum_ref += (int64_t)rv;

        if (iv < min_image) min_image = iv;
        if (iv > max_image) max_image = iv;
        if (rv < min_ref) min_ref = rv;
        if (rv > max_ref) max_ref = rv;

        int diff = iv - rv;
        if (diff < 0) diff = -diff;
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    const int ITER_EPS = MAX_ITER;
    int ok = (max_abs_diff <= ITER_EPS);

    int ok_omp = 1;
    for (int i = 0; i < W * H; ++i) {
        if (ref_omp[i] != ref[i]) { ok_omp = 0; break; }
    }

    write_ppm("mandelbrot.ppm", image.data(), W, H, max_iter);

    printf("[mandelbrot-multidevice-omp-target] W=%d H=%d MAX_ITER=%d\n",
           W, H, MAX_ITER);
    printf("[mandelbrot-multidevice-omp-target] devices=%d\n",
           device_count);
    printf("[mandelbrot-multidevice-omp-target] cpu_serial_time=%.6f s\n", t_cpu_serial);
    printf("[mandelbrot-multidevice-omp-target] cpu_omp_parallel_for_time=%.6f s | %s\n",
           t_cpu_omp, ok_omp ? "PASS" : "FAIL");
    printf("[mandelbrot-multidevice-omp-target] target_time=%.6f s\n", t_target);
    printf("[mandelbrot-multidevice-omp-target] max_abs_diff_iters=%d (ITER_EPS=%d)\n",
           max_abs_diff, ITER_EPS);
    printf("[mandelbrot-multidevice-omp-target] checksum=%lld (ref=%lld) | %s\n",
           (long long)checksum_image, (long long)checksum_ref, ok ? "PASS" : "FAIL");
    if (t_target > 0.0) {
        printf("[mandelbrot-multidevice-omp-target] speedup_serial_vs_target=%.2fx\n",
               t_cpu_serial / t_target);
        printf("[mandelbrot-multidevice-omp-target] speedup_omp_vs_target=%.2fx\n",
               t_cpu_omp / t_target);
    }
    printf("[mandelbrot-multidevice-omp-target] image=mandelbrot.ppm\n");

    return ok ? 0 : 1;
}
