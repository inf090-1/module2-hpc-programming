#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <climits>

// Mandelbrot using OpenMP target offload across multiple devices.
//
// TODO: Implement multi-device offload.
// Requirements:
// - No OpenMP tasks.
// - Split the image among available devices (e.g., vertical strips
//   or round-robin quadtree leaves).
// - Each device runs a single target region for its assigned pixels.
// - Produce a final image and write mandelbrot.ppm.

#ifndef WIDTH
#define WIDTH 4096
#endif
#ifndef HEIGHT
#define HEIGHT 4096
#endif
#ifndef MAX_ITER
#define MAX_ITER 5000
#endif
#ifndef LEAF_SIZE
#define LEAF_SIZE 32
#endif

struct Rect {
    int x0;
    int y0;
    int w;
    int h;
};

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

static void collect_quadtree_leaves(std::vector<Rect>& leaves,
                                    int x0, int y0,
                                    int w, int h) {
    if (w <= LEAF_SIZE && h <= LEAF_SIZE) {
        Rect r;
        r.x0 = x0;
        r.y0 = y0;
        r.w = w;
        r.h = h;
        leaves.push_back(r);
        return;
    }

    int w2 = w / 2;
    int h2 = h / 2;

    int w_left = w2;
    int w_right = w - w2;
    int h_top = h2;
    int h_bottom = h - h2;

    if (w_left > 0 && h_top > 0)
        collect_quadtree_leaves(leaves, x0, y0, w_left, h_top);
    if (w_right > 0 && h_top > 0)
        collect_quadtree_leaves(leaves, x0 + w_left, y0, w_right, h_top);
    if (w_left > 0 && h_bottom > 0)
        collect_quadtree_leaves(leaves, x0, y0 + h_top, w_left, h_bottom);
    if (w_right > 0 && h_bottom > 0)
        collect_quadtree_leaves(leaves, x0 + w_left, y0 + h_top, w_right, h_bottom);
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

    // Baseline 1: CPU serial.
    std::vector<int> ref(W * H, -1);
    double t_cpu_serial0 = omp_get_wtime();
    mandelbrot_cpu(ref.data(), W, H, MAX_ITER, xmin, xmax, ymin, ymax);
    double t_cpu_serial1 = omp_get_wtime();
    double t_cpu_serial = t_cpu_serial1 - t_cpu_serial0;

    // Baseline 2: CPU OpenMP parallel-for.
    std::vector<int> ref_omp(W * H, -1);
    double t_cpu_omp0 = omp_get_wtime();
    mandelbrot_omp_parallel_for(ref_omp.data(), W, H, MAX_ITER, xmin, xmax, ymin, ymax);
    double t_cpu_omp1 = omp_get_wtime();
    double t_cpu_omp = t_cpu_omp1 - t_cpu_omp0;

    // Build quadtree leaves on host.
    std::vector<Rect> leaves;
    leaves.reserve((W * H) / (LEAF_SIZE * LEAF_SIZE) + 4);
    collect_quadtree_leaves(leaves, 0, 0, W, H);

    int device_count = omp_get_num_devices();
    if (device_count < 1) {
        device_count = 1;
    }

    std::vector<std::vector<Rect>> leaves_per_device((size_t)device_count);
    for (size_t i = 0; i < leaves.size(); ++i) {
        int dev = (int)(i % (size_t)device_count);
        leaves_per_device[(size_t)dev].push_back(leaves[i]);
    }

    // Multi-device OpenMP target.
    std::vector<int> image(W * H, -1);
    double t_target0 = omp_get_wtime();

    // TODO: Implement multi-device OpenMP target offload.

    // Stub (incorrect): fill with zeros so validation fails until implemented.
    for (int i = 0; i < W * H; ++i) {
        image[i] = 0;
    }

    double t_target1 = omp_get_wtime();
    double t_target = t_target1 - t_target0;

    // Validate.
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

    write_ppm("mandelbrot.ppm", image.data(), W, H, MAX_ITER);

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
