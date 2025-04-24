#ifndef FLOAT_VECTOR_H
#define FLOAT_VECTOR_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>

#define VECTOR_NAME float_vector
#define VECTOR_TYPE float
#define VECTOR_TYPE_ABS fabs
#include "numeric.h"
#include "simd_math.h"

#ifndef OMP_PARALLEL_MIN_SIZE
#define OMP_PARALLEL_MIN_SIZE_DEFINED
#define OMP_PARALLEL_MIN_SIZE 1000
#endif

static inline void float_vector_log(float *x, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n - (n % 8); i += 8) {
        simde__m256 x_vec = simde_mm256_loadu_ps(&x[i]);
        simde__m256 y_vec = log256_ps(x_vec);
        simde_mm256_storeu_ps(&x[i], y_vec);
    }
    for (size_t i = n - (n % 8); i < n; i++) {
        x[i] = logf(x[i]);
    }
}

static inline void float_vector_exp(float *x, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n - (n % 8); i += 8) {
        simde__m256 x_vec = simde_mm256_loadu_ps(&x[i]);
        simde__m256 y_vec = exp256_ps(x_vec);
        simde_mm256_storeu_ps(&x[i], y_vec);
    }
    for (size_t i = n - (n % 8); i < n; i++) {
        x[i] = expf(x[i]);
    }
}

static inline void float_vector_sin(float *x, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n - (n % 8); i += 8) {
        simde__m256 x_vec = simde_mm256_loadu_ps(&x[i]);
        simde__m256 y_vec = sin256_ps(x_vec);
        simde_mm256_storeu_ps(&x[i], y_vec);
    }
    for (size_t i = n - (n % 8); i < n; i++) {
        x[i] = sinf(x[i]);
    }
}

static inline void float_vector_cos(float *x, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n - (n % 8); i += 8) {
        simde__m256 x_vec = simde_mm256_loadu_ps(&x[i]);
        simde__m256 y_vec = cos256_ps(x_vec);
        simde_mm256_storeu_ps(&x[i], y_vec);
    }
    for (size_t i = n - (n % 8); i < n; i++) {
        x[i] = cosf(x[i]);
    }
}

#undef VECTOR_NAME
#undef VECTOR_TYPE
#undef VECTOR_TYPE_ABS
#ifdef OMP_PARALLEL_MIN_SIZE_DEFINED
#undef OMP_PARALLEL_MIN_SIZE_DEFINED
#undef OMP_PARALLEL_MIN_SIZE
#endif

#endif
