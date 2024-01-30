#ifndef VECTOR_NUMERIC_H
#define VECTOR_NUMERIC_H
/*
vector/numeric.h

To initialize in a header, use in combination with VECTOR_INIT

*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#endif

#ifndef VECTOR_NAME
#error "Must define VECTOR_NAME"
#endif

#ifndef VECTOR_TYPE
#error "Must define VECTOR_TYPE"
#endif

#ifndef VECTOR_TYPE_UNSIGNED
#define VECTOR_TYPE_UNSIGNED_DEFINED
#define VECTOR_TYPE_UNSIGNED VECTOR_TYPE
#endif

#ifndef OMP_PARALLEL_MIN_SIZE
#define OMP_PARALLEL_MIN_SIZE_DEFINED
#define OMP_PARALLEL_MIN_SIZE 1000
#endif


#define VECTOR_CONCAT_(a, b) a ## b
#define VECTOR_CONCAT(a, b) VECTOR_CONCAT_(a, b)
#define VECTOR_FUNC(name) VECTOR_CONCAT(VECTOR_NAME, _##name)

static inline void VECTOR_FUNC(zero)(VECTOR_TYPE *array, size_t n) {
    memset(array, 0, n * sizeof(VECTOR_TYPE));
}

static inline void VECTOR_FUNC(copy)(VECTOR_TYPE *dst, const VECTOR_TYPE *src, size_t n) {
    memcpy(dst, src, n * sizeof(VECTOR_TYPE));
}

static inline void VECTOR_FUNC(set)(VECTOR_TYPE *array, VECTOR_TYPE value, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        array[i] = value;
    }
}

static inline VECTOR_TYPE VECTOR_FUNC(max)(VECTOR_TYPE *array, size_t n) {
    if (n < 1) return (VECTOR_TYPE) 0;
    VECTOR_TYPE val = array[0];
    VECTOR_TYPE max_val = val;
    for (size_t i = 1; i < n; i++) {
        val = array[i];
        if (val > max_val) max_val = val;
    }
    return max_val;
}

static inline VECTOR_TYPE VECTOR_FUNC(min)(VECTOR_TYPE *array, size_t n) {
    if (n < 1) return (VECTOR_TYPE) 0;
    VECTOR_TYPE val = array[0];
    VECTOR_TYPE min_val = val;
    for (size_t i = 1; i < n; i++) {
        val = array[i];
        if (val < min_val) min_val = val;
    }
    return min_val;
}

static inline int64_t VECTOR_FUNC(argmax)(VECTOR_TYPE *array, size_t n) {
    if (n < 1) return -1;
    VECTOR_TYPE val = array[0];
    VECTOR_TYPE max_val = val;
    int64_t argmax = 0;
    for (size_t i = 0; i < n; i++) {
        val = array[i];
        if (val > max_val) {
            max_val = val;
            argmax = i;
        }
    }
    return argmax;
}

static inline int64_t VECTOR_FUNC(argmin)(VECTOR_TYPE *array, size_t n) {
    if (n < 1) return (VECTOR_TYPE) -1;
    VECTOR_TYPE val = array[0];
    VECTOR_TYPE min_val = val;
    int64_t argmin = 0;
    for (size_t i = 1; i < n; i++) {
        val = array[i];
        if (val < min_val) {
            min_val = val;
            argmin = i;
        }
    }
    return argmin;
}

#define VECTOR_INDEX_NAME VECTOR_CONCAT(VECTOR_NAME, _index)
#define VECTOR_SORT_FUNC(name) VECTOR_CONCAT(name##_, VECTOR_NAME)
#define VECTOR_SORT_INDEX_FUNC(name) VECTOR_CONCAT(name##_, VECTOR_INDEX_NAME)

#define INTROSORT_NAME VECTOR_NAME
#define INTROSORT_TYPE VECTOR_TYPE
#include "sorting/introsort.h"
#undef INTROSORT_TYPE
#undef INTROSORT_NAME

static inline void VECTOR_FUNC(sort)(VECTOR_TYPE *array, size_t n) {
    VECTOR_SORT_FUNC(introsort)(n, array);
}

#define INTROSORT_NAME VECTOR_INDEX_NAME
#define INTROSORT_TYPE size_t
#define INTROSORT_AUX_TYPE VECTOR_TYPE
#include "sorting/introsort.h"
#undef INTROSORT_NAME
#undef INTROSORT_TYPE
#undef INTROSORT_AUX_TYPE

static inline bool VECTOR_FUNC(argsort)(VECTOR_TYPE *array, size_t n, size_t *indices) {
    if (indices == NULL) return false;
    for (size_t i = 0; i < n; i++) {
        indices[i] = i;
    }
    VECTOR_SORT_INDEX_FUNC(introsort)(n, indices, array);
    return true;
}

#undef VECTOR_INDEX_NAME
#undef VECTOR_SORT_FUNC
#undef VECTOR_SORT_INDEX_FUNC

static inline void VECTOR_FUNC(add)(VECTOR_TYPE *array, VECTOR_TYPE c, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        array[i] += c;
    }
}

static inline void VECTOR_FUNC(sub)(VECTOR_TYPE *array, VECTOR_TYPE c, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        array[i] -= c;
    }
}

static inline void VECTOR_FUNC(mul)(VECTOR_TYPE *array, VECTOR_TYPE c, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        array[i] *= c;
    }
}

static inline void VECTOR_FUNC(div)(VECTOR_TYPE *array, VECTOR_TYPE c, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        array[i] /= c;
    }
}

static inline VECTOR_TYPE VECTOR_FUNC(sum)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE result = 0;
    #pragma omp parallel for reduction (+:result) if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        result += array[i];
    }
    return result;
}

static inline VECTOR_TYPE_UNSIGNED VECTOR_FUNC(l1_norm)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE_UNSIGNED result = 0;
    #pragma omp parallel for reduction (+:result) if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        #ifdef VECTOR_TYPE_ABS
        result += VECTOR_TYPE_ABS(array[i]);
        #else
        result += abs(array[i]);
        #endif
    }
    return result;
}

static inline double VECTOR_FUNC(l2_norm)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE_UNSIGNED result = 0;
    #pragma omp parallel for reduction (+:result) if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        result += array[i] * array[i];
    }
    return sqrt((double)result);
}

static inline VECTOR_TYPE_UNSIGNED VECTOR_FUNC(sum_sq)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE_UNSIGNED result = 0;
    #pragma omp parallel for reduction (+:result) if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        result += array[i] * array[i];
    }
    return result;
}

static inline double VECTOR_FUNC(mean)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE_UNSIGNED sum = VECTOR_FUNC(sum)(array, n);
    return (double)sum / n;
}

static inline double VECTOR_FUNC(var)(VECTOR_TYPE *array, size_t n) {
    double mu = VECTOR_FUNC(mean)(array, n);
    double sigma2 = 0.0;
    #pragma omp parallel for reduction (+:sigma2) if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        double dev = (double)array[i] - mu;
        sigma2 += dev * dev;
    }
    return sigma2 / n;
}

static inline double VECTOR_FUNC(std)(VECTOR_TYPE *array, size_t n) {
    double sigma2 = VECTOR_FUNC(var)(array, n);
    return sqrt(sigma2);
}

static inline VECTOR_TYPE VECTOR_FUNC(product)(VECTOR_TYPE *array, size_t n) {
    if (n < 1) return (VECTOR_TYPE) 0;
    VECTOR_TYPE result = array[0];
    #pragma omp parallel for reduction (*:result) if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 1; i < n; i++) {
        result *= array[i];
    }
    return result;
}

static inline void VECTOR_FUNC(add_vector)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        a1[i] += a2[i];
    }
}

static inline void VECTOR_FUNC(add_vector_scaled)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, double v, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        a1[i] += a2[i] * v;
    }
}

static inline void VECTOR_FUNC(sub_vector)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        a1[i] -= a2[i];
    }
}


static inline void VECTOR_FUNC(sub_vector_scaled)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, double v, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        a1[i] -= a2[i] * v;
    }
}

static inline void VECTOR_FUNC(mul_vector)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        a1[i] *= a2[i];
    }
}

static inline void VECTOR_FUNC(mul_vector_scaled)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, double v, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        a1[i] *= a2[i] * v;
    }
}

static inline void VECTOR_FUNC(div_vector)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        a1[i] /= a2[i];
    }
}

static inline void VECTOR_FUNC(div_vector_scaled)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, double v, size_t n) {
    #pragma omp parallel for if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        a1[i] /= a2[i] * v;
    }
}

static inline VECTOR_TYPE VECTOR_FUNC(dot)(const VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    VECTOR_TYPE result = 0;
    #pragma omp parallel for reduction (+:result) if (n > OMP_PARALLEL_MIN_SIZE)
    for (size_t i = 0; i < n; i++) {
        result += a1[i] * a2[i];
    }
    return result;
}

#undef VECTOR_FUNC
#undef VECTOR_CONCAT_
#undef VECTOR_CONCAT

#ifdef VECTOR_TYPE_UNSIGNED_DEFINED
#undef VECTOR_TYPE_UNSIGNED
#undef VECTOR_TYPE_UNSIGNED_DEFINED
#endif

#ifdef OMP_PARALLEL_MIN_SIZE_DEFINED
#undef OMP_PARALLEL_MIN_SIZE
#undef OMP_PARALLEL_MIN_SIZE_DEFINED
#endif