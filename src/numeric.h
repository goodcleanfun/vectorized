#ifndef VECTOR_NUMERIC_H
#define VECTOR_NUMERIC_H
/*
vector/numeric.h

To initialize in a header, use in combination with VECTOR_INIT

*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sorting/introsort.h"

#define ks_lt_index(a, b) ((a).value < (b).value)

#endif

#ifndef VECTOR_NAME
#error "Must define VECTOR_NAME"
#endif

#ifndef VECTOR_TYPE
#error "Must define VECTOR_TYPE"
#endif

#ifndef VECTOR_TYPE_UNSIGNED
#define VECTOR_TYPE_UNSIGNED VECTOR_TYPE
#endif

#ifndef VECTOR_TYPE_ABS
#error "Must define VECTOR_TYPE_ABS"
#endif

#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)
#define VECTOR_NAMESPACED(name) CONCAT(VECTOR_NAME, _##name)

static inline void VECTOR_NAMESPACED(zero)(VECTOR_TYPE *array, size_t n) {
    memset(array, 0, n * sizeof(VECTOR_TYPE));
}

static inline void VECTOR_NAMESPACED(copy)(VECTOR_TYPE *dst, const VECTOR_TYPE *src, size_t n) {
    memcpy(dst, src, n * sizeof(VECTOR_TYPE));
}

static inline void VECTOR_NAMESPACED(set)(VECTOR_TYPE *array, VECTOR_TYPE value, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        array[i] = value;
    }
}

static inline VECTOR_TYPE VECTOR_NAMESPACED(max)(VECTOR_TYPE *array, size_t n) {
    if (n < 1) return (VECTOR_TYPE) 0;
    VECTOR_TYPE val = array[0];
    VECTOR_TYPE max_val = val;
    for (size_t i = 1; i < n; i++) {
        val = array[i];
        if (val > max_val) max_val = val;
    }
    return max_val;
}

static inline VECTOR_TYPE VECTOR_NAMESPACED(min)(VECTOR_TYPE *array, size_t n) {
    if (n < 1) return (VECTOR_TYPE) 0;
    VECTOR_TYPE val = array[0];
    VECTOR_TYPE min_val = val;
    for (size_t i = 1; i < n; i++) {
        val = array[i];
        if (val < min_val) min_val = val;
    }
    return min_val;
}

static inline int64_t VECTOR_NAMESPACED(argmax)(VECTOR_TYPE *array, size_t n) {
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

static inline int64_t VECTOR_NAMESPACED(argmin)(VECTOR_TYPE *array, size_t n) {
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

#define VECTOR_INIT_SORT(type) INTROSORT_INIT_GENERIC(type)
#define VECTOR_SORT(type, n, array) ks_introsort(type, n, array)
#define VECTOR_INDEX_TYPE VECTOR_NAMESPACED(index_t)
#define VECTOR_INDEX_TYPE_NAME VECTOR_NAMESPACED(indices)

typedef struct {
    size_t index;
    VECTOR_TYPE value;
} VECTOR_INDEX_TYPE;

VECTOR_INIT_SORT(VECTOR_TYPE)
INTROSORT_INIT(VECTOR_INDEX_TYPE_NAME, VECTOR_INDEX_TYPE, ks_lt_index)

static inline void VECTOR_NAMESPACED(sort)(VECTOR_TYPE *array, size_t n) {
    VECTOR_SORT(VECTOR_TYPE, n, array);
}

static inline size_t *VECTOR_NAMESPACED(argsort)(VECTOR_TYPE *array, size_t n) {
    VECTOR_INDEX_TYPE *type_indices = malloc(sizeof(VECTOR_INDEX_TYPE) * n);
    size_t i;
    for (i = 0; i < n; i++) {
        type_indices[i] = (VECTOR_INDEX_TYPE){i, array[i]};
    }
    ks_introsort(VECTOR_INDEX_TYPE_NAME, n, type_indices);
    size_t *indices = malloc(sizeof(size_t) * n);
    for (i = 0; i < n; i++) {
        indices[i] = type_indices[i].index;
    }
    free(type_indices);
    return indices;
}

#undef VECTOR_INDEX
#undef VECTOR_INDEX_TYPE
#undef VECTOR_INDEX_TYPE_NAME


static inline void VECTOR_NAMESPACED(add)(VECTOR_TYPE *array, VECTOR_TYPE c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        array[i] += c;
    }
}

static inline void VECTOR_NAMESPACED(sub)(VECTOR_TYPE *array, VECTOR_TYPE c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        array[i] -= c;
    }
}

static inline void VECTOR_NAMESPACED(mul)(VECTOR_TYPE *array, VECTOR_TYPE c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        array[i] *= c;
    }
}

static inline void VECTOR_NAMESPACED(div)(VECTOR_TYPE *array, VECTOR_TYPE c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        array[i] /= c;
    }
}

static inline VECTOR_TYPE VECTOR_NAMESPACED(sum)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE result = 0;
    #pragma omp parallel for reduction (+:result)
    for (size_t i = 0; i < n; i++) {
        result += array[i];
    }
    return result;
}

static inline VECTOR_TYPE_UNSIGNED VECTOR_NAMESPACED(l1_norm)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE_UNSIGNED result = 0;
    #pragma omp parallel for reduction (+:result)
    for (size_t i = 0; i < n; i++) {
        result += VECTOR_TYPE_ABS(array[i]);
    }
    return result;
}

static inline double VECTOR_NAMESPACED(l2_norm)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE_UNSIGNED result = 0;
    #pragma omp parallel for reduction (+:result)
    for (size_t i = 0; i < n; i++) {
        result += array[i] * array[i];
    }
    return sqrt((double)result);
}

static inline VECTOR_TYPE_UNSIGNED VECTOR_NAMESPACED(sum_sq)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE_UNSIGNED result = 0;
    #pragma omp parallel for reduction (+:result)
    for (size_t i = 0; i < n; i++) {
        result += array[i] * array[i];
    }
    return result;
}

static inline double VECTOR_NAMESPACED(mean)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE_UNSIGNED sum = VECTOR_NAMESPACED(sum)(array, n);
    return (double)sum / n;
}

static inline double VECTOR_NAMESPACED(var)(VECTOR_TYPE *array, size_t n) {
    double mu = VECTOR_NAMESPACED(mean)(array, n);
    double sigma2 = 0.0;
    #pragma omp parallel for reduction (+:sigma2)
    for (size_t i = 0; i < n; i++) {
        double dev = (double)array[i] - mu;
        sigma2 += dev * dev;
    }
    return sigma2 / n;
}

static inline double VECTOR_NAMESPACED(std)(VECTOR_TYPE *array, size_t n) {
    double sigma2 = VECTOR_NAMESPACED(var)(array, n);
    return sqrt(sigma2);
}

static inline VECTOR_TYPE VECTOR_NAMESPACED(product)(VECTOR_TYPE *array, size_t n) {
    VECTOR_TYPE result = 0;
    #pragma omp parallel for reduction (+:result)
    for (size_t i = 0; i < n; i++) {
        result *= array[i];
    }
    return result;
}

static inline void VECTOR_NAMESPACED(add_array)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a1[i] += a2[i];
    }
}

static inline void VECTOR_NAMESPACED(add_array_scaled)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, double v, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a1[i] += a2[i] * v;
    }
}

static inline void VECTOR_NAMESPACED(sub_array)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a1[i] -= a2[i];
    }
}


static inline void VECTOR_NAMESPACED(sub_array_scaled)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, double v, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a1[i] -= a2[i] * v;
    }
}

static inline void VECTOR_NAMESPACED(mul_array)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a1[i] *= a2[i];
    }
}

static inline void VECTOR_NAMESPACED(mul_array_scaled)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, double v, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a1[i] *= a2[i] * v;
    }
}

static inline void VECTOR_NAMESPACED(div_array)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a1[i] /= a2[i];
    }
}

static inline void VECTOR_NAMESPACED(div_array_scaled)(VECTOR_TYPE *a1, const VECTOR_TYPE *a2, double v, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a1[i] /= a2[i] * v;
    }
}

static inline VECTOR_TYPE VECTOR_NAMESPACED(dot)(const VECTOR_TYPE *a1, const VECTOR_TYPE *a2, size_t n) {
    VECTOR_TYPE result = 0;
    #pragma omp parallel for reduction (+:result)
    for (size_t i = 0; i < n; i++) {
        result += a1[i] * a2[i];
    }
    return result;
}

#undef CONCAT_
#undef CONCAT
#undef VECTOR_NAMESPACED
