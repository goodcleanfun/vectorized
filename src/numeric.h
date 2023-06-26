#ifndef VECTOR_NUMERIC_H
#define VECTOR_NUMERIC_H
/*
vector/numeric.h

Because ksort has functions that are not inline, even though our functions are,
this macro must be used only in source files, not in headers, otherwise
it can cause duplicate symbol issues.

To initialize, in a header it's ok to use

float_array.h
=============
VECTOR_INIT(float_array, float)

float_array.c
=============
VECTOR_NUMERIC(float_array, float, float, fabsf)

*/

#include "ksort/ksort.h"

#define ks_lt_index(a, b) ((a).value < (b).value)

#define VECTOR_NUMERIC(name, type, unsigned_type, type_abs)                                             \
    static inline void name##_zero(type *array, size_t n) {                                             \
        memset(array, 0, n * sizeof(type));                                                             \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_raw_copy(type *dst, const type *src, size_t n) {                          \
        memcpy(dst, src, n * sizeof(type));                                                             \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_set(type *array, type value, size_t n) {                                  \
        for (size_t i = 0; i < n; i++) {                                                                \
            array[i] = value;                                                                           \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline name *name##_new_value(size_t n, type value) {                                        \
        name *vector = name##_new_size(n);                                                              \
        if (vector == NULL) return NULL;                                                                \
        name##_set(vector->a, n, (type)value);                                                          \
        vector->n = n;                                                                                  \
        return vector;                                                                                  \
    }                                                                                                   \
                                                                                                        \
    static inline name *name##_new_ones(size_t n) {                                                     \
        return name##_new_value(n, (type)1);                                                            \
    }                                                                                                   \
                                                                                                        \
    static inline name *name##_new_zeros(size_t n) {                                                    \
        name *vector = name##_new_size(n);                                                              \
        if (vector == NULL) return NULL;                                                                \
        name##_zero(vector->a, n);                                                                      \
        vector->n = n;                                                                                  \
        return vector;                                                                                  \
    }                                                                                                   \
                                                                                                        \
    static inline bool name##_resize_fill_zeros(name *self, size_t n) {                             \
        size_t old_n = self->n;                                                                         \
        bool ret = name##_resize(self, n);                                                              \
        if (ret && n > old_n) {                                                                         \
            memset(self->a + old_n, 0, (n - old_n) * sizeof(type));                                \
        }                                                                                               \
        return ret;                                                                                     \
    }                                                                                                   \
                                                                                                        \
    static inline bool name##_resize_aligned_fill_zeros(name *self, size_t n, size_t alignment) {   \
        size_t old_n = self->n;                                                                         \
        bool ret = name##_resize_aligned(self, n, alignment);                                           \
        if (ret && n > old_n) {                                                                         \
            memset(self->a + old_n, 0, (n - old_n) * sizeof(type));                                \
        }                                                                                               \
        return ret;                                                                                     \
    }                                                                                                   \
                                                                                                        \
    static inline type name##_max(type *array, size_t n) {                                              \
        if (n < 1) return (type) 0;                                                                     \
        type val = array[0];                                                                            \
        type max_val = val;                                                                             \
        for (size_t i = 1; i < n; i++) {                                                                \
            val = array[i];                                                                             \
            if (val > max_val) max_val = val;                                                           \
        }                                                                                               \
        return max_val;                                                                                 \
    }                                                                                                   \
                                                                                                        \
    static inline type name##_min(type *array, size_t n) {                                              \
        if (n < 1) return (type) 0;                                                                     \
        type val = array[0];                                                                            \
        type min_val = val;                                                                             \
        for (size_t i = 1; i < n; i++) {                                                                \
            val = array[i];                                                                             \
            if (val < min_val) min_val = val;                                                           \
        }                                                                                               \
        return min_val;                                                                                 \
    }                                                                                                   \
                                                                                                        \
    static inline int64_t name##_argmax(type *array, size_t n) {                                        \
        if (n < 1) return -1;                                                                           \
        type val = array[0];                                                                            \
        type max_val = val;                                                                             \
        int64_t argmax = 0;                                                                             \
        for (size_t i = 0; i < n; i++) {                                                                \
            val = array[i];                                                                             \
            if (val > max_val) {                                                                        \
                max_val = val;                                                                          \
                argmax = i;                                                                             \
            }                                                                                           \
        }                                                                                               \
        return argmax;                                                                                  \
    }                                                                                                   \
                                                                                                        \
    static inline int64_t name##_argmin(type *array, size_t n) {                                        \
        if (n < 1) return (type) -1;                                                                    \
        type val = array[0];                                                                            \
        type min_val = val;                                                                             \
        int64_t argmin = 0;                                                                             \
        for (size_t i = 1; i < n; i++) {                                                                \
            val = array[i];                                                                             \
            if (val < min_val) {                                                                        \
                min_val = val;                                                                          \
                argmin = i;                                                                             \
            }                                                                                           \
        }                                                                                               \
        return argmin;                                                                                  \
    }                                                                                                   \
                                                                                                        \
    typedef struct type##_index {                                                                       \
        size_t index;                                                                                   \
        type value;                                                                                     \
    } type##_index_t;                                                                                   \
                                                                                                        \
    KSORT_INIT_GENERIC(type)                                                                            \
    KSORT_INIT(type##_indices, type##_index_t, ks_lt_index)                                             \
                                                                                                        \
    static inline void name##_sort(type *array, size_t n) {                                             \
        ks_introsort(type, n, array);                                                                   \
    }                                                                                                   \
                                                                                                        \
    static inline size_t *name##_argsort(type *array, size_t n) {                                       \
        type##_index_t *type_indices = malloc(sizeof(type##_index_t) * n);                              \
        size_t i;                                                                                       \
        for (i = 0; i < n; i++) {                                                                       \
            type_indices[i] = (type##_index_t){i, array[i]};                                            \
        }                                                                                               \
        ks_introsort(type##_indices, n, type_indices);                                                  \
        size_t *indices = malloc(sizeof(size_t) * n);                                                   \
        for (i = 0; i < n; i++) {                                                                       \
            indices[i] = type_indices[i].index;                                                         \
        }                                                                                               \
        free(type_indices);                                                                             \
        return indices;                                                                                 \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_add(type *array, type c, size_t n) {                                      \
        for (size_t i = 0; i < n; i++) {                                                                \
            array[i] += c;                                                                              \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_sub(type *array, type c, size_t n) {                                      \
        for (size_t i = 0; i < n; i++) {                                                                \
            array[i] -= c;                                                                              \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_mul(type *array, type c, size_t n) {                                      \
        for (size_t i = 0; i < n; i++) {                                                                \
            array[i] *= c;                                                                              \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_div(type *array, type c, size_t n) {                                      \
        for (size_t i = 0; i < n; i++) {                                                                \
            array[i] /= c;                                                                              \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline type name##_sum(type *array, size_t n) {                                              \
        type result = 0;                                                                                \
        for (size_t i = 0; i < n; i++) {                                                                \
            result += array[i];                                                                         \
        }                                                                                               \
        return result;                                                                                  \
    }                                                                                                   \
                                                                                                        \
    static inline unsigned_type name##_l1_norm(type *array, size_t n) {                                 \
        unsigned_type result = 0;                                                                       \
        for (size_t i = 0; i < n; i++) {                                                                \
            result += type_abs(array[i]);                                                               \
        }                                                                                               \
        return result;                                                                                  \
    }                                                                                                   \
                                                                                                        \
    static inline double name##_l2_norm(type *array, size_t n) {                                        \
        unsigned_type result = 0;                                                                       \
        for (size_t i = 0; i < n; i++) {                                                                \
            result += array[i] * array[i];                                                              \
        }                                                                                               \
        return sqrt((double)result);                                                                    \
    }                                                                                                   \
                                                                                                        \
    static inline unsigned_type name##_sum_sq(type *array, size_t n) {                                  \
        unsigned_type result = 0;                                                                       \
        for (size_t i = 0; i < n; i++) {                                                                \
            result += array[i] * array[i];                                                              \
        }                                                                                               \
        return result;                                                                                  \
    }                                                                                                   \
                                                                                                        \
    static inline double name##_mean(type *array, size_t n) {                                           \
        unsigned_type sum = name##_sum(array, n);                                                       \
        return (double)sum / n;                                                                         \
    }                                                                                                   \
                                                                                                        \
    static inline double name##_var(type *array, size_t n) {                                            \
        double mu = name##_mean(array, n);                                                              \
        double sigma2 = 0.0;                                                                            \
        for (size_t i = 0; i < n; i++) {                                                                \
            double dev = (double)array[i] - mu;                                                         \
            sigma2 += dev * dev;                                                                        \
        }                                                                                               \
        return sigma2 / n;                                                                              \
    }                                                                                                   \
                                                                                                        \
    static inline double name##_std(type *array, size_t n) {                                            \
        double sigma2 = name##_var(array, n);                                                           \
        return sqrt(sigma2);                                                                            \
    }                                                                                                   \
                                                                                                        \
    static inline type name##_product(type *array, size_t n) {                                          \
        type result = 0;                                                                                \
        for (size_t i = 0; i < n; i++) {                                                                \
            result *= array[i];                                                                         \
        }                                                                                               \
        return result;                                                                                  \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_add_array(type *a1, const type *a2, size_t n) {                           \
        for (size_t i = 0; i < n; i++) {                                                                \
            a1[i] += a2[i];                                                                             \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_add_array_times_scalar(type *a1, const type *a2, double v, size_t n) {    \
        for (size_t i = 0; i < n; i++) {                                                                \
            a1[i] += a2[i] * v;                                                                         \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_sub_array(type *a1, const type *a2, size_t n) {                           \
        for (size_t i = 0; i < n; i++) {                                                                \
            a1[i] -= a2[i];                                                                             \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
                                                                                                        \
    static inline void name##_sub_array_times_scalar(type *a1, const type *a2, double v, size_t n) {    \
        for (size_t i = 0; i < n; i++) {                                                                \
            a1[i] -= a2[i] * v;                                                                         \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_mul_array(type *a1, const type *a2, size_t n) {                           \
        for (size_t i = 0; i < n; i++) {                                                                \
            a1[i] *= a2[i];                                                                             \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_mul_array_times_scalar(type *a1, const type *a2, double v, size_t n) {    \
        for (size_t i = 0; i < n; i++) {                                                                \
            a1[i] *= a2[i] * v;                                                                         \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_div_array(type *a1, const type *a2, size_t n) {                           \
        for (size_t i = 0; i < n; i++) {                                                                \
            a1[i] /= a2[i];                                                                             \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline void name##_div_array_times_scalar(type *a1, const type *a2, double v, size_t n) {    \
        for (size_t i = 0; i < n; i++) {                                                                \
            a1[i] /= a2[i] * v;                                                                         \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    static inline type name##_dot(const type *a1, const type *a2, size_t n) {                           \
        type result = 0;                                                                                \
        for (size_t i = 0; i < n; i++) {                                                                \
            result += a1[i] * a2[i];                                                                    \
        }                                                                                               \
        return result;                                                                                  \
    }

#endif