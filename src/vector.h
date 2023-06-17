#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "ksort/ksort.h"

#define DEFAULT_VECTOR_SIZE 8

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
#else
#include <stdlib.h>
static inline void *_aligned_malloc(size_t size, size_t alignment)
{
    void *p;
    int ret = posix_memalign(&p, alignment, size);
    return (ret == 0) ? p : NULL;
}
static inline void *_aligned_realloc(void *p, size_t size, size_t alignment)
{
    if ((alignment == 0) || ((alignment & (alignment - 1)) != 0) || (alignment < sizeof(void *))) {
        return NULL;
    }

    if (size == 0) {
        return NULL;
    }

    void *rp = realloc(p, size);

    /* If realloc result is not already at an aligned boundary,
       _aligned_malloc a new block and copy the contents of the realloc'd
       pointer to the aligned block, free the realloc'd pointer and return
       the aligned pointer.
    */
    if ( ((size_t)rp & (alignment - 1)) != 0) {
        void *p1 = _aligned_malloc(size, alignment);
        if (p1 != NULL) {
            memcpy(p1, rp, size);
        }
        free(rp);
        rp = p1;
    }

    return rp;
}
static inline void _aligned_free(void *p)
{
    free(p);
}
#endif

#ifdef _MSC_VER
#define MIE_ALIGN(x) __declspec(align(x))
#else
#define MIE_ALIGN(x) __attribute__((aligned(x)))
#endif

// Based on kvec.h, dynamic vectors of any type
#define __VECTOR_BASE(name, type) typedef struct { size_t n, m; type *a; } name;            \
    static inline name *name##_new_size(size_t size) {                                      \
        name *array = malloc(sizeof(name));                                                 \
        if (array == NULL) return NULL;                                                     \
        array->n = array->m = 0;                                                            \
        array->a = malloc((size > 0 ? size : 1) * sizeof(type));                            \
        if (array->a == NULL) return NULL;                                                  \
        array->m = size;                                                                    \
        return array;                                                                       \
    }                                                                                       \
    static inline name *name##_new(void) {                                                  \
        return name##_new_size(DEFAULT_VECTOR_SIZE);                                        \
    }                                                                                       \
    static inline name *name##_new_size_fixed(size_t size) {                                \
        name *array = name##_new_size(size);                                                \
        if (array == NULL) return NULL;                                                     \
        array->n = size;                                                                    \
        return array;                                                                       \
    }                                                                                       \
    static inline name *name##_new_aligned(size_t size, size_t alignment) {                 \
        name *array = malloc(sizeof(name));                                                 \
        if (array == NULL) return NULL;                                                     \
        array->n = array->m = 0;                                                            \
        array->a = _aligned_malloc(size * sizeof(type), alignment);                         \
        if (array->a == NULL) return NULL;                                                  \
        array->m = size;                                                                    \
        return array;                                                                       \
    }                                                                                       \
    static inline bool name##_resize(name *array, size_t size) {                            \
        if (size <= array->m)return true;                                                   \
        type *ptr = realloc(array->a, sizeof(type) * size);                                 \
        if (ptr == NULL) return false;                                                      \
        array->a = ptr;                                                                     \
        array->m = size;                                                                    \
        return true;                                                                        \
    }                                                                                       \
    static inline bool name##_resize_aligned(name *array, size_t size, size_t alignment) {  \
        if (size <= array->m) return true;                                                  \
        type *ptr = _aligned_realloc(array->a, sizeof(type) * size, alignment);             \
        if (ptr == NULL) return false;                                                      \
        array->a = ptr;                                                                     \
        array->m = size;                                                                    \
        return true;                                                                        \
    }                                                                                       \
    static inline bool name##_resize_fixed(name *array, size_t size) {                      \
        if (!name##_resize(array, size)) return false;                                      \
        array->n = size;                                                                    \
        return true;                                                                        \
    }                                                                                       \
    static inline bool name##_resize_fixed_aligned(name *array, size_t size, size_t alignment) {  \
        if (!name##_resize_aligned(array, size, alignment)) return false;                   \
        array->n = size;                                                                    \
        return true;                                                                        \
    }                                                                                       \
    static inline void name##_push(name *array, type value) {                               \
        if (array->n == array->m) {                                                         \
            size_t size = array->m ? array->m << 1 : 2;                                     \
            type *ptr = realloc(array->a, sizeof(type) * size);                             \
            if (ptr == NULL) {                                                              \
                fprintf(stderr, "realloc failed during " #name "_push\n");                  \
                exit(EXIT_FAILURE);                                                         \
            }                                                                               \
            array->a = ptr;                                                                 \
            array->m = size;                                                                \
        }                                                                                   \
        array->a[array->n++] = value;                                                       \
    }                                                                                       \
    static inline bool name##_extend(name *array, name *other) {                            \
        bool ret = false;                                                                   \
        size_t new_size = array->n + other->n;                                              \
        if (new_size > array->m) ret = name##_resize(array, new_size);                      \
        if (!ret) return false;                                                             \
        memcpy(array->a + array->n, other->a, other->n * sizeof(type));                     \
        array->n = new_size;                                                                \
        return ret;                                                                         \
    }                                                                                       \
    static inline void name##_pop(name *array) {                                            \
        if (array->n > 0) array->n--;                                                       \
    }                                                                                       \
    static inline void name##_clear(name *array) {                                          \
        array->n = 0;                                                                       \
    }                                                                                       \
    static inline bool name##_copy(name *dst, name *src, size_t n) {                        \
        bool ret = true;                                                                    \
        if (dst->m < n) ret = name##_resize(dst, n);                                        \
        if (!ret) return false;                                                             \
        memcpy(dst->a, src->a, n * sizeof(type));                                           \
        dst->n = n;                                                                         \
        return ret;                                                                         \
    }                                                                                       \
    static inline name *name##_new_copy(name *vector, size_t n) {                           \
        name *cpy = name##_new_size(n);                                                     \
        if (!name##_copy(cpy, vector, n)) return NULL;                                      \
        return cpy;                                                                         \
    }

#define __VECTOR_DESTROY(name, type)                                    \
    static inline void name##_destroy(name *array) {                    \
        if (array == NULL) return;                                      \
        if (array->a != NULL) free(array->a);                           \
        free(array);                                                    \
    }                                                                   \
    static inline void name##_destroy_aligned(name *array) {            \
        if (array == NULL) return;                                      \
        if (array->a != NULL) _aligned_free(array->a);                  \
        free(array);                                                    \
    }

#define __VECTOR_DESTROY_FREE_DATA(name, type, free_func)               \
    static inline void name##_destroy(name *array) {                    \
        if (array == NULL) return;                                      \
        if (array->a != NULL) {                                         \
            for (size_t i = 0; i < array->n; i++) {                     \
                free_func(array->a[i]);                                 \
            }                                                           \
        }                                                               \
        free(array->a);                                                 \
        free(array);                                                    \
    }                                                                   \
    static inline void name##_destroy_aligned(name *array) {            \
        if (array == NULL) return;                                      \
        if (array->a != NULL) {                                         \
            for (size_t i = 0; i < array->n; i++) {                     \
                free_func(array->a[i]);                                 \
            }                                                           \
        }                                                               \
        _aligned_free(array->a);                                        \
        free(array);                                                    \
    }

#define VECTOR_INIT(name, type)                                         \
    __VECTOR_BASE(name, type)                                           \
    __VECTOR_DESTROY(name, type)                                      


#define VECTOR_INIT_NUMERIC(name, type, unsigned_type, type_abs)                                        \
    __VECTOR_BASE(name, type)                                                                           \
    __VECTOR_DESTROY(name, type)                                                                        \
                                                                                                        \
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


#define VECTOR_INIT_FREE_DATA(name, type, free_func)                    \
    __VECTOR_BASE(name, type)                                           \
    __VECTOR_DESTROY_FREE_DATA(name, type, free_func)                 

#endif
