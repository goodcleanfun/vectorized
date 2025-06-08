#ifndef INT64_VECTOR_H
#define INT64_VECTOR_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#define VECTOR_NAME int64_vector
#define VECTOR_TYPE int64_t
#define VECTOR_TYPE_ABS llabs
#define VECTOR_TYPE_UNSIGNED uint64_t
#include "numeric.h"
#undef VECTOR_NAME
#undef VECTOR_TYPE
#undef VECTOR_TYPE_ABS
#undef VECTOR_TYPE_UNSIGNED

#endif
