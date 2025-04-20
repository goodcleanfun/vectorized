#ifndef INT32_VECTOR_H
#define INT32_VECTOR_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#define VECTOR_NAME int32_vector
#define VECTOR_TYPE int32_t
#define VECTOR_TYPE_ABS abs
#define VECTOR_TYPE_UNSIGNED uint32_t
#include "vectorized/numeric.h"
#undef VECTOR_NAME
#undef VECTOR_TYPE
#undef VECTOR_TYPE_ABS
#undef VECTOR_TYPE_UNSIGNED

#endif
