#ifndef UINT64_VECTOR_H
#define UINT64_VECTOR_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "vectorized/nop.h"

#define VECTOR_NAME uint64_vector
#define VECTOR_TYPE uint64_t
#define VECTOR_TYPE_ABS nop
#include "vectorized/numeric.h"
#undef VECTOR_NAME
#undef VECTOR_TYPE
#undef VECTOR_TYPE_ABS

#endif
