#ifndef FLOAT_VECTOR_H
#define FLOAT_VECTOR_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>

#define VECTOR_NAME float_vector
#define VECTOR_TYPE float
#define VECTOR_TYPE_ABS fabs
#include "vectorized/numeric.h"
#undef VECTOR_NAME
#undef VECTOR_TYPE
#undef VECTOR_TYPE_ABS

#endif
