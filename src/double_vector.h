#ifndef DOUBLE_VECTOR_H
#define DOUBLE_VECTOR_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>

#define VECTOR_NAME double_vector
#define VECTOR_TYPE double
#define VECTOR_TYPE_ABS fabs
#include "numeric.h"
#undef VECTOR_NAME
#undef VECTOR_TYPE
#undef VECTOR_TYPE_ABS

#endif
