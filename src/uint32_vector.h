#ifndef UINT32_VECTOR_H
#define UINT32_VECTOR_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "nop.h"

#define VECTOR_NAME uint32_vector
#define VECTOR_TYPE uint32_t
#define VECTOR_TYPE_ABS nop
#include "numeric.h"
#undef VECTOR_NAME
#undef VECTOR_TYPE
#undef VECTOR_TYPE_ABS

#endif
