#ifndef VECTORIZED_SIMD_MATH_H
#define VECTORIZED_SIMD_MATH_H

#include <stdint.h>
#include <float.h>
#include <math.h>
#include <fenv.h>
#ifdef _MSC_VER
#include <limits.h>
#endif

#include "aligned/aligned.h"
#include "simde_avx2/avx2.h"

#define _PI32AVX_CONST(name, val)                                            \
  static const alignas(32) int simde_pi32avx_##name[4] = { val, val, val, val }

_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);

#define _PS256_CONST(name, val)                                            \
  static const alignas(32) float _ps256_##name[8] = { val, val, val, val, val, val, val, val }
#define _PI32_CONST256(name, val)                                            \
  static const alignas(32) int _pi32_256_##name[8] = { val, val, val, val, val, val, val, val }
#define _PS256_CONST_TYPE(name, type, val)                                 \
  static const alignas(32) type _ps256_##name[8] = { val, val, val, val, val, val, val, val }

_PS256_CONST(1  , 1.0f);
_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2);
_PS256_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1);
_PS256_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS256_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS256_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS256_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS256_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS256_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS256_CONST(cephes_log_q1, -2.12194440e-4);
_PS256_CONST(cephes_log_q2, 0.693359375);

/* natural logarithm computed for 8 simultaneous float 
   return NaN for x <= 0
*/
simde__m256 log256_ps(simde__m256 x) {
  simde__m256i imm0;
  simde__m256 one = *(simde__m256*)_ps256_1;

  //simde__m256 invalid_mask = simde_mm256_cmple_ps(x, simde_mm256_setzero_ps());
  simde__m256 invalid_mask = simde_mm256_cmp_ps(x, simde_mm256_setzero_ps(), SIMDE_CMP_LE_OS);

  x = simde_mm256_max_ps(x, *(simde__m256*)_ps256_min_norm_pos);  /* cut off denormalized stuff */

  // can be done with AVX2
  imm0 = simde_mm256_srli_epi32(simde_mm256_castps_si256(x), 23);

  /* keep only the fractional part */
  x = simde_mm256_and_ps(x, *(simde__m256*)_ps256_inv_mant_mask);
  x = simde_mm256_or_ps(x, *(simde__m256*)_ps256_0p5);

  // this is again another AVX2 instruction
  imm0 = simde_mm256_sub_epi32(imm0, *(simde__m256i*)_pi32_256_0x7f);
  simde__m256 e = simde_mm256_cvtepi32_ps(imm0);

  e = simde_mm256_add_ps(e, one);

  /* part2: 
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  //simde__m256 mask = simde_mm256_cmplt_ps(x, *(simde__m256*)_ps256_cephes_SQRTHF);
  simde__m256 mask = simde_mm256_cmp_ps(x, *(simde__m256*)_ps256_cephes_SQRTHF, SIMDE_CMP_LT_OS);
  simde__m256 tmp = simde_mm256_and_ps(x, mask);
  x = simde_mm256_sub_ps(x, one);
  e = simde_mm256_sub_ps(e, simde_mm256_and_ps(one, mask));
  x = simde_mm256_add_ps(x, tmp);

  simde__m256 z = simde_mm256_mul_ps(x,x);

  simde__m256 y = *(simde__m256*)_ps256_cephes_log_p0;
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_log_p1);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_log_p2);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_log_p3);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_log_p4);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_log_p5);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_log_p6);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_log_p7);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_log_p8);
  y = simde_mm256_mul_ps(y, x);

  y = simde_mm256_mul_ps(y, z);
  
  tmp = simde_mm256_mul_ps(e, *(simde__m256*)_ps256_cephes_log_q1);
  y = simde_mm256_add_ps(y, tmp);


  tmp = simde_mm256_mul_ps(z, *(simde__m256*)_ps256_0p5);
  y = simde_mm256_sub_ps(y, tmp);

  tmp = simde_mm256_mul_ps(e, *(simde__m256*)_ps256_cephes_log_q2);
  x = simde_mm256_add_ps(x, y);
  x = simde_mm256_add_ps(x, tmp);
  x = simde_mm256_or_ps(x, invalid_mask); // negative arg will be NAN
  return x;
}

_PS256_CONST(exp_hi,	88.3762626647949f);
_PS256_CONST(exp_lo,	-88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

simde__m256 exp256_ps(simde__m256 x) {
  simde__m256 tmp = simde_mm256_setzero_ps(), fx;
  simde__m256i imm0;
  simde__m256 one = *(simde__m256*)_ps256_1;

  x = simde_mm256_min_ps(x, *(simde__m256*)_ps256_exp_hi);
  x = simde_mm256_max_ps(x, *(simde__m256*)_ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = simde_mm256_mul_ps(x, *(simde__m256*)_ps256_cephes_LOG2EF);
  fx = simde_mm256_add_ps(fx, *(simde__m256*)_ps256_0p5);

  /* how to perform a floorf with SSE: just below */
  //imm0 = simde_mm256_cvttps_epi32(fx);
  //tmp  = simde_mm256_cvtepi32_ps(imm0);
  
  tmp = simde_mm256_floor_ps(fx);

  /* if greater, substract 1 */
  //simde__m256 mask = simde_mm256_cmpgt_ps(tmp, fx);    
  simde__m256 mask = simde_mm256_cmp_ps(tmp, fx, SIMDE_CMP_GT_OS);
  mask = simde_mm256_and_ps(mask, one);
  fx = simde_mm256_sub_ps(tmp, mask);

  tmp = simde_mm256_mul_ps(fx, *(simde__m256*)_ps256_cephes_exp_C1);
  simde__m256 z = simde_mm256_mul_ps(fx, *(simde__m256*)_ps256_cephes_exp_C2);
  x = simde_mm256_sub_ps(x, tmp);
  x = simde_mm256_sub_ps(x, z);

  z = simde_mm256_mul_ps(x,x);
  
  simde__m256 y = *(simde__m256*)_ps256_cephes_exp_p0;
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_exp_p1);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_exp_p2);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_exp_p3);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_exp_p4);
  y = simde_mm256_mul_ps(y, x);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_cephes_exp_p5);
  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_add_ps(y, x);
  y = simde_mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = simde_mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = simde_mm256_add_epi32(imm0, *(simde__m256i*)_pi32_256_0x7f);
  imm0 = simde_mm256_slli_epi32(imm0, 23);
  simde__m256 pow2n = simde_mm256_castsi256_ps(imm0);
  y = simde_mm256_mul_ps(y, pow2n);
  return y;
}

_PS256_CONST(minus_cephes_DP1, -0.78515625);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS256_CONST(sincof_p0, -1.9515295891E-4);
_PS256_CONST(sincof_p1,  8.3321608736E-3);
_PS256_CONST(sincof_p2, -1.6666654611E-1);
_PS256_CONST(coscof_p0,  2.443315711809948E-005);
_PS256_CONST(coscof_p1, -1.388731625493765E-003);
_PS256_CONST(coscof_p2,  4.166664568298827E-002);
_PS256_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI


/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
simde__m256 sin256_ps(simde__m256 x) { // any x
  simde__m256 xmm1, xmm2 = simde_mm256_setzero_ps(), xmm3, sign_bit, y;
  simde__m256i imm0, imm2;

  sign_bit = x;
  /* take the absolute value */
  x = simde_mm256_and_ps(x, *(simde__m256*)_ps256_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit = simde_mm256_and_ps(sign_bit, *(simde__m256*)_ps256_sign_mask);
  
  /* scale by 4/Pi */
  y = simde_mm256_mul_ps(x, *(simde__m256*)_ps256_cephes_FOPI);

  /* store the integer part of y in mm0 */
  imm2 = simde_mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  // another two AVX2 instruction
  imm2 = simde_mm256_add_epi32(imm2, *(simde__m256i*)_pi32_256_1);
  imm2 = simde_mm256_and_si256(imm2, *(simde__m256i*)_pi32_256_inv1);
  y = simde_mm256_cvtepi32_ps(imm2);

  /* get the swap sign flag */
  imm0 = simde_mm256_and_si256(imm2, *(simde__m256i*)_pi32_256_4);
  imm0 = simde_mm256_slli_epi32(imm0, 29);
  /* get the polynom selection mask 
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  imm2 = simde_mm256_and_si256(imm2, *(simde__m256i*)_pi32_256_2);
  imm2 = simde_mm256_cmpeq_epi32(imm2,*(simde__m256i*)_pi32_256_0);

  simde__m256 swap_sign_bit = simde_mm256_castsi256_ps(imm0);
  simde__m256 poly_mask = simde_mm256_castsi256_ps(imm2);
  sign_bit = simde_mm256_xor_ps(sign_bit, swap_sign_bit);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(simde__m256*)_ps256_minus_cephes_DP1;
  xmm2 = *(simde__m256*)_ps256_minus_cephes_DP2;
  xmm3 = *(simde__m256*)_ps256_minus_cephes_DP3;
  xmm1 = simde_mm256_mul_ps(y, xmm1);
  xmm2 = simde_mm256_mul_ps(y, xmm2);
  xmm3 = simde_mm256_mul_ps(y, xmm3);
  x = simde_mm256_add_ps(x, xmm1);
  x = simde_mm256_add_ps(x, xmm2);
  x = simde_mm256_add_ps(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(simde__m256*)_ps256_coscof_p0;
  simde__m256 z = simde_mm256_mul_ps(x,x);

  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_coscof_p1);
  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_coscof_p2);
  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_mul_ps(y, z);
  simde__m256 tmp = simde_mm256_mul_ps(z, *(simde__m256*)_ps256_0p5);
  y = simde_mm256_sub_ps(y, tmp);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_1);
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  simde__m256 y2 = *(simde__m256*)_ps256_sincof_p0;
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_add_ps(y2, *(simde__m256*)_ps256_sincof_p1);
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_add_ps(y2, *(simde__m256*)_ps256_sincof_p2);
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_mul_ps(y2, x);
  y2 = simde_mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  y2 = simde_mm256_and_ps(xmm3, y2); //, xmm3);
  y = simde_mm256_andnot_ps(xmm3, y);
  y = simde_mm256_add_ps(y,y2);
  /* update the sign */
  y = simde_mm256_xor_ps(y, sign_bit);

  return y;
}

/* almost the same as sin_ps */
simde__m256 cos256_ps(simde__m256 x) { // any x
  simde__m256 xmm1, xmm2 = simde_mm256_setzero_ps(), xmm3, y;
  simde__m256i imm0, imm2;

  /* take the absolute value */
  x = simde_mm256_and_ps(x, *(simde__m256*)_ps256_inv_sign_mask);
  
  /* scale by 4/Pi */
  y = simde_mm256_mul_ps(x, *(simde__m256*)_ps256_cephes_FOPI);
  
  /* store the integer part of y in mm0 */
  imm2 = simde_mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = simde_mm256_add_epi32(imm2, *(simde__m256i*)_pi32_256_1);
  imm2 = simde_mm256_and_si256(imm2, *(simde__m256i*)_pi32_256_inv1);
  y = simde_mm256_cvtepi32_ps(imm2);
  imm2 = simde_mm256_sub_epi32(imm2, *(simde__m256i*)_pi32_256_2);
  
  /* get the swap sign flag */
  imm0 = simde_mm256_andnot_si256(imm2, *(simde__m256i*)_pi32_256_4);
  imm0 = simde_mm256_slli_epi32(imm0, 29);
  /* get the polynom selection mask */
  imm2 = simde_mm256_and_si256(imm2, *(simde__m256i*)_pi32_256_2);
  imm2 = simde_mm256_cmpeq_epi32(imm2, *(simde__m256i*)_pi32_256_0);

  simde__m256 sign_bit = simde_mm256_castsi256_ps(imm0);
  simde__m256 poly_mask = simde_mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(simde__m256*)_ps256_minus_cephes_DP1;
  xmm2 = *(simde__m256*)_ps256_minus_cephes_DP2;
  xmm3 = *(simde__m256*)_ps256_minus_cephes_DP3;
  xmm1 = simde_mm256_mul_ps(y, xmm1);
  xmm2 = simde_mm256_mul_ps(y, xmm2);
  xmm3 = simde_mm256_mul_ps(y, xmm3);
  x = simde_mm256_add_ps(x, xmm1);
  x = simde_mm256_add_ps(x, xmm2);
  x = simde_mm256_add_ps(x, xmm3);
  
  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(simde__m256*)_ps256_coscof_p0;
  simde__m256 z = simde_mm256_mul_ps(x,x);

  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_coscof_p1);
  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_coscof_p2);
  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_mul_ps(y, z);
  simde__m256 tmp = simde_mm256_mul_ps(z, *(simde__m256*)_ps256_0p5);
  y = simde_mm256_sub_ps(y, tmp);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_1);
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  simde__m256 y2 = *(simde__m256*)_ps256_sincof_p0;
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_add_ps(y2, *(simde__m256*)_ps256_sincof_p1);
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_add_ps(y2, *(simde__m256*)_ps256_sincof_p2);
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_mul_ps(y2, x);
  y2 = simde_mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  y2 = simde_mm256_and_ps(xmm3, y2); //, xmm3);
  y = simde_mm256_andnot_ps(xmm3, y);
  y = simde_mm256_add_ps(y,y2);
  /* update the sign */
  y = simde_mm256_xor_ps(y, sign_bit);

  return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */

void sincos256_ps(simde__m256 x, simde__m256 *s, simde__m256 *c) {

  simde__m256 xmm1, xmm2, xmm3 = simde_mm256_setzero_ps(), sign_bit_sin, y;
  simde__m256i imm0, imm2, imm4;

  sign_bit_sin = x;
  /* take the absolute value */
  x = simde_mm256_and_ps(x, *(simde__m256*)_ps256_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit_sin = simde_mm256_and_ps(sign_bit_sin, *(simde__m256*)_ps256_sign_mask);
  
  /* scale by 4/Pi */
  y = simde_mm256_mul_ps(x, *(simde__m256*)_ps256_cephes_FOPI);

  /* store the integer part of y in imm2 */
  imm2 = simde_mm256_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = simde_mm256_add_epi32(imm2, *(simde__m256i*)_pi32_256_1);
  imm2 = simde_mm256_and_si256(imm2, *(simde__m256i*)_pi32_256_inv1);

  y = simde_mm256_cvtepi32_ps(imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = simde_mm256_and_si256(imm2, *(simde__m256i*)_pi32_256_4);
  imm0 = simde_mm256_slli_epi32(imm0, 29);
  //simde__m256 swap_sign_bit_sin = simde_mm256_castsi256_ps(imm0);

  /* get the polynom selection mask for the sine*/
  imm2 = simde_mm256_and_si256(imm2, *(simde__m256i*)_pi32_256_2);
  imm2 = simde_mm256_cmpeq_epi32(imm2, *(simde__m256i*)_pi32_256_0);
  //simde__m256 poly_mask = simde_mm256_castsi256_ps(imm2);

  simde__m256 swap_sign_bit_sin = simde_mm256_castsi256_ps(imm0);
  simde__m256 poly_mask = simde_mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(simde__m256*)_ps256_minus_cephes_DP1;
  xmm2 = *(simde__m256*)_ps256_minus_cephes_DP2;
  xmm3 = *(simde__m256*)_ps256_minus_cephes_DP3;
  xmm1 = simde_mm256_mul_ps(y, xmm1);
  xmm2 = simde_mm256_mul_ps(y, xmm2);
  xmm3 = simde_mm256_mul_ps(y, xmm3);
  x = simde_mm256_add_ps(x, xmm1);
  x = simde_mm256_add_ps(x, xmm2);
  x = simde_mm256_add_ps(x, xmm3);

  imm4 = simde_mm256_sub_epi32(imm4, *(simde__m256i*)_pi32_256_2);
  imm4 = simde_mm256_andnot_si256(imm4, *(simde__m256i*)_pi32_256_4);
  imm4 = simde_mm256_slli_epi32(imm4, 29);

  simde__m256 sign_bit_cos = simde_mm256_castsi256_ps(imm4);

  sign_bit_sin = simde_mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);
  
  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  simde__m256 z = simde_mm256_mul_ps(x,x);
  y = *(simde__m256*)_ps256_coscof_p0;

  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_coscof_p1);
  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_coscof_p2);
  y = simde_mm256_mul_ps(y, z);
  y = simde_mm256_mul_ps(y, z);
  simde__m256 tmp = simde_mm256_mul_ps(z, *(simde__m256*)_ps256_0p5);
  y = simde_mm256_sub_ps(y, tmp);
  y = simde_mm256_add_ps(y, *(simde__m256*)_ps256_1);
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  simde__m256 y2 = *(simde__m256*)_ps256_sincof_p0;
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_add_ps(y2, *(simde__m256*)_ps256_sincof_p1);
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_add_ps(y2, *(simde__m256*)_ps256_sincof_p2);
  y2 = simde_mm256_mul_ps(y2, z);
  y2 = simde_mm256_mul_ps(y2, x);
  y2 = simde_mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  simde__m256 ysin2 = simde_mm256_and_ps(xmm3, y2);
  simde__m256 ysin1 = simde_mm256_andnot_ps(xmm3, y);
  y2 = simde_mm256_sub_ps(y2,ysin2);
  y = simde_mm256_sub_ps(y, ysin1);

  xmm1 = simde_mm256_add_ps(ysin1,ysin2);
  xmm2 = simde_mm256_add_ps(y,y2);
 
  /* update the sign */
  *s = simde_mm256_xor_ps(xmm1, sign_bit_sin);
  *c = simde_mm256_xor_ps(xmm2, sign_bit_cos);
}

#endif