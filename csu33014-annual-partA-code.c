//
// CSU33014 Summer 2020 Additional Assignment
// Part A of a two-part assignment
//

// Please examine version each of the following routines with names
// starting partA. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics. Where is cannot be vectorized...

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".

#include <immintrin.h>
#include <stdio.h>

#include "csu33014-annual-partA-code.h"

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void partA_routine0(float *restrict a, float *restrict b,
                    float *restrict c)
{
  for (int i = 0; i < 1024; i++)
  {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void partA_vectorized0(float *restrict a, float *restrict b,
                       float *restrict c)
{
  __m128 a4, b4, c4;

  for (int i = 0; i < 1024; i = i + 4)
  {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float partA_routine1(float *restrict a, float *restrict b,
                     int size)
{
  float sum = 0.0;

  for (int i = 0; i < size; i++)
  {
    sum = sum + a[i] * b[i];
  }
  return sum;
}

// insert vectorized code for routine1 here
float partA_vectorized1(float *restrict a, float *restrict b,
                        int size)
{
  // replace the following code with vectorized code
  float *restrict sum;
  __m128 a4, b4, temp, sum4;
  for (int i = 0; i < size; i = i + 4)
  {
    a4 = _mm_loadu_ps(&a[i]);
    b4 = _mm_loadu_ps(&b[i]);
    temp = _mm_mul_ps(a4, b4);
    sum4 = _mm_add_ps(sum4, temp);
    _mm_storeu_ps(&sum[i], sum4);
  }
  //return sum;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void partA_routine2(float *restrict a, float *restrict b, int size)
{
  for (int i = 0; i < size; i++)
  {
    a[i] = 1 - (1.0 / (b[i] + 1.0));
  }
}

// in the following, size can have any positive value
void partA_vectorized2(float *restrict a, float *restrict b, int size)
{
  // replace the following code with vectorized code
  __m128 a4, b4, dem, one4, temp;
  for (int i = 0; i < size; i = i + 4)
  {
    //a[i] = 1 - (1.0/(b[i]+1.0));
    b4 = _mm_loadu_ps(&b[i]);
    one4 = _mm_loadu_ps(1);
    dem = _mm_add_ps(b4, one4);
    temp = _mm_div_ps(one4, dem);
    a4 = _mm_sub_ps(one4, temp);
    _mm_storeu(&a[i], a4);
  }
}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void partA_routine3(float *restrict a, float *restrict b, int size)
{
  for (int i = 0; i < size; i++)
  {
    if (a[i] < 0.0)
    {
      a[i] = b[i];
    }
  }
}

// in the following, size can have any positive value
void partA_vectorized3(float *restrict a, float *restrict b, int size)
{
  // replace the following code with vectorized code
  __m128 a4, b4, zero4, mask, a_min, zero_min;
  for (int i = 0; i < size; i++)
  {
    //if ( a[i] < 0.0 ) {
    // a[i] = b[i];
    //}
    a4 = _mm_loadu_ps(&a[i]);
    zero4 = _mm_loadu_ps(0);
    mask = _mm_cmplt_ps(a4, zero4);     // make mask to find min betweeen a and 0
    a_min = _mm_and_ps(a4, mask);       //find where a is min
    zero_min = _mm_and_ps(zero4, mask); // find where 0 is min
    a4 = _mm_xor_ps(a_min, zero_min);
    //b4 =
    _mm_storeu_ps(&a[i], a4);
  }
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void partA_routine4(float *restrict a, float *restrict b,
                    float *restrict c)
{
  for (int i = 0; i < 2048; i = i + 2)
  {
    a[i] = b[i] * c[i] - b[i + 1] * c[i + 1];
    a[i + 1] = b[i] * c[i + 1] + b[i + 1] * c[i];
  }
}

void partA_vectorized4(float *restrict a, float *restrict b,
                       float *restrict c)
{
  __m128 a4, b4, c4, b2, c2, zero4, t1, t2, ai4;
  int getai;
  // replace the following code with vectorized code
  for (int i = 0; i < 2048; i = i + 2)
  {
    //  a[i] = b[i]*c[i] - b[i+1]*c[i+1];
    //  a[i+1] = b[i]*c[i+1] + b[i+1]*c[i];
    b4 = _mm_loadu_ps(&b);
    c4 = _mm_loadu_ps(&c);
    zero4 = _mm_loadu_ps(0);
    b2 = _mm_shuffle_ps(b4, zero4, _MM_SHUFFLE(3, 2, 1, 0)); //create vector |b[i]|b[i+1]|0|0|
    c2 = _mm_shuffle_ps(c4, zero4, _MM_SHUFFLE(3, 2, 1, 0)); //create vector |c[i]|c[i+1]|0|0|
    t1 = _mm_mul_ps(b2, c2);                                 //t1 = |b[i]*c[i]|b[i+1]*c[i+1]|0|0|
    t2 = t1;
    //turn t2 into negative of t1
    //hadd (i.e t1 + (-t2))
    //store in a
    //a[i+1].....
    //temp = _mm_shuffle(t1,t1, _MM_SHUFFLE())
    ai4 = _mm_sub_ps(t1, t1);
    _mm_storeu_ps(&a[i], a4);
  }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
void partA_routine5(unsigned char *restrict a,
                    unsigned char *restrict b, int size)
{
  for (int i = 0; i < size; i++)
  {
    a[i] = b[i];
  }
}

void partA_vectorized5(unsigned char *restrict a,
                       unsigned char *restrict b, int size)
{
  // replace the following code with vectorized code
  __m128 a4, b4;
  for (int i = 0; i < size; i = i + 4)
  {
    //a[i] = b[i];
    b4 = _mm_loadu_ps(&b);
    a4 = b4;
    _mm_storeu_ps(&a[i], a4);
  }
}

/********************* routine 6 ***********************/

void partA_routine6(float *restrict a, float *restrict b,
                    float *restrict c)
{
  a[0] = 0.0;
  for (int i = 1; i < 1023; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < 3; j++)
    {
      sum = sum + b[i + j - 1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void partA_vectorized6(float *restrict a, float *restrict b,
                       float *restrict c)
{
  // replace the following code with vectorized code
  __m128 a4, b4, c4, sum4, temp;
  a[0] = 0.0;
  a4 = _mm_loadu_ps(&a);
  for (int i = 1; i < 1023; i++)
  {
    float sum = 0.0;
    sum4 = _mm_loadu_ps(&sum);
    for (int j = 0; j < 3; j++)
    {
      //sum = sum +  b[i+j-1] * c[j];
      b4 = _mm_loadu_ps(&b);
      c4 = _mm_loadu_ps(&c);
      temp = _mm_mul_ps(b4, c4); //***not correct!
      sum4 = _mm_add_ps(sum4, temp);
    }
    //a[i] = sum;
    a4 = sum4;
    _mm_storeu_ps(&a[i], a4);
  }
  a[1023] = 0.0;
}


