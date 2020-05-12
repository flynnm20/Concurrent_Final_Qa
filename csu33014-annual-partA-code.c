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
  float sum = 0.0;
  __m128 a4, b4;
  // idea. keep taking groups of 4 until the difference between i and size is less than 4.
  for (int i = 0; i < size - 3; i++)
  {
    sum = sum + a[i] * b[i];
  }
  return sum;
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
  __m128 a4, b4, sum, division;
  float one = 1;
  int max_Mulitiple = size - (size % 4);
  __m128 ones = _mm_set1_ps(one);
  for (int i = 0; i < max_Mulitiple; i + 4)
  {
    b4 = _mm_loadu_ps(&b[i]);
    sum = _mm_add_ps(b4, ones);      // b[i]+1
    division = _mm_div_ps(ones, b4); //1/(b[i]-1)
    a4 = _mm_sub_ps(ones, b4);       //a[i] = 1 - (1.0 / (b[i] + 1.0));
    _mm_storeu_ps(&a[i], a4);        // store in a.
  }
  // now have at most 3 extra values;
  for (int j = max_Mulitiple; j < size; j++)
  {
    a[size - j] = 1 - (1.0 / (b[size - j] + 1.0));
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
  __m128 a4, b4, mask;
  __m128 zeros = _mm_set1_ps(0.0);
  int max_Mulitiple = size - (size % 4);
  for (int i = 0; i < max_Mulitiple; i + 4)
  {
    a4 = _mm_loadu_ps(&a[i]);       // get 4 valuse of a
    b4 = _mm_loadu_ps(&b[i]);       // get 4 values of b
    mask = _mm_cmplt_ps(a4, zeros); // create a mask using comparrison.
    b4 = _mm_and_ps(b4, mask);      // anding with the mask will give the values which need replacement
    a4 = _mm_andnot_ps(mask, a4);   // remove a's that need to be replaced.
    a4 = _mm_andnot_ps(a4, b4);     // fill in all the area's that need to be filled in.
    _mm_storeu_ps(&a[i], a4);       // store the updated a4 back in the orignal a.
  }
  for (int i = max_Mulitiple; i < size; i++)
  {
    if (a[i] < 0.0)
    {
      a[i] = b[i];
    }
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
  // replace the following code with vectorized code
  for (int i = 0; i < 2048; i = i + 2)
  {
    a[i] = b[i] * c[i] - b[i + 1] * c[i + 1];
    a[i + 1] = b[i] * c[i + 1] + b[i + 1] * c[i];
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
  int max_mul = size - (size % 4);
  for (int i = 0; i < max_mul; i + 4)
  {
    a4 = _mm_loadu_ps(&a[i]); // get 4 valuse of a
    b4 = _mm_loadu_ps(&b[i]); // get 4 values of b
    _mm_storeu_ps(&a[i], b4); // store the b4 in a.
    _mm_storeu_ps(&b[i], a4); // store the a4 in b.
  }
  for (int i = max_mul; i < size; i++)
  {
    a[i] = b[i];
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
