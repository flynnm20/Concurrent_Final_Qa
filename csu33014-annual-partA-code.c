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
  float sum = 0.0;                              // track sum
  __m128 a4, b4, product, brokeSum;             // generate variables
  brokeSum = _mm_set1_ps(0.0);                  // initialise sum.
  int max_Mulitiple = size - (size % 4);        // get the max multiple of 4
  for (int i = 0; i < max_Mulitiple; i = i + 4) // preform the vectoried section
  {
    a4 = _mm_loadu_ps(&a[i]);                 // load 4 variables from a
    b4 = _mm_loadu_ps(&b[i]);                 // load 4 variables from b
    product = _mm_mul_ps(a4, b4);             // a[i] * b[i]
    brokeSum = _mm_add_ps(brokeSum, product); // sum = sum + a[i] * b[i]
  }
  brokeSum = _mm_hadd_ps(brokeSum, brokeSum); // combine the first pair of results and second pair of results
  brokeSum = _mm_hadd_ps(brokeSum, brokeSum); // sum the remaining resuls so it is all in 1 variable.
  sum = _mm_cvtss_f32(brokeSum);              // get the sum from the front.
  for (int j = max_Mulitiple; j < size; j++)  // handle the extra end.
  {
    sum = sum + a[j] * b[j];
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
  __m128 a4, b4, sum, division;          // generate variables
  float one = 1;                         // Use to initialise a constant vector.
  int max_Mulitiple = size - (size % 4); // get the max multiple of 4 less then size
  __m128 ones = _mm_set1_ps(one);        // create a constant vector

  for (int i = 0; i < max_Mulitiple; i = i + 4) // vectorised loop
  {
    b4 = _mm_loadu_ps(&b[i]);         // load 4 values from b
    sum = _mm_add_ps(b4, ones);       // b[i]+1
    division = _mm_div_ps(ones, sum); //1/(b[i]-1)
    a4 = _mm_sub_ps(ones, division);  //a[i] = 1 - (1.0 / (b[i] + 1.0));
    _mm_storeu_ps(&a[i], a4);         // store in a.
  }
  // now have at most 3 extra values
  for (int j = max_Mulitiple; j < size; j++)
  {
    a[j] = 1 - (1.0 / (b[j] + 1.0));
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
  __m128 a4, b4, mask, removedElemsB, removedElemsA, mask2, results;
  __m128 zeros = _mm_set1_ps(0.0);
  int max_Mulitiple = size - (size % 4);
  for (int i = 0; i < max_Mulitiple; i = i + 4)
  {
    a4 = _mm_loadu_ps(&a[i]);                           // get 4 valuse of a
    b4 = _mm_loadu_ps(&b[i]);                           // get 4 values of b
    mask = _mm_cmplt_ps(a4, zeros);                     // create a mask using comparrison.
    mask2 = _mm_cmpge_ps(a4, zeros);                    //inverse of mask                  // create a mask using comparrison.
    removedElemsB = _mm_and_ps(b4, mask);               // anding with the mask will give the values which need replacemen    a4 = _mm_andnot_ps(mask, a4);   // remove a's that need to be replaced.
    removedElemsA = _mm_and_ps(a4, mask2);              // anding with the mask will give the values which need replacemen    a4 = _mm_andnot_ps(mask, a4);   // remove a's that need to be replaced.
    results = _mm_add_ps(removedElemsA, removedElemsB); // fill in all the area's that need to be filled in.
    _mm_storeu_ps(&a[i], results);                      // store the updated a4 back in the orignal a.
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
  __m128 b4, c4, b4plus, c4plus, product, productplus1, firstgroup, secondgroup, results;
  // replace the following code with vectorized code
  for (int i = 0; i < 2048; i = i + 4)
  {
    b4 = _mm_loadu_ps(&b[i]);         // get 4 values of b
    c4 = _mm_loadu_ps(&c[i]);         // get 4 valuse of a
    b4plus = _mm_loadu_ps(&b[i + 1]); // get i+1 4 values of b
    c4plus = _mm_loadu_ps(&c[i + 1]); // get i+1 4 valuse of a

    product = _mm_mul_ps(b4, c4);                   //b[i] * c[i]
    productplus1 = _mm_mul_ps(c4plus, b4plus);      // b[i + 1] * c[i + 1];
    firstgroup = _mm_sub_ps(product, productplus1); //a[i] = b[i] * c[i] - b[i + 1] * c[i + 1];

    product = _mm_mul_ps(b4, c4plus);                // b[i] * c[i + 1]
    productplus1 = _mm_mul_ps(c4, b4plus);           // b[i + 1] * c[i]
    secondgroup = _mm_add_ps(productplus1, product); // a[i + 1] = b[i] * c[i + 1] + b[i + 1] * c[i];

    results = _mm_shuffle_ps(firstgroup, secondgroup, _MM_SHUFFLE(3, 1, 2, 0)); // combine the 2 halfs
    results = _mm_shuffle_ps(results, results, _MM_SHUFFLE(3, 2, 1, 0));        // combine the 2 halfs

    _mm_storeu_ps(&a[i], results); // store the updated a4 back in the orignal a.
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
  // since we aren't working with floats we need to cast them correctly.
  __m128i a4, b4, tmp;
  int max_mul = size - (size % 4);
  for (int i = 0; i < max_mul; i = i + 16)
  {
    b4 = _mm_load_si128((const __m128i *)&b[i]); //cast char as an _m128i
    _mm_storeu_si128(&a[i], b4);                 // store b[i] in a[i].
  }
  for (int i = max_mul; i < size; i++) // handle extra end values.
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
  __m128 sumVector, b4, b4Minus, b4Plus, j0, j1, j2;
  __m128 c0 = _mm_load1_ps(&c[0]); // initialise vector of c[0]
  __m128 c1 = _mm_load1_ps(&c[1]); // initialise vector of c[1]
  __m128 c2 = _mm_load1_ps(&c[2]); // initialise vector of c[2]
  // easier to create vectors outside the loop.
  a[0] = 0.0;
  for (int i = 1; i < 1020; i = i + 4)
  {
    sumVector = _mm_set1_ps(0.0);      // sum = 0.0
    b4 = _mm_loadu_ps(&b[i]);          // get b when j =0
    b4Minus = _mm_loadu_ps(&b[i - 1]); // get b when j = 1
    b4Plus = _mm_loadu_ps(&b[i + 1]);  // get b when j = 2

    j0 = _mm_mul_ps(b4Minus, c0); // b[i-1] * c[0]
    j1 = _mm_mul_ps(b4, c1);      // b[i] *c[1]
    j2 = _mm_mul_ps(b4Plus, c2);  // b[i+2] * c[2]

    sumVector = _mm_add_ps(j0, j1);        // sum = (b[i-1] * c[0]) + (b[i] *c[1])
    sumVector = _mm_add_ps(sumVector, j2); // sum = sum + (b[i+2] * c[2])
    _mm_storeu_ps(&a[i], sumVector);       // a[i] = sum
  }
  // handle the last 3 iterations linearly.
  for (int i = 1021; i < 1023; i++)
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
