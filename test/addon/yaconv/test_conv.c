/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <blis.h>
#include <cblas.h>

float *alloc_random( int size );

float *alloc_yaconv_output( int H, int FH, int PH, int PW, int OW, int M );

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            float *data_col);

int main( int argc, char** argv ) {
    // Usage message
    if ( argc < 10 )
    {
        fprintf( stderr, "Usage: ./test_conv N C H W M FH FW PH PW\n");
        return -1;
    }

    // Get convolution parameters from CLI, compute OH and OW
    const int N = atoi(argv[1]);
    const int C = atoi(argv[2]);
    const int H = atoi(argv[3]);
    const int W = atoi(argv[4]);
    const int M = atoi(argv[5]);
    const int FH = atoi(argv[6]);
    const int FW = atoi(argv[7]);
    const int PH = atoi(argv[8]);
    const int PW = atoi(argv[9]);
    const int OH = H + 2 * PH - FH + 1;
    const int OW = W + 2 * PW - FW + 1;

    // index variable over N input and output images
    int i;

    // Allocate arrays
    float *filter = alloc_random( M * FH * FW * C );
    float *images = alloc_random( N * H * W * C );
    float *image;
    float **output = ( float** )malloc( N * sizeof( float* ) );

    // im2col buffer
#if defined(IM2COL_BLIS) || defined(IM2COL_OPENBLAS)
    float *im2col_buf = alloc_random( OH * OW * FH * FW * C );
#endif

    // Output arrays
    // TODO: now yaconv requires some extra memory
    for ( i = 0; i < N; ++i )
#if defined(IM2COL_BLIS) || defined(IM2COL_OPENBLAS)
        output[i] = alloc_random( M * OH * OW );
#else
        output[i] = alloc_yaconv_output( H, FH, PH, PW, OW, M );
#endif

    // Timing and gflops variables
    double dtime;
    double dtime_save = DBL_MAX;
#if defined(IM2COL_BLIS) || defined(IM2COL_OPENBLAS)
    double dtime_save_im2col = DBL_MAX;
#endif
    double gflops;

    for ( i = 0; i < N; ++i )
    {
        dtime = bli_clock();

        image = images + i * H * W * C;

#if defined(IM2COL_BLIS) || defined(IM2COL_OPENBLAS)
        im2col
        (
          image, C, H, W, FH, FW, PH, PW, 1, 1, 1, 1, im2col_buf
        );

        dtime_save_im2col = bli_clock_min_diff( dtime_save_im2col, dtime );

#if defined(IM2COL_BLIS)
        bli_sgemm
        (
          BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, OH * OW, FH * FW * C,
          bli_s1, filter, FH * FW * C, 1, im2col_buf, OH * OW, 1,
          bli_s0, output[i], 1, M
        );
#elif defined(IM2COL_OPENBLAS)
       cblas_sgemm
       (
         CblasColMajor, CblasTrans, CblasTrans, M, OH * OW, FH * FW * C,
         1.0, filter, FH * FW * C, im2col_buf, OH * OW, 0.0, output[i], M
       );
#endif
#else
        bao_yaconv( filter, image, output[i], C, H, W, M, FH, FW, PH, PW );
#endif

        dtime_save = bli_clock_min_diff( dtime_save, dtime );
    }

    gflops = ( 2.0 * M * C * FH * FW * OH * OW ) / ( dtime_save * 1.0e9 );
#if defined(IM2COL_BLIS) || defined(IM2COL_OPENBLAS)
    printf( "%.3f,%.2f\n", dtime_save_im2col / dtime_save, gflops );
#else
    printf( "%7.2f\n", gflops );
#endif

    free( filter );
    free( images );

    // Do not free output in yaconv because it does not point to the beginning
    // of the allocated memory.
    // TODO: change yaconv to not use extra memory at all or store the offset
    // from alloc to be able to free
#if defined(IM2COL_BLIS) || defined(IM2COL_OPENBLAS)
    free( output );
#endif

    return 0;
}

float *alloc_random( int size )
{
    // Try to allocate
    float *data = ( float* )malloc( size * sizeof( float ) );

    // Handle potential malloc error
    if ( data == NULL )
    {
        fprintf( stderr, "Some error in malloc!\n" );
        exit( -1 );
    }

    // // Fill with a sequence
    // for ( int i = 0; i < size; ++i )
    //     data[i] = i;
    // Randomize
    bli_srandv( size, data, 1 );

    return data;
}

float *alloc_yaconv_output( int H, int FH, int PH, int PW, int OW, int M )
{
    cntx_t *cntx = bli_gks_query_cntx();
    int NR = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_NR, cntx );

    int whole_H = H % NR == 0 ? H : H + NR - H % NR;

    int extra_before = ( FH - 1 ) * OW * M;
    int output_and_after = ( whole_H + PH ) * OW * M;

    float *output = alloc_random( extra_before + output_and_after );

    return output + extra_before;
}

// The following two functions are taken from
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp#L14-L55
// static_cast in is_a_ge_zero_and_a_lt_b was changed to C-style cast
inline bool is_a_ge_zero_and_a_lt_b( int a, int b )
{
    return ( unsigned )a < ( unsigned )b;
}

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            float *data_col) {

  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if ( !is_a_ge_zero_and_a_lt_b(input_row, height ) ) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if ( is_a_ge_zero_and_a_lt_b( input_col, width ) ) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}
