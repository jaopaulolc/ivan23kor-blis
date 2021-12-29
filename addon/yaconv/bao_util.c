/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

#include "blis.h"

float *bao_aligned_alloc( int size )
{
    float *data = NULL;
    int ret = posix_memalign
    (
      ( void** )&data, BLIS_PAGE_SIZE, size * sizeof( float )
    );

    if ( ret == 0 )
      return data;

    fprintf(stderr, "Some allocation error in posix_memalign\n");
    exit(ret);
}

float *bao_seq_aligned_alloc( int size )
{
    float *data = bao_aligned_alloc( size );
    int i;

    for ( i = 0; i < size; ++i )
        data[i] = i + 1;
    return data;
}

static float **yaconv_pools = NULL;

void bao_init_yaconv_pools( int num, int *sizes, float*(malloc_fp)(int) )
{
    int i;

    if ( yaconv_pools != NULL )
      return;

    yaconv_pools = ( float** )malloc( num * sizeof( float* ) );
    for ( i = 0; i < num; ++i )
        yaconv_pools[i] = malloc_fp( sizes[i] );
}

float *bao_get_yaconv_pool( int index ) { return yaconv_pools[index]; }

void bao_yaconv_pack
     (
       float *src, dim_t rss, dim_t css,
       float *dst,
       int MN, int k, int MNR,
       cntx_t *cntx
     )
{
    int mn;

    for ( mn = 0; mn < MN; mn += MNR )
        /* If a packing microkernel for this register block size is found,
           use it. Otherwise, use scal2m */
        bli_spackm_cxk
        (
          BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS,
          bli_min( MN - mn, MNR ), MNR, k, k,
          bli_s1,
          src + mn * rss, rss, css,
          dst + mn * k, MNR,
          cntx
        );
}
