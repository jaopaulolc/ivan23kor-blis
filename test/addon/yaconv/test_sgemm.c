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


#ifndef REPEAT
#define REPEAT 10
#endif


float *alloc( int size ) {
  float *data = ( float * )malloc( size * sizeof( float ) );
  for ( int i = 0; i < size; ++i )
    data[i] = i + 1;
  return data;
}


int main( int argc, char** argv ) {
  float *a, *b, *c;
	float alpha = -0.8, beta = 1.3;
	dim_t m, n, k;
	dim_t p_begin = 100, p_inc = 100, p_end = 2000, p;

	double dtime;
	double dtime_save;
	double gflops;

	for ( p = p_begin; p <= p_end; p += p_inc ) {
    m = k = n = p;

    a = alloc( m * k );
    b = alloc( k * n );
    c = alloc( m * n );

		dtime_save = DBL_MAX;

		for ( int r = 0; r < REPEAT; ++r )
		{
			dtime = bli_clock();

#if   defined(BLIS)
      bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha, a, k, 1, b, n, 1, &beta, c, n, 1 );
#elif defined(OPENBLAS)
      cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, m );
#else
      bao_sgemm( a, b, c, m, k, n, k, n, n, alpha, beta );
#endif

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

		printf( "sgemm" );
		printf( "( %2llu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )k,
		        ( unsigned long )n, gflops );

		free( a );
		free( b );
		free( c );
	}

	return 0;
}

