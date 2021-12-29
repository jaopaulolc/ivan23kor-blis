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
   THEORY OF LIABILOAOAITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include "bao_util.h"

BLIS_EXPORT_ADDON void bao_sgemm
     (
       float *a, float *b, float *c,
       int m, int k, int n,
       int lda, int ldb, int ldc,
       float alpha, float beta
     )
{

    // Get BLIS context
    cntx_t *cntx = bli_gks_query_cntx();

    // Get blocksizes
    int MR = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_MR, cntx );
    int NR = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_NR, cntx );
    int MC = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_MC, cntx );
    int KC = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_KC, cntx );
    int NC = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_NC, cntx );
    // printf( "MR = %d, NR = %d, MC = %d, KC = %d, NC = %d\n", MR, NR, MC, KC, NC );

    // Get sgemm microkernel
    auxinfo_t *auxinfo = ( auxinfo_t * )malloc( sizeof( auxinfo_t ) );
    sgemm_ukr_ft sgemm_ukr = bli_cntx_get_l3_nat_ukr_dt
    (
      BLIS_FLOAT, BLIS_GEMM_UKR, cntx
    );

    // Allocate buffers
    int pool_sizes[] = { MC * KC, KC * NC, MR * NR };
    bao_init_yaconv_pools( 3, pool_sizes, bao_aligned_alloc );
    float *a_buff = bao_get_yaconv_pool( 0 );
    float *b_buff = bao_get_yaconv_pool( 1 );
    float *c_buff = bao_get_yaconv_pool( 2 );

    // Loop indices
    int nc, kc, mc, nr, mr;

    for ( nc = 0; nc < n; nc += NC )
    {
        int nc_curr = bli_min( n - nc, NC );

        for ( kc = 0; kc < k; kc += KC )
        {
            float *beta_ = ( kc == 0 ? &beta : bli_s1 );

            int kc_curr = bli_min( k - kc, KC );

            bao_yaconv_pack
            (
              b + kc * ldb + nc, 1, ldb, b_buff, nc_curr, kc_curr, NR, cntx
            );

            for ( mc = 0; mc < m; mc += MC )
            {
                int mc_curr = bli_min( m - mc, MC );

                bao_yaconv_pack
                (
                  a + mc * lda + kc, lda, 1, a_buff, mc_curr, kc_curr, MR, cntx
                );

                for ( nr = 0; nr < nc_curr; nr += NR )
                {

                    int nr_curr = bli_min( nc_curr - nr, NR );

                    for ( mr = 0; mr < mc_curr; mr += MR )
                    {
                        int mr_curr = bli_min( mc_curr - mr, MR );

                        float *ar = a_buff + mr * kc_curr;
                        float *br = b_buff + nr * kc_curr;
                        float *cr = c + (mc + mr) * ldc + nc + nr;

                        // cr = beta_ * cr + alpha * ar x br
                        if ( ( mr_curr == MR ) && ( nr_curr == NR ) )
                            sgemm_ukr
                            (
                              mr_curr, nr_curr, kc_curr, &alpha, ar, br,
                              beta_, cr, ldc, 1, auxinfo, cntx
                            );
                        else
                        {
                            sgemm_ukr
                            (
                              mr_curr, nr_curr, kc_curr, &alpha, ar, br,
                              bli_s0, c_buff, NR, 1, auxinfo, cntx
                            );
                            bli_sxpbys_mxn
                            (
                              mr_curr, nr_curr, c_buff, NR, 1, beta_, cr, ldc, 1
                            );
                        }
                    }
                }
            }
        }
    }
}
