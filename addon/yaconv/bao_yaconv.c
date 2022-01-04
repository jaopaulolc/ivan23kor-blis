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
#include "bao_util.h"

BLIS_EXPORT_ADDON void bao_yaconv
     (
       float *filter, float *image, float *output,
       int C, int H, int W, int M, int FH, int FW, int PH, int PW
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

    // Compute runtime-based slice of the image that fits in L3
    NC = MC * NC / W / C;
    if ( NC % NR != 0 )
        NC += NR - NC % NR;

    // Allocate buffers
    int pool_sizes[] = { MC * KC, W * C * NC, MR * NR };
    bao_init_yaconv_pools( 3, pool_sizes, bao_aligned_alloc );
    float *filter_buf = bao_get_yaconv_pool( 0 );
    float *image_buf  = bao_get_yaconv_pool( 1 );
    float *output_buf = bao_get_yaconv_pool( 2 );

    // Output sizes
    const int OW = W + 2 * PW - FW + 1;

    for ( int nc = 0; nc < H; nc += NC )
    {
        int nc_curr = bli_min( H - nc, NC );

        // bli_sprintm
        // (
        //   "image:", W * C, H, image, 1, W * C, "%.0f", "=====================\n"
        // );
        bao_yaconv_pack
        (
          image + nc * W * C, W * C, 1, image_buf, nc_curr, W * C, NR, cntx
        );
        for ( int fh = 0; fh < FH; ++fh )
        {
            for ( int m = 0; m < M; m += MC )
            {
                int mc_curr = bli_min( M - m, MC );
                for ( int kc = 0; kc < FW * C; kc += KC )
                {
                    int kc_curr = bli_min( FW * C - kc, KC );
                    bao_yaconv_pack
                    (
                      filter + ( m * FH + fh ) * FW * C + kc, FH * FW * C, 1,
                      filter_buf, mc_curr, kc_curr, MR, cntx
                    );
                    for ( int nr = 0; nr < nc_curr; nr += NR )
                    {
                        for ( int ow = 0; ow < OW; ++ow )
                        {
                            int image_start = kc + ( ow - PW ) * C;
                            int image_end = bli_min
                            (
                              W * C, image_start + kc_curr
                            );

                            float *ar = filter_buf;
                            if ( image_start < 0 )
                            {
                              ar -= image_start * MR;
                              image_start = 0;
                            }

                            int K = image_end - image_start;
                            if ( K <= 0 )
                              continue;

                            float *br = image_buf + nr * W * C + image_start * NR;
                            float *cr = output + ( ( nc + nr - fh + PH ) * OW + ow ) * M + m;

                            for ( int mr = 0; mr < mc_curr; mr += MR )
                            {
                                if ( mr + MR <= mc_curr )
                                {
                                    sgemm_ukr
                                    (
                                      MR, NR, K, bli_s1, ar, br,
                                      bli_s1, cr, 1, OW * M, auxinfo, cntx
                                    );
                                }
                                else
                                {
                                    sgemm_ukr
                                    (
                                      mc_curr - mr, NR, K, bli_s1, ar, br,
                                      bli_s0, output_buf, NR, 1, auxinfo, cntx
                                    );
                                    bli_sxpbys_mxn
                                    (
                                      mc_curr - mr, NR, output_buf, NR, 1,
                                      bli_s1, cr, 1, OW * M
                                    );
                                }

                                ar += MR * kc_curr;
                                cr += MR;
                            }
                        }
                    }
                }
            }
        }
    }
}
