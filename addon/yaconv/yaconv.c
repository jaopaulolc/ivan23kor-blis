/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2021 - 2022, Ivan Korostelev

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

// BLIS structures, block sizes and buffers
static cntx_t *cntx = NULL;
static auxinfo_t *auxinfo = NULL;
static int MR, NR, MC, KC, NC;
static float *filter_buf = NULL, *image_buf = NULL, *output_buf = NULL;

// GEMM microkernel
static void sgemm_ukr(int mr, int nr, int k,
                      float *alpha, float *a, float *b,
                      float *beta, float *c, int rsc, int csc) {
  bli_sgemm_ukernel(mr, nr, k, alpha, a, b, beta, c, rsc, csc, auxinfo, cntx);
}

// Convenience page-aligned alloc with return check
static float *aligned_alloc(int size) {
    float *data = NULL;
    int ret = posix_memalign((void**)&data, BLIS_PAGE_SIZE,
                             size * sizeof(float));

    if (ret == 0)
      return data;

    fprintf(stderr, "\033[31mAllocation error in posix_memalign!\033[0m\n");
    exit(ret);
}

// Packing functions
static void yaconv_pack(float *src, int rss, int css,
                        float *dst, int MN, int k, int MNR) {
  for (int mn = 0; mn < MN; mn += MNR)
    /* If a packing microkernel for this register block size is found,
       use it. Otherwise, use scal2m */
    bli_spackm_cxk(BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS,
                   bli_min(MN - mn, MNR), MNR, k, k,
                   bli_s1, src + mn * rss, rss, css,
                   dst + mn * k, MNR,
                   cntx);
}

// Extra size functions
static int yaconv_extra_size_after(int H, int FH, int PH, int OW, int M) {
  cntx_t *cntx = bli_gks_query_cntx();
  int NR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);

  int extra_h = 0;
  if (H % NR)
    extra_h = NR - H % NR;

  return (extra_h + FH - 1 - PH) * OW * M;
}

BLIS_EXPORT_ADDON int yaconv_extra_size_before(int FH, int PH, int OW, int M) {
  return (FH - 1 - PH) * OW * M;
}

BLIS_EXPORT_ADDON int yaconv_extra_size(int H, int FH, int PH, int OW, int M) {
  return yaconv_extra_size_before(FH, PH, OW, M)
         + yaconv_extra_size_after(H, FH, PH, OW, M);
}

// The main yaconv function that computes convolution on a signle image
static void yaconv_single_image(float *image, int H, int W, int C,
                                float *filter, int FH, int FW, int M,
                                float *output, int PH, int PW) {

  // First, compute the spatial width of the output
  const int OH = H + 2 * PH - FH + 1;
  const int OW = W + 2 * PW - FW + 1;

  output += yaconv_extra_size_before(FH, PH, OW, M);
  bli_ssetv(BLIS_NO_CONJUGATE, OH * OW * M, bli_s0, output, 1);

  for (int nc = 0; nc < H; nc += NC) {

    int nc_curr = bli_min(H - nc, NC);

    yaconv_pack(image + nc * W * C, W * C, 1, image_buf, nc_curr, W * C, NR);

    for (int fh = 0; fh < FH; ++fh) {
      for (int m = 0; m < M; m += MC) {

        int mc_curr = bli_min(M - m, MC);

        for (int kc = 0; kc < FW * C; kc += KC) {

          int kc_curr = bli_min(FW * C - kc, KC);
          yaconv_pack(filter + (fh * FW * C + kc) * M + m, 1, M,
                      filter_buf, mc_curr, kc_curr, MR);

          for (int nr = 0; nr < nc_curr; nr += NR) {
            for (int ow = 0; ow < OW; ++ow) {

              int image_start = (ow - PW) * C + kc;
              int image_end = bli_min(W * C, image_start + kc_curr);

              float *ar = filter_buf;
              if (image_start < 0) {
                ar -= image_start * MR;
                image_start = 0;
              }

              int K = image_end - image_start;
              if (K <= 0)
                continue;

              float *br = image_buf + nr * W * C + image_start * NR;
              float *cr = output + ((nc + nr - fh + PH) * OW + ow) * M + m;

              for (int mr = 0; mr < mc_curr; mr += MR) {
                if (mr + MR <= mc_curr)
                  sgemm_ukr(MR, NR, K, bli_s1, ar, br,
                            bli_s1, cr, 1, OW * M);
                else {
                  sgemm_ukr(MR, NR, K, bli_s1, ar, br,
                            bli_s0, output_buf, NR, 1);
                  bli_sxpbys_mxn(mc_curr - mr, NR, output_buf, NR, 1,
                                 bli_s1, cr, 1, OW * M);
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

// This function performs intitialization of BLIS structures, block sizes and
// buffer allocation, then calls yaconv implementation for each image
static void yaconv_init_once(int W, int FW, int C) {
  if (filter_buf != NULL)
    return;

  // Fetch BLIS structures
  cntx = bli_gks_query_cntx();
  auxinfo = (auxinfo_t *)malloc(sizeof(auxinfo_t));

  // Get blocksizes
  MR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
  NR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
  MC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
  KC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);
  NC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);

  // Adjust KC and NC at run-time so that the packed image buffer fits in L3
  KC = bli_min(KC, FW * C);
  NC = KC * NC / W / C;

  // Expand NC to the nearest multiple of NR, as NC is chosen too conservatively
  // for GEMM in BLIS
  NC += (NC % NR) ? NR - NC % NR : 0;

  // printf("MR = %d, NR = %d, MC = %d, KC = %d, NC = %d\n", MR, NR, MC, KC, NC);

  // Compute buffer offsets
  int page_size_minus_one = BLIS_PAGE_SIZE * sizeof(float) - 1;
  int image_buf_off = (MC * KC + page_size_minus_one) & ~page_size_minus_one;
  int output_buf_off = (W * C * NC + page_size_minus_one) & ~page_size_minus_one;

  // Allocate buffer space
  filter_buf = aligned_alloc(image_buf_off + output_buf_off + MR * NR);
  image_buf = filter_buf + image_buf_off;
  output_buf = image_buf + output_buf_off;
}

void yaconv_deinit() {
  // All buffers are actually at different offsets within one
  free(filter_buf);
}

// This function performs intitialization of BLIS structures, block sizes and
// buffer allocation, then calls yaconv implementation for each image
BLIS_EXPORT_ADDON void yaconv(float **images, int N, int H, int W, int C,
                              float *filter, int FH, int FW, int M,
                              float **outputs, int PH, int PW) {
  yaconv_init_once(W, FW, C);

  // Run yaconv on each image
  for (int i = 0; i < N; ++i)
    yaconv_single_image(images[i], H, W, C, filter, FH, FW, M,
                        outputs[i], PH, PW);

  yaconv_deinit();
}
