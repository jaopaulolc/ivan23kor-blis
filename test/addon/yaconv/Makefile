#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2017 - 2020, Advanced Micro Devices, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# Makefile
#
# Ivan Korostelev
#
# Makefile for standalone yaconv BLIS addon test drivers.
#

.PHONY: all conv gemm \
				clean


#
# --- Determine makefile fragment location -------------------------------------
#

# Comments:
# - DIST_PATH is assumed to not exist if BLIS_INSTALL_PATH is given.
# - We must use recursively expanded assignment for LIB_PATH and INC_PATH in
#   the second case because CONFIG_NAME is not yet set.
ifneq ($(strip $(BLIS_INSTALL_PATH)),)
LIB_PATH   := $(BLIS_INSTALL_PATH)/lib
INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
SHARE_PATH := $(BLIS_INSTALL_PATH)/share/blis
else
DIST_PATH  := ../../../
LIB_PATH    = $(DIST_PATH)/lib/$(CONFIG_NAME)
INC_PATH    = $(DIST_PATH)/include/$(CONFIG_NAME)
SHARE_PATH := $(DIST_PATH)/
endif


#
# --- Include common makefile definitions --------------------------------------
#

# Include the common makefile fragment.
-include $(SHARE_PATH)/common.mk

#
# --- BLAS and LAPACK implementations ------------------------------------------
#

# OpenBLAS
LIBOPENBLAS_LINK := -lopenblas


#
# --- General build definitions ------------------------------------------------
#

CFLAGS := $(call get-user-cflags-for,$(CONFIG_NAME))

# Repeat count in sgemm and yaconv
REPEAT ?= 10
CFLAGS += -DREPEAT=$(REPEAT)


# --Targets --

all: conv

conv: test_yaconv.x test_im2col_blis.x test_im2col_openblas.x

gemm: test_bao_sgemm.x test_sgemm_blis.x test_sgemm_openblas.x

# --Object file rules --

test_bao_sgemm.o: test_sgemm.c
	$(CC) $(CFLAGS) -c $< -o $@

test_sgemm_blis.o: test_sgemm.c
	$(CC) $(CFLAGS) -DBLIS -c $< -o $@

test_sgemm_openblas.o: test_sgemm.c
	$(CC) $(CFLAGS) -DOPENBLAS -c $< -o $@

test_yaconv.o: test_conv.c
	$(CC) $(CFLAGS) -c $< -o $@

test_im2col_blis.o: test_conv.c
	$(CC) $(CFLAGS) -DIM2COL_BLIS -c $< -o $@

test_im2col_openblas.o: test_conv.c
	$(CC) $(CFLAGS) -DIM2COL_OPENBLAS -c $< -o $@


# --Executable file rules --

test_bao_sgemm.x: test_bao_sgemm.o $(LIBBLIS_LINK)
	$(LINKER) $< $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_sgemm_blis.x: test_sgemm_blis.o $(LIBBLIS_LINK)
	$(LINKER) $< $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_sgemm_openblas.x: test_sgemm_openblas.o $(LIBBLIS_LINK)
	$(LINKER) $< $(LIBOPENBLAS_LINK) $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_yaconv.x: test_yaconv.o $(LIBBLIS_LINK)
	$(LINKER) $< $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_im2col_blis.x: test_im2col_blis.o $(LIBBLIS_LINK)
	$(LINKER) $< $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_im2col_openblas.x: test_im2col_openblas.o $(LIBBLIS_LINK)
	$(LINKER) $< $(LIBBLIS_LINK) $(LIBOPENBLAS_LINK) $(LDFLAGS) -o $@

# -- Clean rules --

clean:
	- $(RM_F) *.o *.x
