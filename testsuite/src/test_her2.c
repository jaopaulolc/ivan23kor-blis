/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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

#include "blis2.h"
#include "test_libblis.h"


// Static variables.
static char*     op_str                    = "her2";
static char*     o_types                   = "vvm";  // x y a
static char*     p_types                   = "ucc";  // uploa conjx conjy
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_her2_deps( test_params_t* params,
                             test_op_t*     op );

void libblis_test_her2_experiment( test_params_t* params,
                                   test_op_t*     op,
                                   mt_impl_t      impl,
                                   num_t          datatype,
                                   char*          pc_str,
                                   char*          sc_str,
                                   dim_t          p_cur,
                                   double*        perf,
                                   double*        resid );

void libblis_test_her2_impl( mt_impl_t impl,
                             obj_t*    alpha,
                             obj_t*    x,
                             obj_t*    y,
                             obj_t*    a );

void libblis_test_her2_check( obj_t*  alpha,
                              obj_t*  x,
                              obj_t*  y,
                              obj_t*  a,
                              obj_t*  a_orig,
                              double* resid );



void libblis_test_her2_deps( test_params_t* params, test_op_t* op )
{
	libblis_test_randv( params, &(op->ops->randv) );
	libblis_test_randm( params, &(op->ops->randm) );
	libblis_test_fnormv( params, &(op->ops->fnormv) );
	libblis_test_subv( params, &(op->ops->subv) );
	libblis_test_copym( params, &(op->ops->copym) );
	libblis_test_scal2v( params, &(op->ops->scal2v) );
	libblis_test_dotv( params, &(op->ops->dotv) );
	libblis_test_gemv( params, &(op->ops->gemv) );
}



void libblis_test_her2( test_params_t* params, test_op_t* op )
{

	// Return early if this test has already been done.
	if ( op->test_done == TRUE ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_her2_deps( params, op );

	// Execute the test driver for each implementation requested.
	if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( params,
		                        op,
		                        BLIS_TEST_SEQ_FRONT_END,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_her2_experiment );
	}
}



void libblis_test_her2_experiment( test_params_t* params,
                                   test_op_t*     op,
                                   mt_impl_t      impl,
                                   num_t          datatype,
                                   char*          pc_str,
                                   char*          sc_str,
                                   dim_t          p_cur,
                                   double*        perf,
                                   double*        resid )
{
	unsigned int n_repeats = params->n_repeats;
	unsigned int i;

	double       time_min  = 1e9;
	double       time;

	dim_t        m;

	uplo_t       uploa;
	conj_t       conjx, conjy;

	obj_t        alpha, x, y, a;
	obj_t        a_save;


	// Map the dimension specifier to an actual dimension.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Map parameter characters to BLIS constants.
	bl2_param_map_char_to_blis_uplo( pc_str[0], &uploa );
	bl2_param_map_char_to_blis_conj( pc_str[1], &conjx );
	bl2_param_map_char_to_blis_conj( pc_str[2], &conjy );

	// Create test scalars.
	bl2_obj_init_scalar( datatype, &alpha );

	// Create test operands (vectors and/or matrices).
	libblis_test_vobj_create( params, datatype,
		                      sc_str[0], m,    &x );
	libblis_test_vobj_create( params, datatype,
		                      sc_str[1], m,    &y );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
		                      sc_str[2], m, m, &a );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
		                      sc_str[2], m, m, &a_save );

	// Set alpha.
	//bl2_copysc( &BLIS_MINUS_ONE, &alpha );
	bl2_setsc( -1.0, 1.0, &alpha );

	// Randomize x and y.
	bl2_randv( &x );
	bl2_randv( &y );

	// Set the structure and uplo properties of A.
	bl2_obj_set_struc( BLIS_HERMITIAN, a );
	bl2_obj_set_uplo( uploa, a );

	// Randomize A, make it densely Hermitian, and zero the unstored triangle
	// to ensure the implementation is reads only from the stored region.
	bl2_randm( &a );
	bl2_mkherm( &a );
	bl2_mktrim( &a );

	// Save A and set its structure and uplo properties.
	bl2_obj_set_struc( BLIS_HERMITIAN, a_save );
	bl2_obj_set_uplo( uploa, a_save );
	bl2_copym( &a, &a_save );

	// Apply the remaining parameters.
	bl2_obj_set_conj( conjx, x );
	bl2_obj_set_conj( conjy, y );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bl2_copym( &a_save, &a );

		time = bl2_clock();

		libblis_test_her2_impl( impl, &alpha, &x, &y, &a );

		time_min = bl2_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m * m ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bl2_obj_is_complex( a ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_her2_check( &alpha, &x, &y, &a, &a_save, resid );

	// Free the test objects.
	bl2_obj_free( &x );
	bl2_obj_free( &y );
	bl2_obj_free( &a );
	bl2_obj_free( &a_save );
}



void libblis_test_her2_impl( mt_impl_t impl,
                             obj_t*    alpha,
                             obj_t*    x,
                             obj_t*    y,
                             obj_t*    a )
{
	switch ( impl )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bl2_her2( alpha, x, y, a );
		break;

		default:
		libblis_test_printf_error( "Invalid implementation type.\n" );
	}
}



void libblis_test_her2_check( obj_t*  alpha,
                              obj_t*  x,
                              obj_t*  y,
                              obj_t*  a,
                              obj_t*  a_orig,
                              double* resid )
{
	num_t  dt      = bl2_obj_datatype( *a );
	num_t  dt_real = bl2_obj_datatype_proj_to_real( *a );

	dim_t  m_a     = bl2_obj_length( *a );

	obj_t  xh, yh, alphac;
	obj_t  t, v, w1, w2;
	obj_t  tau, rho, norm;

	double junk;

	//
	// Pre-conditions:
	// - x is randomized.
	// - y is randomized.
	// - a is randomized and Hermitian.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   A := A_orig + alpha * conjx(x) * conjy(y)^H + conj(alpha) * conjy(y) * conjx(x)^H
	//
	// is functioning correctly if
	//
	//   fnorm( v - w )
	//
	// is negligible, where
	//
	//   v = A * t
	//   w = ( A_orig + alpha * conjx(x) * conjy(y)^H + conj(alpha) * conjy(y) * conjx(x)^H ) * t
	//     = A_orig * t + alpha * conjx(x) * conjy(y)^H * t + conj(alpha) * conjy(y) * conjx(x)^H * t
	//     = A_orig * t + alpha * conjx(x) * conjy(y)^H * t + conj(alpha) * conjy(y) * rho
	//     = A_orig * t + alpha * conjx(x) * conjy(y)^H * t + w1
	//     = A_orig * t + alpha * conjx(x) * rho            + w1
	//     = A_orig * t + w2                                + w1
	//

	bl2_mkherm( a );
	bl2_mkherm( a_orig );
	bl2_obj_set_struc( BLIS_GENERAL, *a );
	bl2_obj_set_struc( BLIS_GENERAL, *a_orig );
	bl2_obj_set_uplo( BLIS_DENSE, *a );
	bl2_obj_set_uplo( BLIS_DENSE, *a_orig );

	bl2_obj_init_scalar( dt,      &tau );
	bl2_obj_init_scalar( dt,      &rho );
	bl2_obj_init_scalar( dt,      &alphac );
	bl2_obj_init_scalar( dt_real, &norm );

	bl2_obj_create( dt, m_a, 1, 0, 0, &t );
	bl2_obj_create( dt, m_a, 1, 0, 0, &v );
	bl2_obj_create( dt, m_a, 1, 0, 0, &w1 );
	bl2_obj_create( dt, m_a, 1, 0, 0, &w2 );

	bl2_obj_alias_with_conj( BLIS_CONJUGATE, *x, xh );
	bl2_obj_alias_with_conj( BLIS_CONJUGATE, *y, yh );
	bl2_obj_alias_with_conj( BLIS_CONJUGATE, *alpha, alphac );

	bl2_setsc( 1.0/( double )m_a, -1.0/( double )m_a, &tau );
	bl2_setv( &tau, &t );

	bl2_gemv( &BLIS_ONE, a, &t, &BLIS_ZERO, &v );

	bl2_dotv( &xh, &t, &rho );
	bl2_mulsc( &alphac, &rho );
	bl2_scal2v( &rho, y, &w1 );

	bl2_dotv( &yh, &t, &rho );
	bl2_mulsc( alpha, &rho );
	bl2_scal2v( &rho, x, &w2 );

	bl2_addv( &w2, &w1 );

	bl2_gemv( &BLIS_ONE, a_orig, &t, &BLIS_ONE, &w1 );

	bl2_subv( &w1, &v );
	bl2_fnormv( &v, &norm );
	bl2_getsc( &norm, resid, &junk );

	bl2_obj_free( &t );
	bl2_obj_free( &v );
	bl2_obj_free( &w1 );
	bl2_obj_free( &w2 );
}
