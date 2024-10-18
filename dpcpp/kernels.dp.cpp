/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.
Copyright (C) 2022 Intel Corporation

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cassert>
#include "defines.h"
#include "calcenergy.h"
#include "GpuData.h"
#include "dpcpp_migration.h"

inline uint64_t llitoulli(int64_t l)
{
	uint64_t u;
	/*
	DPCT1053:0: Migration of device assembly code is not supported.
	*/
	// asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
	u = l;
	return u;
}

inline int64_t ullitolli(uint64_t u)
{
	int64_t l;
	/*
	DPCT1053:1: Migration of device assembly code is not supported.
	*/
	// asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
	l = u;
	return l;
}

#define ATOMICADDI32(pAccumulator, value) \
	sycl::atomic_ref<int, SYCL_ATOMICS_MEMORY_ORDER, SYCL_ATOMICS_MEM_SCOPE, sycl::access::address_space::local_space>(*pAccumulator) += ((int)(value))

#define ATOMICSUBI32(pAccumulator, value) \
	sycl::atomic_ref<int, SYCL_ATOMICS_MEMORY_ORDER, SYCL_ATOMICS_MEM_SCOPE, sycl::access::address_space::local_space>(*pAccumulator) -= ((int)(value))

#define ATOMICADDF32(pAccumulator, value) \
	sycl::atomic_ref<float, SYCL_ATOMICS_MEMORY_ORDER, SYCL_ATOMICS_MEM_SCOPE, sycl::access::address_space::local_space>(*pAccumulator) += ((float)(value))

#define ATOMICSUBF32(pAccumulator, value) \
	sycl::atomic_ref<float, SYCL_ATOMICS_MEMORY_ORDER, SYCL_ATOMICS_MEM_SCOPE, sycl::access::address_space::local_space>(*pAccumulator) -= ((float)(value))

#ifdef USE_XMX
/* Reduction using matrix units */
using namespace sycl::ext::oneapi::experimental::matrix;

// Implementation based on M.Sc. thesis by Gabin Schieffer at KTH:
// "Accelerating a Molecular Docking Application by Leveraging Modern Heterogeneous Computing Systemx"
// https://www.diva-portal.org/smash/get/diva2:1786161/FULLTEXT01.pdf

// If enabled, then using hardcoded inputs
//#define DEBUG_XMX_INPUTS

// Number of rows/cols of a submatrix: M, N, K
constexpr int rowscols_M = 16;
constexpr int rowscols_N = 16;
constexpr int rowscols_K = 16;

constexpr sycl::half HALF_ONE = sycl::half(1.0f);
constexpr sycl::half HALF_ZERO = sycl::half(0.0f);

constexpr sycl::half I4[16] =
{
	HALF_ONE,  HALF_ZERO, HALF_ZERO, HALF_ZERO,
	HALF_ZERO, HALF_ONE,  HALF_ZERO, HALF_ZERO,
	HALF_ZERO, HALF_ZERO, HALF_ONE,  HALF_ZERO,
	HALF_ZERO, HALF_ZERO, HALF_ZERO, HALF_ONE
};

// Printing submatrices contents,
// which have to be previously copied into an array in local memory.
// Enclosing the implementation of print_submatrix()
// within barriers (as initially thought) produces wrong results.
// Such mistake makes sense since print_submatrix() is called if(wi_Id_Wg <= 31).
// Extra sync before printing is not needed as long as
// print_submatrix() is called after joint_matrix functions,
// which are executed by the entire sub_group (i.e., wi_Id_Wg <= 31)
template <typename T>
void print_submatrix (
	sycl::nd_item<3> item,
	const char *msg,
	T *data_to_print
) {
	int wi_Id_Wg = item.get_local_id(2);
	int wg_Id_ND = item.get_group(2);

	if (wg_Id_ND == 0 && wi_Id_Wg == 0) {
		printf("\n%s", msg);
		for (uint i = 0; i < 16; i++) {
			for (uint j = 0; j < 16; j++) {
				if ((j % 16) == 0) {
					printf("\n[Row %2u]: ", i);
				}
				// Printing row-major
				//printf(" %5.3f ", float(data_to_print[i * 16 + j]));
				// Printing column-major
				printf(" %5.3f ", float(data_to_print[j * 16 + i]));
			}
		}
		printf("\n");
	}
}

void print_wi_indexes (
	sycl::nd_item<3> item
) {
	// Identifying global, local, and group ids
	int wi_Id_ND = item.get_global_id(2); // Returns the wi's position in the NDRange (in dimension 2)
	int wi_Id_Wg = item.get_local_id(2); // Returns the wi's position within the current wg (in dimension 2)
	int wg_Id_ND = item.get_group(2); // Returns the wg's position within the overal NDRange (in dimension 2)
	int wg_Size = item.get_local_range(2); // Returns the number of wis per wg (in dimension 2)

	// Identifying sub-groups
	sycl::sub_group sg = item.get_sub_group();
	int sg_Range = sg.get_group_range().get(0); // Returns the number of subgroups within the wg
	int sg_Id_Wg = sg.get_group_id().get(0); // Returns the index of the subgroup within the wg
	int sg_Size = sg.get_local_range().get(0); // Returns the number of wis per subgroup
	int wi_Id_sg = sg.get_local_id(); // Returns the index of the work-item within its subgroup

	/*
	printf("wi_Id_ND: %i, \twi_Id_Wg: %i, \twg_Id_ND: %i, \twg_Size: %i, \tsg_Range: %i, \tsg_Id_Wg: %i, \tsg_Size: %i, \twi_Id_sg: %i\n",
		wi_Id_ND, wi_Id_Wg, wg_Id_ND, wg_Size, sg_Range, sg_Id_Wg, sg_Size, wi_Id_sg);
	*/
}

void fill_Q (
	sycl::nd_item<3> item,
	sycl::half *Q_data
) {
	int wi_Id_Wg = item.get_local_id(2);
	int wg_Size = item.get_local_range().get(2);

	/*
	// Naive implementation: a single work-item fills data in
	if(wi_Id_Wg == 0) {
		for(uint i = 0; i < 4; i++) {	// How many rows (of 4x4 blocks) are there in matrix A?
			for(uint j = 0; j < 4; j++) {	// How many cols (of 4x4 blocks) are there in matrix A?
				for(uint ii = 0; ii < 4; ii++) {
					for(uint jj = 0; jj < 4; jj++) {
						Q_data[4*i + 64*j + ii + 16*jj] = I4[4*ii + jj];
					}
				}
			}
		}
	}
	*/

	// Slightly improved multi-threaded implementation
	for (uint i = wi_Id_Wg; i < 4; i+=wg_Size) {	// How many rows (of 4x4 blocks) are there in matrix A?
		for (uint j = 0; j < 4; j++) {	// How many cols (of 4x4 blocks) are there in matrix A?
			for (uint ii = 0; ii < 4; ii++) {
				for (uint jj = 0; jj < 4; jj++) {
					Q_data[4*i + 64*j + ii + 16*jj] = I4 [4*ii + jj];
				}
			}
		}
	}

	/*
	// Further improved multi-threaded implementation
	// (It didn't provide significant performance improvements -> commented out)
	// Fusing two outer loops into a single one
	// To do that: coeffs = 4i + 64j
	constexpr uint coeffs [16] = {0, 64, 128, 192, 4, 68, 132, 196, 8, 72, 136, 200, 12, 76, 140, 204};
	for (uint k = wi_Id_Wg; k < 16; k+=wg_Size) {
		for (uint ii = 0; ii < 4; ii++) {
			for (uint jj = 0; jj < 4; jj++) {
				Q_data[coeffs[k] + ii + 16*jj] = I4 [4*ii + jj];
			}
		}
	}
	*/

	/*
	print_submatrix<sycl::half>(item, "Q_data [inside fill_Q()]", Q_data);
	*/
}

void fill_identity (
	sycl::nd_item<3> item,
	sycl::half *Q_data
) {
	int wi_Id_Wg = item.get_local_id(2);

	// Naive implementation: a single work-item fills data in
	if(wi_Id_Wg == 0) {
		for(uint i = 0; i < 16; i++) {
			for(uint j = 0; j < 16; j++) {
				if (i == j) {
					Q_data[16*i + j] = HALF_ONE;
				}
				else {
					Q_data[16*i + j] = HALF_ZERO;
				}
			}
		}
	}

	/*
	print_submatrix<sycl::half>(item, "Q_data [inside fill_identity()]", Q_data);
	*/
}

using T_jm_a = joint_matrix<sycl::sub_group, sycl::half, use::a, rowscols_M, rowscols_K, layout::col_major>;
using T_jm_b = joint_matrix<sycl::sub_group, sycl::half, use::b, rowscols_K, rowscols_N, layout::col_major>;
using T_jm_acc = joint_matrix<sycl::sub_group, sycl::half, use::accumulator, rowscols_M, rowscols_N>;

// The most elegant way to print a "use::b" sub_Input_b matrix would be
// to copy such matrix into any "use::accumulator" matrix,
// which in turn would be stored into shared memory, and then finally printed.
//
// Copying matrices contents via joint_matrix_copy()
// from "use::b" into "use::accumulator" doesn't work in oneAPI v2024.2.1.
// Thus, "use::b" sub_Input_b matrix is moved to shared memory via matrix multiply-add,
// where sub_Identity_a is used as a temporal identity matrix,
// and sub_Acc is initialized with zeros but ends up storing sub_Input_b contents.
void move_matrix_a_to_acc (
	sycl::nd_item<3> item,
	sycl::half *tmp,
	T_jm_a &sub_Input_a,
	T_jm_acc &sub_Acc
){
	sycl::sub_group sg = item.get_sub_group();

	// Loading identity values to sub_Identity
	fill_identity(item, tmp);
	T_jm_b sub_Identity_b;
	joint_matrix_load(sg, sub_Identity_b, sycl::local_ptr<sycl::half>(tmp), 16);

	// Initializing sub_Acc with zeros
	joint_matrix_fill(sg, sub_Acc, HALF_ZERO);

	// sub_Acc <- sub_Input_a x sub_Identity_b + sub_Acc
	joint_matrix_mad(sg, sub_Acc, sub_Input_a, sub_Identity_b, sub_Acc);
}

void move_matrix_b_to_acc (
	sycl::nd_item<3> item,
	sycl::half *tmp,
	T_jm_b &sub_Input_b,
	T_jm_acc &sub_Acc
){
	sycl::sub_group sg = item.get_sub_group();

	// Loading identity values to sub_Identity
	fill_identity(item, tmp);
	T_jm_a sub_Identity_a;
	joint_matrix_load(sg, sub_Identity_a, sycl::local_ptr<sycl::half>(tmp), 16);

	// Initializing sub_Acc with zeros
	joint_matrix_fill(sg, sub_Acc, HALF_ZERO);

	// sub_Acc <- sub_Identity_a x sub_Input_b + sub_Acc
	joint_matrix_mad(sg, sub_Acc, sub_Identity_a, sub_Input_b, sub_Acc);
}


// Implementation based on MSc thesis at KTH:
// "Accelerating a Molecular Docking Application by Leveraging Modern Heterogeneous Computing Systemx"
// https://www.diva-portal.org/smash/get/diva2:1786161/FULLTEXT01.pdf
//
// We consider that a CUDA fragment is equivalent to a SYCL submatrix
//
// Compilation: make DEVICE=XeGPU PLATFORM=NvGPU XMX=ON TESTLS=ad NUMWI=64 test
void reduce_via_matrix_units (
	sycl::nd_item<3> item,
	sycl::half *data_to_be_reduced,
	sycl::half *Q_data,
	sycl::half *tmp
) {
	int wi_Id_Wg = item.get_local_id(2);
	int wg_Id_ND = item.get_group(2);
	sycl::sub_group sg = item.get_sub_group();

	item.barrier(SYCL_MEMORY_SPACE);

	///*
	print_wi_indexes(item);
	//*/

	// Only one sub-group performs reduction
	if(wi_Id_Wg <= 31) {
		fill_Q(item, Q_data);

		// Declaring and filling submatrices
		T_jm_b sub_P;
		T_jm_acc sub_V;
		T_jm_a sub_Q;
		T_jm_b sub_W;
		T_jm_acc sub_C;
		joint_matrix_fill(sg, sub_P, HALF_ONE); // P: only ones
		joint_matrix_fill(sg, sub_V, HALF_ZERO); // Output: initialize to zeros
		joint_matrix_fill(sg, sub_C, HALF_ZERO); // Final result
		joint_matrix_load(sg, sub_Q, sycl::local_ptr<sycl::half>(Q_data), 16);

		// 1. Accumulate the values: V <- AP + V
		for(uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK) / (16 * 16);  i++) {
			const uint offset = i * 16 * 16;

			/*
			if (wg_Id_ND == 0 && wi_Id_Wg == 0) {
				printf("\ni = %d, tripcount= %d, offset = %d ", i, (4 * NUM_OF_THREADS_PER_BLOCK) / (16 * 16), offset);
			}
			*/

			/*
			print_submatrix<sycl::half>(item, "data_to_be_reduced [inside main loop]", data_to_be_reduced);
			*/

			T_jm_a sub_A;
			joint_matrix_load(sg, sub_A, sycl::local_ptr<sycl::half>(data_to_be_reduced + offset), 16);

			/*
			// Printing sub_A y sub_P
			T_jm_acc sub_Acc;
			move_matrix_a_to_acc(item, tmp, sub_A, sub_Acc);
			joint_matrix_store(sg, sub_Acc, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
			print_submatrix<sycl::half>(item, "sub_A", tmp);

			move_matrix_b_to_acc(item, tmp, sub_P, sub_Acc);
			joint_matrix_store(sg, sub_Acc, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
			print_submatrix<sycl::half>(item, "sub_P", tmp);
			*/

			/*
			// Printing sub_V (before mad)
			joint_matrix_store(sg, sub_V, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
			print_submatrix<sycl::half>(item, "sub_V (before mad)", tmp);
			*/

			//sub_V = joint_matrix_mad(sg, sub_A, sub_P, sub_V);	// 2024.1
			joint_matrix_mad(sg, sub_V, sub_A, sub_P, sub_V);	// 2024.2.1

			/*
			// Printing sub_V (after mad)
			joint_matrix_store(sg, sub_V, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
			print_submatrix<sycl::half>(item, "sub_V (after mad)", tmp);
			*/
		}

		// W <- V (required since we need V as a "use::b")
		joint_matrix_store(sg, sub_V, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
		joint_matrix_load(sg, sub_W, sycl::local_ptr<sycl::half>(tmp), 16);

		/*
		// Printing sub_Q y sub_W
		T_jm_acc sub_Acc2;
		move_matrix_a_to_acc(item, tmp, sub_Q, sub_Acc2);
		joint_matrix_store(sg, sub_Acc2, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
		print_submatrix<sycl::half>(item, "sub_Q", tmp);

		move_matrix_b_to_acc(item, tmp, sub_W, sub_Acc2);
		joint_matrix_store(sg, sub_Acc2, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
		print_submatrix<sycl::half>(item, "sub_W", tmp);
		*/

		/*
		// Printing sub_C (before mad)
		joint_matrix_store(sg, sub_C, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
		print_submatrix<sycl::half>(item, "sub_C (before mad)", tmp);
		*/

		// 2. Perform line sum: C <- QW + C (zero)
		//sub_C = joint_matrix_mad(sg, sub_Q, sub_W, sub_C);	// 2024.1
		joint_matrix_mad(sg, sub_C, sub_Q, sub_W, sub_C);	// 2024.2.1

		/*
		// Printing sub_C (after mad)
		joint_matrix_store(sg, sub_C, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
		print_submatrix<sycl::half>(item, "sub_C", tmp);
		*/

		// 3. Store result in shared memory
		joint_matrix_store(sg, sub_C, sycl::local_ptr<sycl::half>(data_to_be_reduced), 16, layout::col_major);
	}

	item.barrier(SYCL_MEMORY_SPACE);
}

// Validating inputs
using namespace sycl::ext::oneapi::experimental;

// Compile-Time Query of matrix sizes and types configurations

// References in oneAPI 2024.2.1
//
// ---------------------------------------------
// Intel
// Architectures recognized in oneAPI 2024.2.1:
// /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/ext/oneapi/experimental/device_architecture.hpp
// intel_gpu_bdw // Intel(R) microarchitecture code name Broadwell
// intel_gpu_skl // Intel(R) microarchitecture code name Skylake
// intel_gpu_kbl // Kaby Lake
// intel_gpu_cfl // Coffee Lake
// intel_gpu_apl // Apollo Lake
// intel_gpu_bxt = intel_gpu_apl // Broxton
// intel_gpu_glk // Gemini Lake
// intel_gpu_whl // Whiskey Lake
// intel_gpu_aml // Amber Lake
// intel_gpu_cml // Comet Lake
// intel_gpu_icllp // Ice Lake
// intel_gpu_icl = intel_gpu_icllp // Ice Lake
// intel_gpu_ehl // Elkhart Lake
// intel_gpu_jsl = intel_gpu_ehl // Jasper Lake
// intel_gpu_tgllp // Tiger Lake
// intel_gpu_tgl = intel_gpu_tgllp // Tiger Lake
// intel_gpu_rkl // Rocket Lake
// intel_gpu_adl_s // Alder Lake S
// intel_gpu_rpl_s = intel_gpu_adl_s // Raptor Lake
// intel_gpu_adl_p // Alder Lake P
// intel_gpu_adl_n // Alder Lake N
// intel_gpu_dg1 // DG1
// intel_gpu_acm_g10 // Alchemist G10
// intel_gpu_dg2_g10 = intel_gpu_acm_g10 // Alchemist G10
// intel_gpu_acm_g11 // Alchemist G11
// intel_gpu_dg2_g11 = intel_gpu_acm_g11 // Alchemist G11
// intel_gpu_acm_g12 // Alchemist G12
// intel_gpu_dg2_g12 = intel_gpu_acm_g12 // Alchemist G12
// intel_gpu_pvc // Ponte Vecchio
// intel_gpu_pvc_vg // Ponte Vecchio VG
// intel_gpu_mtl_u // Meteor Lake U
// intel_gpu_mtl_s = intel_gpu_mtl_u // Meteor Lake S
// intel_gpu_arl_u = intel_gpu_mtl_u // Arrow Lake U
// intel_gpu_arl_s = intel_gpu_mtl_u // Arrow Lake S
// intel_gpu_mtl_h // Meteor Lake H
// intel_gpu_arl_h // Arrow Lake H
// intel_gpu_bmg_g21 // Battlemage G21
// intel_gpu_lnl_m // Lunar Lake
//
// Architectures with support for static queries in oneAPI 2024.2.1:
// /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/ext/oneapi/matrix/static-query-use.hpp
// intel_gpu_dg2_g10
// intel_gpu_dg2_g11
// intel_gpu_dg2_g12
// intel_gpu_pvc
//
// ---------------------------------------------
// NVIDIA
// Architectures recognized in oneAPI 2024.2.1:
// /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/ext/oneapi/experimental/device_architecture.hpp
// nvidia_gpu_sm_50
// nvidia_gpu_sm_52
// nvidia_gpu_sm_53
// nvidia_gpu_sm_60
// nvidia_gpu_sm_61
// nvidia_gpu_sm_62
// nvidia_gpu_sm_70
// nvidia_gpu_sm_72
// nvidia_gpu_sm_75
// nvidia_gpu_sm_80
// nvidia_gpu_sm_86
// nvidia_gpu_sm_87
// nvidia_gpu_sm_89
// nvidia_gpu_sm_90
//
// Architectures with support for static queries in oneAPI 2024.2.1:
// /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/ext/oneapi/matrix/static-query-use.hpp
// nvidia_gpu_sm_70
// nvidia_gpu_sm_72
// nvidia_gpu_sm_80
//
// ---------------------------------------------
// Custom check
/*
constexpr architecture nvidia_gpu_family[3] = {
	architecture::nvidia_gpu_sm_70,
	architecture::nvidia_gpu_sm_72,
	architecture::nvidia_gpu_sm_80
};
*/
// Intel (all configuration below are invalid -> compilation failing)
/*
using myparams_intel_gpu_dg2_g10 = matrix_params<architecture::intel_gpu_dg2_g10, sycl::half, sycl::half, sycl::half, sycl::half, rowscols_M, rowscols_N, rowscols_K>;
myparams_intel_gpu_dg2_g10 test_params_intel_gpu_dg2_g10; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_intel_gpu_dg2_g11 = matrix_params<architecture::intel_gpu_dg2_g11, sycl::half, sycl::half, sycl::half, sycl::half, rowscols_M, rowscols_N, rowscols_K>;
myparams_intel_gpu_dg2_g11 test_params_intel_gpu_dg2_g11; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_intel_gpu_dg2_g12 = matrix_params<architecture::intel_gpu_dg2_g12, sycl::half, sycl::half, sycl::half, sycl::half, rowscols_M, rowscols_N, rowscols_K>;
myparams_intel_gpu_dg2_g12 test_params_intel_gpu_dg2_g12; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_intel_gpu_pvc = matrix_params<architecture::intel_gpu_pvc, sycl::half, sycl::half, sycl::half, sycl::half, rowscols_M, rowscols_N, rowscols_K>;
myparams_intel_gpu_pvc test_params_intel_gpu_pvc; // Checking with object definition because internal asserts happen at struct instantiation!
*/

// NVIDIA (all configurations below pass -> these are replaced with runtime query)
/*
using myparams_nvidia_gpu_sm_70 = matrix_params<architecture::nvidia_gpu_sm_70, sycl::half, sycl::half, sycl::half, sycl::half, rowscols_M, rowscols_N, rowscols_K>;
myparams_nvidia_gpu_sm_70 test_params_nvidia_gpu_sm_70; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_nvidia_gpu_sm_72 = matrix_params<architecture::nvidia_gpu_sm_72, sycl::half, sycl::half, sycl::half, sycl::half, rowscols_M, rowscols_N, rowscols_K>;
myparams_nvidia_gpu_sm_72 test_params_nvidia_gpu_sm_72; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_nvidia_gpu_sm_80 = matrix_params<architecture::nvidia_gpu_sm_80, sycl::half, sycl::half, sycl::half, sycl::half, rowscols_M, rowscols_N, rowscols_K>;
myparams_nvidia_gpu_sm_80 test_params_nvidia_gpu_sm_80; // Checking with object definition because internal asserts happen at struct instantiation!
*/
/* Reduction using matrix units */
#endif

static dpct::constant_memory<GpuData, 0> cData;
static GpuData cpuData;

void SetKernelsGpuData(GpuData *pData) try
{
	int status;
	status = (dpct::get_default_queue().memcpy(cData.get_ptr(), pData, sizeof(GpuData)).wait(), 0);
	RTERROR(status, "SetKernelsGpuData copy to cData failed");
	memcpy(&cpuData, pData, sizeof(GpuData));
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
			  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

void GetKernelsGpuData(GpuData *pData) try
{
	int status;
	status = (dpct::get_default_queue().memcpy(pData, cData.get_ptr(), sizeof(GpuData)).wait(), 0);
	RTERROR(status, "GetKernelsGpuData copy From cData failed");
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
			  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

// Kernel files
#include "calcenergy.dp.cpp"
#include "calcMergeEneGra.dp.cpp"
#include "auxiliary_genetic.dp.cpp"
#include "kernel1.dp.cpp"
#include "kernel2.dp.cpp"
#include "kernel3.dp.cpp"
#include "kernel4.dp.cpp"
#include "kernel_ad.dp.cpp"
#include "kernel_adam.dp.cpp"
