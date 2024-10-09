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

#define WARPMINIMUMEXCHANGE(tgx, v0, k0, mask)	\
{	\
	float v1 = v0;	\
	int k1 = k0;	\
	int otgx = tgx ^ mask;	\
	float v2 = item_ct1.get_sub_group().shuffle(v0, otgx);	\
	int k2 = item_ct1.get_sub_group().shuffle(k0, otgx);	\
	int flag = ((v1 < v2) ^ (tgx > otgx)) && (v1 != v2);	\
	k0 = flag ? k1 : k2;	\
	v0 = flag ? v1 : v2;	\
}

#define WARPMINIMUM2(tgx, v0, k0) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 1) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 2) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 4) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 8) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 16)

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

void fill_Q (
	sycl::nd_item<3> item,
	sycl::half *Q_data
) {
	/*
	// Naive implementation: a single work-item fills data in
	if(item.get_local_id(2) == 0) {
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
	for (uint i = item.get_local_id(2); i < 4; i+=item.get_local_range().get(2)) {	// How many rows (of 4x4 blocks) are there in matrix A?
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
	for (uint k = item.get_local_id(2); k < 16; k+=item.get_local_range().get(2)) {
		for (uint ii = 0; ii < 4; ii++) {
			for (uint jj = 0; jj < 4; jj++) {
				Q_data[coeffs[k] + ii + 16*jj] = I4 [4*ii + jj];
			}
		}
	}
	*/

	/*
	// Enable this block to print matrix values
	if (item.get_group(2) == 0 && item.get_local_id(2) == 0) {
		printf("\nQ_data");
		for (uint i = 0; i < 16 * 16; i++) {
			if ((i % 16) == 0) {printf("\n[Row %u]: ", i/16);}
			printf(" %5.3f ", float(Q_data[i]));
		}
		printf("\n");
	}
	*/
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
	item.barrier(sycl::access::fence_space::local_space);

	// Identifying global, local, and group ids
	int globalId = item.get_global_linear_id();
	int localId = item.get_local_id(2);
	int groupId = item.get_group(2);
	int groupSize = item.get_local_range().get(2);

	// Identifying sub-groups
	sycl::sub_group sg = item.get_sub_group();
	int sgGroupRange = sg.get_group_range().get(2); // Returns the number of subgroups within the parent work-group
	int sgGroupId = sg.get_group_id().get(2); // Returns the index of the subgroup
	int sgSize = sg.get_local_range().get(2); // Returns the size of the subgroup
	int sgId = sg.get_local_id().get(2); // Returns the index of the work-item within its subgroup

	/*
	printf("localId = %i, globalId = %i, groupId = %i, groupSize = %i, sgGroupRange = %i, sgGroupId = %i, sgSize = %i, sgId = %i\n",
		localId, globalId, groupId, groupSize, sgGroupRange, sgGroupId, sgSize, sgId);
	*/

	// Only one sub-group (sgId == 0) performs reduction
	//if(sgId == 0) {
        if(localId <= 31) {
		/*
		printf("localId = %i, globalId = %i, groupId = %i, groupSize = %i, sgGroupRange = %i, sgGroupId = %i, sgSize = %i, sgId = %i\n",
			localId, globalId, groupId, groupSize, sgGroupRange, sgGroupId, sgSize, sgId);
		*/
                fill_Q(item, Q_data);

		// Declaring and filling submatrices
		joint_matrix<sycl::sub_group, sycl::half, use::b, rowscols_K, rowscols_N, layout::col_major> sub_P;
		joint_matrix<sycl::sub_group, sycl::half, use::accumulator, rowscols_M, rowscols_N> sub_V;

		joint_matrix<sycl::sub_group, sycl::half, use::a, rowscols_M, rowscols_K, layout::col_major> sub_Q;
		joint_matrix<sycl::sub_group, sycl::half, use::b, rowscols_K, rowscols_N, layout::col_major> sub_W;
		joint_matrix<sycl::sub_group, sycl::half, use::accumulator, rowscols_M, rowscols_N> sub_C;

		joint_matrix_fill(sg, sub_P, HALF_ONE); // P: only ones
		joint_matrix_fill(sg, sub_V, HALF_ZERO); // Output: initialize to zeros
		joint_matrix_fill(sg, sub_C, HALF_ZERO); // Final result

		/*
		// Must be copied from an accumulator matrix operand type
		joint_matrix_store(sg, sub_V, sycl::local_ptr<sycl::half>(data_to_be_reduced), 16, layout::col_major);
		if (groupId == 0 && localId == 0) {
			printf("\nsub_V");
			for (uint i = 0; i < 16 * 16; i++) {
				if ((i % 16) == 0) {printf("\n[Row %u]: ", i/16);}
				printf(" %5.3f ", float(data_to_be_reduced[i]));
			}
			printf("\n");
		}
		*/

		/*
		// Must be copied from an accumulator matrix operand type
		joint_matrix_store(sg, sub_C, sycl::local_ptr<sycl::half>(data_to_be_reduced), 16, layout::col_major);
		if (groupId == 0 && localId == 0) {
			printf("\nsub_C");
			for (uint i = 0; i < 16 * 16; i++) {
				if ((i % 16) == 0) {printf("\n[Row %u]: ", i/16);}
				printf(" %5.3f ", float(data_to_be_reduced[i]));
			}
			printf("\n");
		}
		*/

		// TODO: check the entire data to processed will fit into the matrix registers
		joint_matrix_load(sg, sub_Q, sycl::local_ptr<sycl::half>(Q_data), 16);

		// 1. Accumulate the values: V <- AP + V
		for(uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK) / (16 * 16);  i++) {
			const uint offset = i * 16 * 16;

			/*
			if (groupId == 0 && localId == 0) {
				printf("\ni = %d, tripcount= %d, offset = %d ", i, (4 * NUM_OF_THREADS_PER_BLOCK) / (16 * 16), offset);
			}
			*/

			joint_matrix<sycl::sub_group, sycl::half, use::a, rowscols_M, rowscols_K, layout::col_major> sub_A;

			/*
			if (groupId == 0 && localId == 0) {
				printf("\ndata_to_be_reduced (inside)");
				for (uint i = 0; i < 16 * 16; i++) {
					if ((i % 16) == 0) {printf("\n[Row %2u]: ", i/16);}
					printf(" %5.3f ", float(data_to_be_reduced[i]));
				}
				printf("\n");
			}
			*/

			joint_matrix_load(sg, sub_A, sycl::local_ptr<sycl::half>(data_to_be_reduced + offset), 16);
			//sub_V = joint_matrix_mad(sg, sub_A, sub_P, sub_V);	// 2024.1
			joint_matrix_mad(sg, sub_V, sub_A, sub_P, sub_V);	// 2024.2.1
		}

		// W <- V (required since we need V as a "use::b")
		joint_matrix_store(sg, sub_V, sycl::local_ptr<sycl::half>(tmp), 16, layout::col_major);
		joint_matrix_load(sg, sub_W, sycl::local_ptr<sycl::half>(tmp), 16);

		// 2. Perform line sum: C <- QW + C (zero)
		//sub_C = joint_matrix_mad(sg, sub_Q, sub_W, sub_C);	// 2024.1
		joint_matrix_mad(sg, sub_C, sub_Q, sub_W, sub_C);	// 2024.2.1

		// 3. Store result in shared memory
		joint_matrix_store(sg, sub_C, sycl::local_ptr<sycl::half>(data_to_be_reduced), 16, layout::col_major);
	}

	item.barrier(sycl::access::fence_space::local_space);
}

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
