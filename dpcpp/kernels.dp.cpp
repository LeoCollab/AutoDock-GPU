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
