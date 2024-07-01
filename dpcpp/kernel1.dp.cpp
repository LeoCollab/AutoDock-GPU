#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian
Genetic Algorithm Copyright (C) 2017 TU Darmstadt, Embedded Systems and
Applications Group, Germany. All rights reserved. For some of the code,
Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research
Institute.
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

void
gpu_calc_initpop_kernel(
	float* pMem_conformations_current,
	float* pMem_energies_current
	,
	sycl::nd_item<3> item_ct1,
	GpuData cData,
	sycl::float3 *calc_coords
/*
	,
	float *sFloatAccumulator
*/
	)
{
	float  energy = 0.0f;
	int run_id = item_ct1.get_group(2) / cData.dockpars.pop_size;
	float *pGenotype = pMem_conformations_current + item_ct1.get_group(2) * GENOTYPE_LENGTH_IN_GLOBMEM;

	// =============================================================
	gpu_calc_energy(
		pGenotype,
		energy,
		run_id,
		calc_coords,
/*
		sFloatAccumulator,
*/
		item_ct1,
		cData
	);
	// =============================================================

	// Write out final energy
	if (item_ct1.get_local_id(2) == 0)
	{
		pMem_energies_current[item_ct1.get_group(2)] = energy;
		cData.pMem_evals_of_new_entities[item_ct1.get_group(2)] = 1;
	}
}

void gpu_calc_initpop(
                      uint32_t blocks,
                      uint32_t threadsPerBlock,
                      float*   pConformations_current,
                      float*   pEnergies_current
                     )
{
	// Verifying that the passed work-group size doesn't exceed device's limit
	auto max_wg_size = dpct::get_default_queue().get_device().get_info<sycl::info::device::max_work_group_size>();
	//printf("\tk1: max_wg_size = %lu, passed_wg_size = %u\n", max_wg_size, threadsPerBlock);
	assert(max_wg_size >= threadsPerBlock);

	dpct::get_default_queue().submit([&](sycl::handler &cgh) {
		extern dpct::constant_memory<GpuData, 0> cData;
		cData.init();
		auto cData_ptr_ct1 = cData.get_ptr();

		sycl::local_accessor<sycl::float3, 1> calc_coords_acc_ct1(sycl::range<1>(/*256*/ MAX_NUM_OF_ATOMS), cgh);
/*
		sycl::local_accessor<float, 0> sFloatAccumulator_acc_ct1(cgh);
*/
		cgh.parallel_for(
			sycl::nd_range<3>(
				sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock),
				sycl::range<3>(1, 1, threadsPerBlock)
			),
			[=](sycl::nd_item<3> item_ct1) {
				gpu_calc_initpop_kernel(
					pConformations_current,
					pEnergies_current,
					item_ct1,
					*cData_ptr_ct1,
					calc_coords_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get()
/*	
					,
					sFloatAccumulator_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get()
*/
				);
			});
	});

	LAUNCHERROR("gpu_calc_initpop_kernel");
}
