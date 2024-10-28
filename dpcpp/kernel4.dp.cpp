#include <sycl/sycl.hpp>
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

//#define DEBUG_ENERGY_KERNEL4

//#define DOCK_TRACE

#ifdef DOCK_TRACE
#ifdef __SYCL_DEVICE_ONLY__
          #define CONSTANT __attribute__((opencl_constant))
#else
          #define CONSTANT
#endif
static const CONSTANT char FMT1[] = "DOCK_TRACE: %s globalID: %6d %20s %10.6f %20s %10.6f";
#endif

void
gpu_gen_and_eval_newpops_kernel(
	float* pMem_conformations_current,
	float* pMem_energies_current,
	float* pMem_conformations_next,
	float* pMem_energies_next
	,
	sycl::nd_item<3> item_ct1,
	GpuData cData,
	float *offspring_genotype,
	int *parent_candidates,
	float *candidate_energies,
	int *parents,
	int *covr_point,
	float *randnums,
	float *sBestEnergy,
	int *sBestID,
	sycl::float3 *calc_coords
	)
// The GPU global function
{
	int run_id;
	int temp_covr_point;
	float energy;
	int bestID;
#ifdef DOCK_TRACE
	size_t global_id = item_ct1.get_global_id(2);
#endif 
	// In this case this compute-unit is responsible for elitist selection
	if ((item_ct1.get_group(2) % cData.dockpars.pop_size) == 0)
	{
		// Find and copy best member of population to position 0
		if (item_ct1.get_local_id(2) < cData.dockpars.pop_size)
		{
			sBestEnergy[item_ct1.get_local_id(2)] = pMem_energies_current[item_ct1.get_group(2) + item_ct1.get_local_id(2)];
			sBestID[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2);
		}

		for (int entity_counter = item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
				 entity_counter < cData.dockpars.pop_size;
				 entity_counter += item_ct1.get_local_range().get(2))
		{
			if (pMem_energies_current[item_ct1.get_group(2) + entity_counter] < sBestEnergy[item_ct1.get_local_id(2)])
			{
				sBestEnergy[item_ct1.get_local_id(2)] = pMem_energies_current[item_ct1.get_group(2) + entity_counter];
				sBestID[item_ct1.get_local_id(2)] = entity_counter;
			}
		}

		item_ct1.barrier(SYCL_MEMORY_SPACE);

		// This could be implemented with a tree-like structure
		// which may be slightly faster
		if (item_ct1.get_local_id(2) == 0)
		{
			energy = sBestEnergy[0];
			bestID = sBestID[0];

			for (int entity_counter = 1;
					 entity_counter < item_ct1.get_local_range().get(2);
					 entity_counter++)
			{
				if ( (sBestEnergy[entity_counter] < energy) && (entity_counter < cData.dockpars.pop_size) )
				{
					energy = sBestEnergy[entity_counter];
					bestID = sBestID[entity_counter];
				}
			}

			// Setting energy value of new entity
			pMem_energies_next[item_ct1.get_group(2)] = energy;

			// Zero (0) evals were performed for entity selected with elitism (since it was copied only)
			cData.pMem_evals_of_new_entities[item_ct1.get_group(2)] = 0;
		}

		item_ct1.barrier(SYCL_MEMORY_SPACE);

		// Copy best genome to next generation
		int dOffset = GENOTYPE_LENGTH_IN_GLOBMEM * item_ct1.get_group(2);
		int sOffset = dOffset + GENOTYPE_LENGTH_IN_GLOBMEM * bestID;
		for (int gene_counter = item_ct1.get_local_id(2);
				 gene_counter < cData.dockpars.num_of_genes;
				 gene_counter += item_ct1.get_local_range().get(2))
		{
			pMem_conformations_next[dOffset + gene_counter] = pMem_conformations_current[sOffset + gene_counter];
		}
	}
	else
	{
		// Generating the following random numbers: 
		// [0..3] for parent candidates,
		// [4..5] for binary tournaments, [6] for deciding crossover,
		// [7..8] for crossover points, [9] for local search
		for (uint32_t gene_counter = item_ct1.get_local_id(2);
					  gene_counter < 10;
					  gene_counter += item_ct1.get_local_range().get(2))
		{
			randnums[gene_counter] = gpu_randf(cData.pMem_prng_states, item_ct1);
		}

#if 0
		if ((threadIdx.x == 0) && (blockIdx.x == 1))
		{
			printf("%06d ", blockIdx.x);
			for (int i = 0; i < 10; i++)
				printf("%12.6f ", randnums[i]);
			printf("\n");
		}
#endif

		// Determining run ID
		run_id = item_ct1.get_group(2) / cData.dockpars.pop_size;

		item_ct1.barrier(SYCL_MEMORY_SPACE);

		if (item_ct1.get_local_id(2) < 4) // it is not ensured that the four candidates will be different...
		{
			parent_candidates[item_ct1.get_local_id(2)] = (int)(cData.dockpars.pop_size * randnums[item_ct1.get_local_id(2)]); // using randnums[0..3]
            candidate_energies[item_ct1.get_local_id(2)] = pMem_energies_current[run_id * cData.dockpars.pop_size + parent_candidates[item_ct1.get_local_id(2)]];
		}

		item_ct1.barrier(SYCL_MEMORY_SPACE);

		if (item_ct1.get_local_id(2) < 2)
		{
			// Notice: dockpars_tournament_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
			if (candidate_energies[2 * item_ct1.get_local_id(2)] <
                candidate_energies[2 * item_ct1.get_local_id(2) + 1])
			{
				if (/*100.0f**/ randnums[4 + item_ct1.get_local_id(2)] < cData.dockpars.tournament_rate) { // using randnum[4..5]
					parents[item_ct1.get_local_id(2)] = parent_candidates[2 * item_ct1.get_local_id(2)];
				}
				else
				{
					parents[item_ct1.get_local_id(2)] = parent_candidates[2 * item_ct1.get_local_id(2) + 1];
				}
			}
			else
			{
				if (/*100.0f**/ randnums[4 + item_ct1.get_local_id(2)] < cData.dockpars.tournament_rate) {
					parents[item_ct1.get_local_id(2)] = parent_candidates[2 * item_ct1.get_local_id(2) + 1];
				}
				else
				{
					parents[item_ct1.get_local_id(2)] = parent_candidates[2 * item_ct1.get_local_id(2)];
				}
			}
		}

		item_ct1.barrier(SYCL_MEMORY_SPACE);

		// Performing crossover
		// Notice: dockpars_crossover_rate was scaled down to [0,1] in host
		// to reduce number of operations in device
		if (/*100.0f**/randnums[6] < cData.dockpars.crossover_rate) // Using randnums[6]
		{
			if (item_ct1.get_local_id(2) < 2) { // Using randnum[7..8]
				covr_point[item_ct1.get_local_id(2)] = (int)((cData.dockpars.num_of_genes - 1) * randnums[7 + item_ct1.get_local_id(2)]);
			}

			item_ct1.barrier(SYCL_MEMORY_SPACE);

			// covr_point[0] should store the lower crossover-point
			if (item_ct1.get_local_id(2) == 0)
			{
				if (covr_point[1] < covr_point[0])
				{
					temp_covr_point = covr_point[1];
					covr_point[1]   = covr_point[0];
					covr_point[0]   = temp_covr_point;
				}
			}

			item_ct1.barrier(SYCL_MEMORY_SPACE);

			for (uint32_t gene_counter = item_ct1.get_local_id(2);
						  gene_counter < cData.dockpars.num_of_genes;
						  gene_counter += item_ct1.get_local_range().get(2))
			{
				// Two-point crossover
				if (covr_point[0] != covr_point[1]) 
				{
					if ((gene_counter <= covr_point[0]) || (gene_counter > covr_point[1]))
						offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
					else
						offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
				}
				// Single-point crossover
				else
				{
					if (gene_counter <= covr_point[0])
						offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
					else
						offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
				}
			}
		}
		else //no crossover
		{
			for (uint32_t gene_counter = item_ct1.get_local_id(2);
						  gene_counter < cData.dockpars.num_of_genes;
						  gene_counter += item_ct1.get_local_range().get(2))
			{
				offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter];
			}
		} // End of crossover

		item_ct1.barrier(SYCL_MEMORY_SPACE);

		// Performing mutation
		for (uint32_t gene_counter = item_ct1.get_local_id(2);
					  gene_counter < cData.dockpars.num_of_genes;
					  gene_counter += item_ct1.get_local_range().get(2))
		{
			// Notice: dockpars_mutation_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
			if (/*100.0f**/ gpu_randf(cData.pMem_prng_states, item_ct1) < cData.dockpars.mutation_rate)
			{
				// Translation genes
				if (gene_counter < 3)
				{
					offspring_genotype[gene_counter] += cData.dockpars.abs_max_dmov * (2.0f * gpu_randf(cData.pMem_prng_states, item_ct1) - 1.0f);
				}
				// Orientation and torsion genes
				else
				{
					offspring_genotype[gene_counter] += cData.dockpars.abs_max_dang * (2.0f * gpu_randf(cData.pMem_prng_states, item_ct1) - 1.0f);
					map_angle(offspring_genotype[gene_counter]);
				}
			}
		} // End of mutation

		// Calculating energy of new offspring
		item_ct1.barrier(SYCL_MEMORY_SPACE);

		// =================================================================
		gpu_calc_energy(
			offspring_genotype,
			energy,
			run_id,
			calc_coords,
			item_ct1,
			cData
		);
		// =================================================================

		if (item_ct1.get_local_id(2) == 0)
		{
			pMem_energies_next[item_ct1.get_group(2)] = energy;
			cData.pMem_evals_of_new_entities[item_ct1.get_group(2)] = 1;
			#if defined (DEBUG_ENERGY_KERNEL4)
			printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL4-", "GRIDS", "INTRA", interE, intraE);
			#endif
		}

		// Copying new offspring to next generation
		for (uint32_t gene_counter = item_ct1.get_local_id(2);
					  gene_counter < cData.dockpars.num_of_genes;
					  gene_counter += item_ct1.get_local_range().get(2))
		{
			pMem_conformations_next[item_ct1.get_group(2) * GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter] = offspring_genotype[gene_counter];
		}
	}

#if 0
	if ((threadIdx.x == 0) && (blockIdx.x == 0))
	{
		printf("%06d %16.8f ", blockIdx.x, pMem_energies_next[blockIdx.x]);
		for (int i = 0; i < cData.dockpars.num_of_genes; i++)
			printf("%12.6f ", pMem_conformations_next[GENOTYPE_LENGTH_IN_GLOBMEM*blockIdx.x + i]);
	}
#endif
}

void gpu_gen_and_eval_newpops(
                              uint32_t blocks,
                              uint32_t threadsPerBlock,
                              float*   pMem_conformations_current,
                              float*   pMem_energies_current,
                              float*   pMem_conformations_next,
                              float*   pMem_energies_next
                             )
{
	dpct::get_default_queue().submit([&](sycl::handler &cgh) {
		extern dpct::constant_memory<GpuData, 0> cData;
		cData.init();
		auto cData_ptr_ct1 = cData.get_ptr();

		sycl::local_accessor<float, 1> offspring_genotype_acc_ct1(sycl::range<1>(64 /*ACTUAL_GENOTYPE_LENGTH*/), cgh);
		sycl::local_accessor<int, 1> parent_candidates_acc_ct1(sycl::range<1>(4), cgh);
		sycl::local_accessor<float, 1> candidate_energies_acc_ct1(sycl::range<1>(4), cgh);
		sycl::local_accessor<int, 1> parents_acc_ct1(sycl::range<1>(2), cgh);
		sycl::local_accessor<int, 1> covr_point_acc_ct1(sycl::range<1>(2), cgh);
		sycl::local_accessor<float, 1> randnums_acc_ct1(sycl::range<1>(10), cgh);
		sycl::local_accessor<float, 1> sBestEnergy_acc_ct1(sycl::range<1>(threadsPerBlock), cgh);
		sycl::local_accessor<int, 1> sBestID_acc_ct1(sycl::range<1>(threadsPerBlock), cgh);
		sycl::local_accessor<sycl::float3, 1> calc_coords_acc_ct1(sycl::range<1>(256 /*MAX_NUM_OF_ATOMS*/), cgh);

		cgh.parallel_for(
			sycl::nd_range<3>(
				sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock),
				sycl::range<3>(1, 1, threadsPerBlock)
			),
			[=](sycl::nd_item<3> item_ct1) {
				gpu_gen_and_eval_newpops_kernel(
					pMem_conformations_current,
					pMem_energies_current,
					pMem_conformations_next,
					pMem_energies_next,
					item_ct1,
					*cData_ptr_ct1,
					offspring_genotype_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					parent_candidates_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					candidate_energies_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					parents_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					covr_point_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					randnums_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					sBestEnergy_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					sBestID_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					calc_coords_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get()
				);
		});
	});

	LAUNCHERROR("gpu_gen_and_eval_newpops_kernel");
}
