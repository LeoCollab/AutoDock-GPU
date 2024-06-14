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


// Output interaction pairs
// #define INTERACTION_PAIR_INFO

#ifdef __INTEL_LLVM_COMPILER
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#endif
#include "calcenergy.h"
#ifdef __INTEL_LLVM_COMPILER
#include <cmath>
#endif

int prepare_const_fields_for_gpu(
	Liganddata*                  myligand_reference,
	Dockpars*                    mypars,
	kernelconstant_interintra*   KerConst_interintra,
	kernelconstant_intracontrib* KerConst_intracontrib,
	kernelconstant_intra*        KerConst_intra,
	kernelconstant_rotlist*      KerConst_rotlist,
	kernelconstant_conform*      KerConst_conform,
	kernelconstant_grads*        KerConst_grads
)
// The function fills the constant memory field of the GPU
// based on the parameters describing ligand, flexres, and
// docking parameters as well as reference orientation angles.
{
	// Some variables
	int i, j;
	float* floatpoi;
	int *intpoi;

// --------------------------------------------------
// Allocating memory on the heap (not stack) with new
// --------------------------------------------------
// atom_charges:            Stores the ligand atom charges.
//                          Element i corresponds to atom with atom ID i in myligand_reference.
	float* atom_charges            = new float[MAX_NUM_OF_ATOMS];
// atom_types:              Stores the ligand atom type IDs according to myligand_reference.
//                          Element i corresponds to atom with ID i in myligand_reference.
	int*   atom_types              = new int[MAX_NUM_OF_ATOMS];
// intraE_contributors:     Each three contiguous items describe an intramolecular contributor.
//                          The first two elements store the atom ID of the contributors according to myligand_reference.
//                          The third element is 0, if no H-bond can occur between the two atoms, and 1, if it can.
	int*   intraE_contributors     = new int[2*MAX_INTRAE_CONTRIBUTORS];
	unsigned short* VWpars_exp     = new unsigned short[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
	float* reqm_AB                 = new float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
// VWpars_AC_const:         Stores the A or C van der Waals parameters.
//                          The element i*MAX_NUM_OF_ATYPES+j and j*MAX_NUM_OF_ATYPES+i corresponds to A or C in case of
//                          H-bond for atoms with type ID i and j (according to myligand_reference).
	float* VWpars_AC               = new float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
// VWpars_BD:               Stores the B or D van der Waals parameters similar to VWpars_AC.
	float* VWpars_BD               = new float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
// dspars_S:                Stores the S desolvation parameters.
//                          The element i corresponds to the S parameter of atom with type ID i
//                          according to myligand_reference.
	float* dspars_S                = new float[MAX_NUM_OF_ATYPES];
	float* dspars_V                = new float[MAX_NUM_OF_ATYPES];
// rotlist:                 Stores the data describing the rotations for conformation calculation.
//                          Each element describes one rotation, and the elements are in a proper order,
//                          considering that NUM_OF_THREADS_PER_BLOCK rotations will be performed in
//                          parallel (that is, each block of contiguous NUM_OF_THREADS_PER_BLOCK pieces of
//                          elements describe rotations that can be performed simultaneously).
//                          One element is a 32 bit integer, with bit 0 in the LSB position.
//                          Bit  7-0 describe the atom ID of the atom to be rotated (according to myligand_reference).
//                          Bit 15-7 describe the rotatable bond ID of the bond around which the atom is to be rotated (if this is not a general rotation)
//                                   (bond ID is according to myligand_reference).
//                          If bit 16 is 1, this is the first rotation of the atom.
//                          If bit 17 is 1, this is a general rotation (so rotbond ID has to be ignored).
//                          If bit 18 is 1, this is a "dummy" rotation, that is, no rotation can be performed in this cycle
//                                         (considering the other rotations which are being carried out in this period).
	int*   rotlist                 = new int[MAX_NUM_OF_ROTATIONS];
// subrotlist_*				Stores a sub (rotation) list built from above rotlist.
//							Each of these sublists contains rotations that can be executed in parallel.
//							Reasoning: conformation calculation (originally processing a rotlist) consists of a large loop that cannot be parallelized
//							because every processed atom (identified as "atom_id") may be rotated several times.
// 							Any rotation modifies the spatial coordinates of "atom_id".
// 							A consecutive rotation on "atom_id" requires the latest coordinates of "atom_id",
//							and thus, it needs to wait until the former rotation on that same atom_id has finished.
// 							However, rotations on different "atom_id"s can be performed safely in parallel.
// 							To achieve that, we sort rotlist's rotations into sublists, each containing different "atom_id"s.
//							The loops (rotations) within each of these sublists are independent, and thus, processed in parallel.
//							While the sublists, among themselves, are processed in sequence.
	int* subrotlist_1				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_2				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_3				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_4				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_5				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_6				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_7				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_8				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_9				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_10				= new int[MAX_NUM_OF_ROTATIONS];
	int* subrotlist_11				= new int[MAX_NUM_OF_ROTATIONS];
// ref_coords_x:            Stores the x coordinates of the reference ligand atoms.
//                          Element i corresponds to the x coordinate of the atom with atom ID i (according to myligand_reference).
	float* ref_coords_x            = new float[MAX_NUM_OF_ATOMS];
// ref_coords_y:            Stores the y coordinates of the reference ligand atoms similarly to ref_coords_x.
	float* ref_coords_y            = new float[MAX_NUM_OF_ATOMS];
// ref_coords_z:            Stores the z coordinates of the reference ligand atoms similarly to ref_coords_x.
	float* ref_coords_z            = new float[MAX_NUM_OF_ATOMS];
// rotbonds_moving_vectors: Stores the coordinates of rotatable bond moving vectors. Element i, i+1 and i+2 (where i%3=0)
//                          correspond to the moving vector coordinates x, y and z of rotbond ID i, respectively
//                          (according to myligand_reference).
	float* rotbonds_moving_vectors = new float[3*MAX_NUM_OF_ROTBONDS];
// rotbonds_unit_vectors:   Stores the coordinates of rotatable bond unit vectors similarly to rotbonds_moving_vectors.
	float* rotbonds_unit_vectors   = new float[3*MAX_NUM_OF_ROTBONDS];

// rot_bonds:               Added for calculating torsion-related gradients.
//                          Passing list of rotbond-atom ids to the GPU.
//                          Contains the same information as processligand.h/Liganddata->rotbonds
//                          Each row corresponds to one rotatable bond of the ligand.
//                          The rotatable bond is described with the indices of the two atoms which are connected
//                          to each other by the bond.
//                          The row index is equal to the index of the rotatable bond.
	int*   rotbonds                = new int[2*MAX_NUM_OF_ROTBONDS];

// rotbonds_atoms:          Contains the same information as processligand.h/Liganddata->atom_rotbonds:
//                                  Matrix that contains the rotatable bonds - atoms assignment.
//                                  If the element [atom index][rotatable bond index] is equal to 1,
//                                  then the atom must be rotated if the bond rotates. A 0 means the opposite.
	int*   rotbonds_atoms          = new int[MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS];

// num_rotating_atoms_per_rotbond:
//                          Each entry corresponds to a rotbond_id
//                          The value of an entry indicates the number of atoms that rotate along with that rotbond_id
	int*   num_rotating_atoms_per_rotbond = new int[MAX_NUM_OF_ROTBONDS];
// --------------------------------------------------

	// Number of rotations in every subrotlist
	unsigned int subrotlist_1_length;
	unsigned int subrotlist_2_length;
	unsigned int subrotlist_3_length;
	unsigned int subrotlist_4_length;
	unsigned int subrotlist_5_length;
	unsigned int subrotlist_6_length;
	unsigned int subrotlist_7_length;
	unsigned int subrotlist_8_length;
	unsigned int subrotlist_9_length;
	unsigned int subrotlist_10_length;
	unsigned int subrotlist_11_length;

	// charges and type id-s
	floatpoi = atom_charges;
	intpoi = atom_types;

	for (i=0; i < myligand_reference->num_of_atoms; i++)
	{
		*floatpoi = (float) myligand_reference->atom_idxyzq[i][4];
		*intpoi  = (int) myligand_reference->atom_idxyzq[i][0];
		floatpoi++;
		intpoi++;
	}
	
	// intramolecular energy contributors
	myligand_reference->num_of_intraE_contributors = 0;
	for (i=0; i<myligand_reference->num_of_atoms-1; i++)
		for (j=i+1; j<myligand_reference->num_of_atoms; j++)
		{
			if (myligand_reference->intraE_contributors[i][j]){
#ifdef INTERACTION_PAIR_INFO
				printf("Pair interaction between: %i <-> %i\n",i+1,j+1);
#endif
				myligand_reference->num_of_intraE_contributors++;
			}
		}

	if (myligand_reference->num_of_intraE_contributors > MAX_INTRAE_CONTRIBUTORS)
	{
		printf("Error: Number of intramolecular energy contributor is larger than maximum (%d).\n",MAX_INTRAE_CONTRIBUTORS);
		fflush(stdout);
		return 1;
	}

	intpoi = intraE_contributors;
	for (i=0; i<myligand_reference->num_of_atoms-1; i++)
		for (j=i+1; j<myligand_reference->num_of_atoms; j++)
		{
			if (myligand_reference->intraE_contributors[i][j] == 1)
			{
				*intpoi = (int) i;
				intpoi++;
				*intpoi = (int) j;
				intpoi++;
			}
		}

	// van der Waals parameters
	for (i=0; i<myligand_reference->num_of_atypes; i++)
		for (j=0; j<myligand_reference->num_of_atypes; j++)
		{
			*(VWpars_exp + i*myligand_reference->num_of_atypes + j) = myligand_reference->VWpars_exp[i][j];
			floatpoi = reqm_AB + i*myligand_reference->num_of_atypes + j;
			*floatpoi = (float) myligand_reference->reqm_AB[i][j];

			if (is_H_bond(myligand_reference->base_atom_types[i], myligand_reference->base_atom_types[j]) &&
			    (!is_mod_pair(myligand_reference->atom_types[i], myligand_reference->atom_types[j], mypars->nr_mod_atype_pairs, mypars->mod_atype_pairs)))
			{
				floatpoi = VWpars_AC + i*myligand_reference->num_of_atypes + j;
				*floatpoi = (float) myligand_reference->VWpars_C[i][j];
				floatpoi = VWpars_AC + j*myligand_reference->num_of_atypes + i;
				*floatpoi = (float) myligand_reference->VWpars_C[j][i];

				floatpoi = VWpars_BD + i*myligand_reference->num_of_atypes + j;
				*floatpoi = (float) myligand_reference->VWpars_D[i][j];
				floatpoi = VWpars_BD + j*myligand_reference->num_of_atypes + i;
				*floatpoi = (float) myligand_reference->VWpars_D[j][i];
			}
			else
			{
				floatpoi = VWpars_AC + i*myligand_reference->num_of_atypes + j;
				*floatpoi = (float) myligand_reference->VWpars_A[i][j];
				floatpoi = VWpars_AC + j*myligand_reference->num_of_atypes + i;
				*floatpoi = (float) myligand_reference->VWpars_A[j][i];

				floatpoi = VWpars_BD + i*myligand_reference->num_of_atypes + j;
				*floatpoi = (float) myligand_reference->VWpars_B[i][j];
				floatpoi = VWpars_BD + j*myligand_reference->num_of_atypes + i;
				*floatpoi = (float) myligand_reference->VWpars_B[j][i];
			}
		}

	// desolvation parameters
	for (i=0; i<myligand_reference->num_of_atypes; i++)
	{
		dspars_S[i] = myligand_reference->solpar[i];
		dspars_V[i] = myligand_reference->volume[i];
	}

	// generate rotation list
	if (gen_rotlist(
			myligand_reference,
			rotlist,
			subrotlist_1,
			subrotlist_2,
			subrotlist_3,
			subrotlist_4,
			subrotlist_5,
			subrotlist_6,
			subrotlist_7,
			subrotlist_8,
			subrotlist_9,
			subrotlist_10,
			subrotlist_11,
			&subrotlist_1_length,
			&subrotlist_2_length,
			&subrotlist_3_length,
			&subrotlist_4_length,
			&subrotlist_5_length,
			&subrotlist_6_length,
			&subrotlist_7_length,
			&subrotlist_8_length,
			&subrotlist_9_length,
			&subrotlist_10_length,
			&subrotlist_11_length
			)
		!= 0)
	{
		printf("Error: Number of required rotations is larger than maximum (%d).\n",MAX_NUM_OF_ROTATIONS);
		return 1;
	}

	// coordinates of reference ligand
	for (i=0; i < myligand_reference->num_of_atoms; i++)
	{
		ref_coords_x[i] = myligand_reference->atom_idxyzq[i][1];
		ref_coords_y[i] = myligand_reference->atom_idxyzq[i][2];
		ref_coords_z[i] = myligand_reference->atom_idxyzq[i][3];
	}

	// rotatable bond vectors
	for (i=0; i < myligand_reference->num_of_rotbonds; i++){
		for (j=0; j<3; j++)
		{
			rotbonds_moving_vectors[3*i+j] = myligand_reference->rotbonds_moving_vectors[i][j];
			rotbonds_unit_vectors[3*i+j] = myligand_reference->rotbonds_unit_vectors[i][j];
		}
	}

	// Added for calculating torsion-related gradients.
	// Passing list of rotbond-atoms ids to the GPU.
	// Contains the same information as processligand.h/Liganddata->rotbonds
	for (i=0; i < myligand_reference->num_of_rotbonds; i++)
	{
		rotbonds [2*i]   = myligand_reference->rotbonds[i][0]; // id of first-atom
		rotbonds [2*i+1] = myligand_reference->rotbonds[i][1]; // id of second atom
	}

	// Contains the same information as processligand.h/Liganddata->atom_rotbonds
	// "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
	// If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
	// it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
	for (i=0; i<MAX_NUM_OF_ROTBONDS; i++)
	{
		num_rotating_atoms_per_rotbond [i] = 0;
	}

	for (i=0; i < myligand_reference->num_of_rotbonds; i++)
	{
		// Pointing to the mem area corresponding to a given rotbond
		intpoi = rotbonds_atoms + MAX_NUM_OF_ATOMS*i;

		for (j=0; j < myligand_reference->num_of_atoms; j++)
		{
			/*
			rotbonds_atoms [MAX_NUM_OF_ATOMS*i+j] = myligand_reference->atom_rotbonds [j][i];
			*/
			
			// If an atom rotates with a rotbond, then
			// add its atom-id to the entry corresponding to the rotbond-id.
			// Also, count the number of atoms that rotate with a certain rotbond
			if (myligand_reference->atom_rotbonds [j][i] == 1){
				*intpoi = j;
				intpoi++;
				num_rotating_atoms_per_rotbond [i] ++;
			}

		}
	}

	int m;

	for (m=0;m<MAX_NUM_OF_ATOMS;m++){
		if (m<myligand_reference->num_of_atoms)
			KerConst_interintra->ignore_inter_const[m] = (char)myligand_reference->ignore_inter[m];
		else
			KerConst_interintra->ignore_inter_const[m] = 1;
	}
	for (m=0;m<MAX_NUM_OF_ATOMS;m++){ KerConst_interintra->atom_charges_const[m]    = atom_charges[m];    }
	for (m=0;m<MAX_NUM_OF_ATOMS;m++){ KerConst_interintra->atom_types_const[m]      = atom_types[m];      }
	for (m=0;m<MAX_NUM_OF_ATOMS;m++){ KerConst_interintra->atom_types_map_const[m]  = myligand_reference->atom_map_to_fgrids[m]; }

	for (m=0;m<2*MAX_INTRAE_CONTRIBUTORS;m++){ KerConst_intracontrib->intraE_contributors_const[m] = intraE_contributors[m]; }

	for (m=0;m<MAX_NUM_OF_ATYPES;m++)			{ KerConst_intra->atom_types_reqm_const[m]  = myligand_reference->atom_types_reqm[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES;m++)	{ KerConst_intra->VWpars_exp_const[m]       = VWpars_exp[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES;m++)	{ KerConst_intra->reqm_AB_const[m]          = reqm_AB[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES;m++)	{ KerConst_intra->VWpars_AC_const[m]        = VWpars_AC[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES;m++)	{ KerConst_intra->VWpars_BD_const[m]        = VWpars_BD[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES;m++)		   	{ KerConst_intra->dspars_S_const[m]         = dspars_S[m]; }
	for (m=0;m<MAX_NUM_OF_ATYPES;m++)		   	{ KerConst_intra->dspars_V_const[m]         = dspars_V[m]; }

	for (m=0;m<MAX_NUM_OF_ROTATIONS;m++) {
		KerConst_rotlist->rotlist_const[m] = rotlist[m];
		KerConst_rotlist->subrotlist_1_const[m] = subrotlist_1[m];
		KerConst_rotlist->subrotlist_2_const[m] = subrotlist_2[m];
		KerConst_rotlist->subrotlist_3_const[m] = subrotlist_3[m];
		KerConst_rotlist->subrotlist_4_const[m] = subrotlist_4[m];
		KerConst_rotlist->subrotlist_5_const[m] = subrotlist_5[m];
		KerConst_rotlist->subrotlist_6_const[m] = subrotlist_6[m];
		KerConst_rotlist->subrotlist_7_const[m] = subrotlist_7[m];
		KerConst_rotlist->subrotlist_8_const[m] = subrotlist_8[m];
		KerConst_rotlist->subrotlist_9_const[m] = subrotlist_9[m];
		KerConst_rotlist->subrotlist_10_const[m] = subrotlist_10[m];
		KerConst_rotlist->subrotlist_11_const[m] = subrotlist_11[m];
/*
		if(m!=0 && m%myligand_reference->num_of_atoms==0)
			printf("***\n");
		if(m!=0 && m%NUM_OF_THREADS_PER_BLOCK==0)
			printf("===\n");
		printf("%i (%i): %i -> atom_id: %i, dummy: %i, first: %i, genrot: %i, rotbond_id: %i\n",m,m%NUM_OF_THREADS_PER_BLOCK,rotlist[m],rotlist[m] & RLIST_ATOMID_MASK, rotlist[m] & RLIST_DUMMY_MASK,rotlist[m] & RLIST_FIRSTROT_MASK,rotlist[m] & RLIST_GENROT_MASK,(rotlist[m] & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT);
*/
	}
	KerConst_rotlist->subrotlist_1_length = subrotlist_1_length;
	KerConst_rotlist->subrotlist_2_length = subrotlist_2_length;
	KerConst_rotlist->subrotlist_3_length = subrotlist_3_length;
	KerConst_rotlist->subrotlist_4_length = subrotlist_4_length;
	KerConst_rotlist->subrotlist_5_length = subrotlist_5_length;
	KerConst_rotlist->subrotlist_6_length = subrotlist_6_length;
	KerConst_rotlist->subrotlist_7_length = subrotlist_7_length;
	KerConst_rotlist->subrotlist_8_length = subrotlist_8_length;
	KerConst_rotlist->subrotlist_9_length = subrotlist_9_length;
	KerConst_rotlist->subrotlist_10_length = subrotlist_10_length;
	KerConst_rotlist->subrotlist_11_length = subrotlist_11_length;

	printf("\n");
	for (m = 0; m < MAX_NUM_OF_ROTATIONS; m++)
	{
		bool b1 = (m >= subrotlist_1_length);
		bool b2 = (m >= subrotlist_2_length);
		bool b3 = (m >= subrotlist_3_length);
		bool b4 = (m >= subrotlist_4_length);
		bool b5 = (m >= subrotlist_5_length);
		bool b6 = (m >= subrotlist_6_length);
		bool b7 = (m >= subrotlist_7_length);
		bool b8 = (m >= subrotlist_8_length);
		bool b9 = (m >= subrotlist_9_length);
		bool b10 = (m >= subrotlist_10_length);
		bool b11 = (m >= subrotlist_11_length);
		if (b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8 && b9 && b10 && b11)
		{
			break;
		}

		if ((subrotlist_1_length > 0) && (m < subrotlist_1_length))
		{
			printf("\t%i \t%i", m, KerConst_rotlist->subrotlist_1_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_2_length > 0) && (m < subrotlist_2_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_2_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_3_length > 0) && (m < subrotlist_3_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_3_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_4_length > 0) && (m < subrotlist_4_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_4_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_5_length > 0) && (m < subrotlist_5_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_5_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_6_length > 0) && (m < subrotlist_6_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_6_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_7_length > 0) && (m < subrotlist_7_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_7_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_8_length > 0) && (m < subrotlist_8_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_8_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_9_length > 0) && (m < subrotlist_9_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_9_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_10_length > 0) && (m < subrotlist_10_length))
		{
			printf("\t%i", KerConst_rotlist->subrotlist_10_const[m]);
		}
		else {printf("\t-");}

		if ((subrotlist_11_length > 0) && (m < subrotlist_11_length))
		{
			printf("\t%i\n", KerConst_rotlist->subrotlist_11_const[m]);
		}
		else {printf("\t-\n");}
	}

	printf("\tKerConst_rotlist->subrotlist_1_length: %u\n", KerConst_rotlist->subrotlist_1_length);
	printf("\tKerConst_rotlist->subrotlist_2_length: %u\n", KerConst_rotlist->subrotlist_2_length);
	printf("\tKerConst_rotlist->subrotlist_3_length: %u\n", KerConst_rotlist->subrotlist_3_length);
	printf("\tKerConst_rotlist->subrotlist_4_length: %u\n", KerConst_rotlist->subrotlist_4_length);
	printf("\tKerConst_rotlist->subrotlist_5_length: %u\n", KerConst_rotlist->subrotlist_5_length);
	printf("\tKerConst_rotlist->subrotlist_6_length: %u\n", KerConst_rotlist->subrotlist_6_length);
	printf("\tKerConst_rotlist->subrotlist_7_length: %u\n", KerConst_rotlist->subrotlist_7_length);
	printf("\tKerConst_rotlist->subrotlist_8_length: %u\n", KerConst_rotlist->subrotlist_8_length);
	printf("\tKerConst_rotlist->subrotlist_9_length: %u\n", KerConst_rotlist->subrotlist_9_length);
	printf("\tKerConst_rotlist->subrotlist_10_length: %u\n", KerConst_rotlist->subrotlist_10_length);
	printf("\tKerConst_rotlist->subrotlist_11_length: %u\n", KerConst_rotlist->subrotlist_11_length);
	printf("\n");

	for (m=0;m<MAX_NUM_OF_ATOMS;m++) {
		KerConst_conform->ref_coords_const[3*m]		 = ref_coords_x[m];
		KerConst_conform->ref_coords_const[3*m+1]	 = ref_coords_y[m];
		KerConst_conform->ref_coords_const[3*m+2]	 = ref_coords_z[m];
	}
	for (m=0;m<3*MAX_NUM_OF_ROTBONDS;m++){ KerConst_conform->rotbonds_moving_vectors_const[m]= rotbonds_moving_vectors[m]; }
	for (m=0;m<3*MAX_NUM_OF_ROTBONDS;m++){ KerConst_conform->rotbonds_unit_vectors_const[m]  = rotbonds_unit_vectors[m]; }

	// Added for calculating torsion-related gradients.
	// Passing list of rotbond-atoms ids to the GPU.
	// Contains the same information as processligand.h/Liganddata->rotbonds
	for (m=0;m<2*MAX_NUM_OF_ROTBONDS;m++) 			{ KerConst_grads->rotbonds[m] 			    = rotbonds[m]; }
	for (m=0;m<MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS;m++) 	{ KerConst_grads->rotbonds_atoms[m]                 = rotbonds_atoms[m]; }
	for (m=0;m<MAX_NUM_OF_ROTBONDS;m++) 			{ KerConst_grads->num_rotating_atoms_per_rotbond[m] = num_rotating_atoms_per_rotbond[m]; }

	delete[] atom_charges;
	delete[] atom_types;
	delete[] intraE_contributors;
	delete[] VWpars_exp;
	delete[] reqm_AB;
	delete[] VWpars_AC;
	delete[] VWpars_BD;
	delete[] dspars_S;
	delete[] dspars_V;
	delete[] rotlist;
	delete[] subrotlist_1;
	delete[] subrotlist_2;
	delete[] subrotlist_3;
	delete[] subrotlist_4;
	delete[] subrotlist_5;
	delete[] subrotlist_6;
	delete[] subrotlist_7;
	delete[] subrotlist_8;
	delete[] subrotlist_9;
	delete[] subrotlist_10;
	delete[] subrotlist_11;
	delete[] ref_coords_x;
	delete[] ref_coords_y;
	delete[] ref_coords_z;
	delete[] rotbonds_moving_vectors;
	delete[] rotbonds_unit_vectors;
	delete[] rotbonds;
	delete[] rotbonds_atoms;
	delete[] num_rotating_atoms_per_rotbond;

	return 0;
}



void make_reqrot_ordering(
	int number_of_req_rotations[MAX_NUM_OF_ATOMS],
	int atom_id_of_numrots[MAX_NUM_OF_ATOMS],
	int num_of_atoms
)
// The function puts the first array into a descending order and
// performs the same operations on the second array (since element i of
// number_or_req_rotations and element i of atom_id_of_numrots correspond to each other).
// Element i of the former array stores how many rotations have to be perfomed on the atom
// whose atom ID is stored by element i of the latter one. The third parameter has to be equal
// to the number of ligand atoms
{
	int i, j;
	int temp;

	for (j=0; j<num_of_atoms-1; j++)
		for (i=num_of_atoms-2; i>=j; i--)
			if (number_of_req_rotations[i+1] > number_of_req_rotations[i])
			{
				temp = number_of_req_rotations[i];
				number_of_req_rotations[i] = number_of_req_rotations[i+1];
				number_of_req_rotations[i+1] = temp;

				temp = atom_id_of_numrots[i];
				atom_id_of_numrots[i] = atom_id_of_numrots[i+1];
				atom_id_of_numrots[i+1] = temp;
			}

/*
	printf("\n\nRotation priority list after re-ordering:\n");
	for (i=0; i<num_of_atoms; i++)
		printf("Roatom_rotbondstation of %d (required rots remaining: %d)\n", atom_id_of_numrots[i], number_of_req_rotations[i]);
	printf("\n\n");
*/
}

int gen_rotlist(
	Liganddata* myligand,
	int* rotlist,
	int* subrotlist_1,
	int* subrotlist_2,
	int* subrotlist_3,
	int* subrotlist_4,
	int* subrotlist_5,
	int* subrotlist_6,
	int* subrotlist_7,
	int* subrotlist_8,
	int* subrotlist_9,
	int* subrotlist_10,
	int* subrotlist_11,
	unsigned int* subrotlist_1_length,
	unsigned int* subrotlist_2_length,
	unsigned int* subrotlist_3_length,
	unsigned int* subrotlist_4_length,
	unsigned int* subrotlist_5_length,
	unsigned int* subrotlist_6_length,
	unsigned int* subrotlist_7_length,
	unsigned int* subrotlist_8_length,
	unsigned int* subrotlist_9_length,
	unsigned int* subrotlist_10_length,
	unsigned int* subrotlist_11_length
)
// The function generates the rotation list which will be stored in the constant memory field rotlist_const by
// prepare_const_fields_for_gpu(). The structure of this array is described at that function.
{
	int atom_id, rotb_id, parallel_rot_id, rotlist_id;

	int number_of_req_rotations[MAX_NUM_OF_ATOMS];
	int number_of_req_rotations_copy[MAX_NUM_OF_ATOMS];

	int atom_id_of_numrots[MAX_NUM_OF_ATOMS];
	bool atom_wasnt_rotated_yet[MAX_NUM_OF_ATOMS];

	int new_rotlist_element;
	bool rotbond_found;
	int rotbond_candidate;
	int remaining_rots_around_rotbonds;

	myligand->num_of_rotcyc = 0;
	myligand->num_of_rotations_required = 0;

	// Handling special case when num_of_atoms < NUM_OF_THREADS_PER_BLOCK
	for (atom_id=0; atom_id<NUM_OF_THREADS_PER_BLOCK; atom_id++)
	{
		number_of_req_rotations[atom_id] = 0;
	}

	for (atom_id=0; atom_id<myligand->num_of_atoms; atom_id++)
	{
		atom_id_of_numrots[atom_id] = atom_id;
		atom_wasnt_rotated_yet[atom_id] = true;

		number_of_req_rotations[atom_id] = 1;

		for (rotb_id=0; rotb_id<myligand->num_of_rotbonds; rotb_id++)
			if (myligand->atom_rotbonds[atom_id][rotb_id] != 0)
				(number_of_req_rotations[atom_id])++;

		myligand->num_of_rotations_required += number_of_req_rotations[atom_id];
	}

	for (atom_id=0; atom_id<myligand->num_of_atoms; atom_id++)
	{
		number_of_req_rotations_copy[atom_id] = number_of_req_rotations[atom_id];
	}

	rotlist_id = 0;
	make_reqrot_ordering(number_of_req_rotations, atom_id_of_numrots, myligand->num_of_atoms);

	// If the atom with the most remaining rotations has to be rotated 0 times, done
	while (number_of_req_rotations[0] != 0)
	{
		if (rotlist_id == MAX_NUM_OF_ROTATIONS)
		{
			return 1;
		}

		// Putting the NUM_OF_THREADS_PER_BLOCK pieces of most important rotations to the list
		for (parallel_rot_id=0; parallel_rot_id<NUM_OF_THREADS_PER_BLOCK; parallel_rot_id++)
		{
			// If the atom has not to be rotated anymore, dummy rotation
			if (number_of_req_rotations[parallel_rot_id] == 0)
			{
				new_rotlist_element = RLIST_DUMMY_MASK;
			}
			else
			{
				atom_id = atom_id_of_numrots[parallel_rot_id];
				new_rotlist_element = ((int) atom_id) & RLIST_ATOMID_MASK;

				if (number_of_req_rotations[parallel_rot_id] == 1)
				{
					new_rotlist_element |= RLIST_GENROT_MASK;
				}
				else
				{
					rotbond_found = false;
					rotbond_candidate = myligand->num_of_rotbonds - 1;
					remaining_rots_around_rotbonds = number_of_req_rotations[parallel_rot_id] - 1; // -1 because of genrot

					while (!rotbond_found)
					{
						if (myligand->atom_rotbonds[atom_id][rotbond_candidate] != 0) // if the atom has to be rotated around current candidate
						{
							if (remaining_rots_around_rotbonds == 1) // if current value of remaining rots is 1, the proper rotbond is found
								rotbond_found = true;
							else
								remaining_rots_around_rotbonds--; // if not, decresing remaining rots (that is, skipping rotations which have to be performed later
						}

						if (!rotbond_found)
							rotbond_candidate--;

						if (rotbond_candidate < 0)
							return 1;
					}

					new_rotlist_element |= (((int) rotbond_candidate) << RLIST_RBONDID_SHIFT) & RLIST_RBONDID_MASK;
				}

				if (atom_wasnt_rotated_yet[atom_id])
					new_rotlist_element |= RLIST_FIRSTROT_MASK;

				// Putting atom_id's next rotation to rotlist
				atom_wasnt_rotated_yet[atom_id] = false;
				(number_of_req_rotations[parallel_rot_id])--;
			}

			rotlist[rotlist_id] = new_rotlist_element;
			rotlist_id++;
		}

		make_reqrot_ordering(number_of_req_rotations, atom_id_of_numrots, myligand->num_of_atoms);
		(myligand->num_of_rotcyc)++;
	}

	// ---------------------------------------------------------------------------
	// Building rotation lists
	// ---------------------------------------------------------------------------
	printf("\n# rotlist elements: %u\n", rotlist_id);
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		unsigned int atom_id = rotlist[rot_cnt] & RLIST_ATOMID_MASK;
		printf("rot-id: %u \tatom-id: %u\n", rot_cnt, atom_id);
	}

	printf("\n# atoms: %u\n", myligand->num_of_atoms);
	for (unsigned int atom_cnt = 0; atom_cnt < myligand->num_of_atoms; atom_cnt++) {
		printf("atom-id: %u \tnumber-rot-req: %u\n", atom_cnt, number_of_req_rotations_copy[atom_cnt]);
	}

	// Builing first rotation list
	int num_times_atom_in_subrotlist[MAX_NUM_OF_ATOMS];
	for (unsigned int atom_cnt = 0; atom_cnt < MAX_NUM_OF_ATOMS; atom_cnt++) {
		num_times_atom_in_subrotlist[atom_cnt] = 0;
	}

	// ---------------------------------------------------------------------------
	// Arrays storing rot ids already used in "subrotlist_*"
	int rots_used_in_subrotlist_1[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_2[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_3[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_4[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_5[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_6[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_7[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_8[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_9[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_10[MAX_NUM_OF_ROTATIONS];
	int rots_used_in_subrotlist_11[MAX_NUM_OF_ROTATIONS];

	// Assigning an initial value of MAX_NUM_OF_ROTATIONS,
	// which of course will never be taken by a rot id
	for (unsigned int rot_cnt = 0; rot_cnt < MAX_NUM_OF_ROTATIONS; rot_cnt++) {
		rots_used_in_subrotlist_1[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_2[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_3[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_4[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_5[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_6[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_7[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_8[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_9[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_10[rot_cnt] = MAX_NUM_OF_ROTATIONS;
		rots_used_in_subrotlist_11[rot_cnt] = MAX_NUM_OF_ROTATIONS;
	}

	// ---------------------------------------------------------------------------
	// 1st rotations
	// ---------------------------------------------------------------------------
	//int subrotlist_1[MAX_NUM_OF_ROTATIONS];
	int rot_1_cnt = 0;

	printf("\nsubrotlist_1:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		if ((num_times_atom_in_subrotlist[atom_id] == 0)  && (number_of_req_rotations_copy[atom_id] >= 1)) {
			printf("[subrot_1 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_1_cnt, rot_cnt, atom_id);

			// Storing ids from the original "rotlist" that are used in "subrotlist_1"
			rots_used_in_subrotlist_1[rot_cnt] = rot_cnt;

			// First rotation of this atom is stored in "subrotlist_1"
			subrotlist_1[rot_1_cnt] = rotlist[rot_cnt];
			rot_1_cnt++;

			// An eventual second rotation of this atom will be stored in "subrotlist_2"
			num_times_atom_in_subrotlist[atom_id]++;
		}
	}
	*subrotlist_1_length = rot_1_cnt;
	printf("\tsubrotlist_1 length: %u\n", *subrotlist_1_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_1_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_1[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 2nd rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_2[MAX_NUM_OF_ROTATIONS];
	int rot_2_cnt = 0;

	printf("\nsubrotlist_2:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_2" was not already added to "subrotlist_1"
		if (rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) {

			if ((num_times_atom_in_subrotlist[atom_id] == 1) && (number_of_req_rotations_copy[atom_id] >= 2)) {
				printf("[subrot_2 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_2_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_2"
				rots_used_in_subrotlist_2[rot_cnt] = rot_cnt;

				// Second rotation of this atom is stored in "subrotlist_2"
				subrotlist_2[rot_2_cnt] = rotlist[rot_cnt];
				rot_2_cnt++;

				// An eventual third rotation of this atom will be stored in "subrotlist_3"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_2_length = rot_2_cnt;
	printf("\tsubrotlist_2 length: %u\n", *subrotlist_2_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_2_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_2[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 3rd rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_3[MAX_NUM_OF_ROTATIONS];
	int rot_3_cnt = 0;

	printf("\nsubrotlist_3:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_3"
		// was not already added to neither "subrotlist_1" nor "subrotlist_2"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt)) {

			if ((num_times_atom_in_subrotlist[atom_id] == 2) && (number_of_req_rotations_copy[atom_id] >= 3)) {
				printf("[subrot_3 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_3_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_3"
				rots_used_in_subrotlist_3[rot_cnt] = rot_cnt;

				// Third rotation of this atom is stored in "subrotlist_3"
				subrotlist_3[rot_3_cnt] = rotlist[rot_cnt];
				rot_3_cnt++;

				// An eventual fourth rotation of this atom will be stored in "subrotlist_4"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_3_length = rot_3_cnt;
	printf("\tsubrotlist_3 length: %u\n", *subrotlist_3_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_3_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_3[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 4th rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_4[MAX_NUM_OF_ROTATIONS];
	int rot_4_cnt = 0;

	printf("\nsubrotlist_4:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_4"
		// was not already added to neither
		// "subrotlist_1" nor "subrotlist_2" nor "subrotlist_3"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_3[rot_cnt] != rot_cnt)) {

			if ((num_times_atom_in_subrotlist[atom_id] == 3) && (number_of_req_rotations_copy[atom_id] >= 4)) {
				printf("[subrot_4 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_4_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_4"
				rots_used_in_subrotlist_4[rot_cnt] = rot_cnt;

				// Fourth rotation of this atom is stored in "subrotlist_4"
				subrotlist_4[rot_4_cnt] = rotlist[rot_cnt];
				rot_4_cnt++;

				// An eventual fifth rotation of this atom will be stored in "subrotlist_5"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_4_length = rot_4_cnt;
	printf("\tsubrotlist_4 length: %u\n", *subrotlist_4_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_4_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_4[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 5th rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_5[MAX_NUM_OF_ROTATIONS];
	int rot_5_cnt = 0;

	printf("\nsubrotlist_5:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_5"
		// was not already added to neither
		// "subrotlist_1" nor "subrotlist_2" nor "subrotlist_3" nor "subrotlist_4"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt) &&
		    (rots_used_in_subrotlist_3[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_4[rot_cnt] != rot_cnt)) {

			if ((num_times_atom_in_subrotlist[atom_id] == 4) && (number_of_req_rotations_copy[atom_id] >= 5)) {
				printf("[subrot_5 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_5_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_5"
				rots_used_in_subrotlist_5[rot_cnt] = rot_cnt;

				// Fifth rotation of this atom is stored in "subrotlist_5"
				subrotlist_5[rot_5_cnt] = rotlist[rot_cnt];
				rot_5_cnt++;

				// An eventual 6th rotation of this atom will be stored in "rotlist_six"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_5_length = rot_5_cnt;
	printf("\tsubrotlist_5 length: %u\n", *subrotlist_5_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_5_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_5[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 6th rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_6[MAX_NUM_OF_ROTATIONS];
	int rot_6_cnt = 0;

	printf("\nsubrotlist_6:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_6"
		// was not already added to neither
		// "subrotlist_1" nor "subrotlist_2" nor "subrotlist_3" nor "subrotlist_4" nor "subrotlist_5"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt) &&
		    (rots_used_in_subrotlist_3[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_4[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_5[rot_cnt] != rot_cnt)
			) {

			if ((num_times_atom_in_subrotlist[atom_id] == 5) && (number_of_req_rotations_copy[atom_id] >= 6)) {
				printf("[subrot_6 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_6_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_6"
				rots_used_in_subrotlist_6[rot_cnt] = rot_cnt;

				// Sixth rotation of this atom is stored in "subrotlist_6"
				subrotlist_6[rot_6_cnt] = rotlist[rot_cnt];
				rot_6_cnt++;

				// An eventual 7th rotation of this atom will be stored in "subrotlist_7"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_6_length = rot_6_cnt;
	printf("\tsubrotlist_6 length: %u\n", *subrotlist_6_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_6_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_6[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 7th rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_7[MAX_NUM_OF_ROTATIONS];
	int rot_7_cnt = 0;

	printf("\nsubrotlist_7:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_7"
		// was not already added to neither
		// "subrotlist_1" nor "subrotlist_2" nor "subrotlist_3" nor "subrotlist_4" nor "subrotlist_5" nor "subrotlist_6"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt) &&
		    (rots_used_in_subrotlist_3[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_4[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_5[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_6[rot_cnt] != rot_cnt)
			) {

			if ((num_times_atom_in_subrotlist[atom_id] == 6) && (number_of_req_rotations_copy[atom_id] >= 7)) {
				printf("[subrot_7 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_7_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_7"
				rots_used_in_subrotlist_7[rot_cnt] = rot_cnt;

				// Seventh rotation of this atom is stored in "subrotlist_7"
				subrotlist_7[rot_7_cnt] = rotlist[rot_cnt];
				rot_7_cnt++;

				// An eventual 8th rotation of this atom will be stored in "subrotlist_8"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_7_length = rot_7_cnt;
	printf("\tsubrotlist_7 length: %u\n", *subrotlist_7_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_7_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_7[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 8th rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_8[MAX_NUM_OF_ROTATIONS];
	int rot_8_cnt = 0;

	printf("\nsubrotlist_8:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_8"
		// was not already added to neither
		// "subrotlist_1" nor "subrotlist_2" nor "subrotlist_3" nor "subrotlist_4" nor "subrotlist_5" nor "subrotlist_6" nor "subrotlist_7"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt) &&
		    (rots_used_in_subrotlist_3[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_4[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_5[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_6[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_7[rot_cnt] != rot_cnt)
			) {

			if ((num_times_atom_in_subrotlist[atom_id] == 7) && (number_of_req_rotations_copy[atom_id] >= 8)) {
				printf("[subrot_8 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_8_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_8"
				rots_used_in_subrotlist_8[rot_cnt] = rot_cnt;

				// 8th rotation of this atom is stored in "subrotlist_8"
				subrotlist_8[rot_8_cnt] = rotlist[rot_cnt];
				rot_8_cnt++;

				// An eventual 9th rotation of this atom will be stored in "subrotlist_9"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_8_length = rot_8_cnt;
	printf("\tsubrotlist_8 length: %u\n", *subrotlist_8_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_8_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_8[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 9th rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_9[MAX_NUM_OF_ROTATIONS];
	int rot_9_cnt = 0;

	printf("\nsubrotlist_9:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_9"
		// was not already added to neither
		// "subrotlist_1" nor "subrotlist_2" nor "subrotlist_3" nor "subrotlist_4" nor
		// "subrotlist_5" nor "subrotlist_6" nor "subrotlist_7" nor "subrotlist_8"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt) &&
		    (rots_used_in_subrotlist_3[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_4[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_5[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_6[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_7[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_8[rot_cnt] != rot_cnt)
			) {

			if ((num_times_atom_in_subrotlist[atom_id] == 8) && (number_of_req_rotations_copy[atom_id] >= 9)) {
				printf("[subrot_9 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_9_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_9"
				rots_used_in_subrotlist_9[rot_cnt] = rot_cnt;

				// 9th rotation of this atom is stored in "subrotlist_9"
				subrotlist_9[rot_9_cnt] = rotlist[rot_cnt];
				rot_9_cnt++;

				// An eventual 10th rotation of this atom will be stored in "subrotlist_10"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_9_length = rot_9_cnt;
	printf("\tsubrotlist_9 length: %u\n", *subrotlist_9_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_9_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_9[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 10th rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_10[MAX_NUM_OF_ROTATIONS];
	int rot_10_cnt = 0;

	printf("\nsubrotlist_10:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_10"
		// was not already added to neither
		// "subrotlist_1" nor "subrotlist_2" nor "subrotlist_3" nor "subrotlist_4" nor
		// "subrotlist_5" nor "subrotlist_6" nor "subrotlist_7" nor "subrotlist_8" nor
		// "subrotlist_9"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt) &&
		    (rots_used_in_subrotlist_3[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_4[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_5[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_6[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_7[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_8[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_9[rot_cnt] != rot_cnt)
			) {

			if ((num_times_atom_in_subrotlist[atom_id] == 9) && (number_of_req_rotations_copy[atom_id] >= 10)) {
				printf("[subrot_10 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_10_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_10"
				rots_used_in_subrotlist_10[rot_cnt] = rot_cnt;

				// 10th rotation of this atom is stored in "subrotlist_10"
				subrotlist_10[rot_10_cnt] = rotlist[rot_cnt];
				rot_10_cnt++;

				// An eventual 11th rotation of this atom will be stored in "subrotlist_11"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_10_length = rot_10_cnt;
	printf("\tsubrotlist_10 length: %u\n", *subrotlist_10_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_10_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_10[i]);
	}
	*/

	// ---------------------------------------------------------------------------
	// 11th rotations (for only those atoms that experiment such)
	// ---------------------------------------------------------------------------
	//int subrotlist_11[MAX_NUM_OF_ROTATIONS];
	int rot_11_cnt = 0;

	printf("\nsubrotlist_11:\n");
	for (unsigned int rot_cnt = 0; rot_cnt < myligand->num_of_rotations_required; rot_cnt++) {
		int atom_id = (rotlist[rot_cnt] & RLIST_ATOMID_MASK);

		// Making sure rot id to be added to "subrotlist_11"
		// was not already added to neither
		// "subrotlist_1" nor "subrotlist_2" nor "subrotlist_3" nor "subrotlist_4" nor
		// "subrotlist_5" nor "subrotlist_6" nor "subrotlist_7" nor "subrotlist_8" nor
		// "subrotlist_9" nor "subrotlist_10"
		if ((rots_used_in_subrotlist_1[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_2[rot_cnt] != rot_cnt) &&
		    (rots_used_in_subrotlist_3[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_4[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_5[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_6[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_7[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_8[rot_cnt] != rot_cnt) &&
			(rots_used_in_subrotlist_9[rot_cnt] != rot_cnt) && (rots_used_in_subrotlist_10[rot_cnt] != rot_cnt)
			) {

			if ((num_times_atom_in_subrotlist[atom_id] == 10) && (number_of_req_rotations_copy[atom_id] >= 11)) {
				printf("[subrot_11 rot-id]: %u \t[orig rot-id]: %u \tatom-id: %u\n", rot_11_cnt, rot_cnt, atom_id);

				// Storing ids from the original "rotlist" that are used in "subrotlist_11"
				rots_used_in_subrotlist_11[rot_cnt] = rot_cnt;

				// 11th rotation of this atom is stored in "subrotlist_11"
				subrotlist_11[rot_11_cnt] = rotlist[rot_cnt];
				rot_11_cnt++;

				// An eventual 12th rotation of this atom will be stored in "subrotlist_12"
				num_times_atom_in_subrotlist[atom_id]++;
			}

		}
	}
	*subrotlist_11_length = rot_11_cnt;
	printf("\tsubrotlist_11 length: %u\n", *subrotlist_11_length);
	/*
	for (unsigned int i = 0; i < *subrotlist_11_length; i++)
	{
		printf("\t%i \t%i\n", i, subrotlist_11[i]);
	}
	*/

	return 0;
}
