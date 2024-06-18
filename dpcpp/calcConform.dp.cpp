void calcConform(
	float *pGenotype,
	sycl::float4 genrot_movingvec,
	sycl::float4 genrot_unitvec,
	sycl::float3* calc_coords,
	sycl::nd_item<3> item_ct1,
	GpuData* cData,
	int* subrotlist,
	int subrotlist_length
){
	// ================================================
	// CALCULATING ATOMIC POSITIONS AFTER ROTATIONS
	// ================================================
	if ( (item_ct1.get_local_id(2) == 0) && (item_ct1.get_group(2) == 0) )
	{
	
	for (uint rotation_counter = 0;
			  rotation_counter < subrotlist_length;
			  rotation_counter ++)
	{
/*
	for (uint rotation_counter = item_ct1.get_local_id(2);
			  rotation_counter < subrotlist_length;
			  rotation_counter += item_ct1.get_local_range().get(2))
	{
*/
		//int rotation_list_element = (cData->pKerconst_rotlist)->rotlist_const[rotation_counter];
		int rotation_list_element = subrotlist[rotation_counter];
		printf("rot_counter = %i \trot_list_element = %i\n", rotation_counter, rotation_list_element);

		if ((rotation_list_element & RLIST_DUMMY_MASK) == 0) // If not dummy rotation
		{
			uint atom_id = rotation_list_element & RLIST_ATOMID_MASK;
			printf("\tatom_id = %i\n", atom_id);

			// Capturing atom coordinates
			sycl::float4 atom_to_rotate;
			atom_to_rotate.x() = calc_coords[atom_id].x();
			atom_to_rotate.y() = calc_coords[atom_id].y();
			atom_to_rotate.z() = calc_coords[atom_id].z();
			atom_to_rotate.w() = 0.0f;
			printf("atom_to_rotate: x=%f \ty=%f \tz=%f \tz=%f\n", atom_to_rotate.x(), atom_to_rotate.y(), atom_to_rotate.z(), atom_to_rotate.w());

			// initialize with general rotation values
			sycl::float4 rotation_unitvec;
			sycl::float4 rotation_movingvec;

			if (atom_id < (cData->dockpars).true_ligand_atoms)
			{
				rotation_unitvec = genrot_unitvec;
				rotation_movingvec = genrot_movingvec;
			} else
			{
				rotation_unitvec.x() = 0.0f;
				rotation_unitvec.y() = 0.0f;
				rotation_unitvec.z() = 0.0f;
				rotation_unitvec.w() = 1.0f;
				rotation_movingvec.x() = 0.0f;
				rotation_movingvec.y() = 0.0f;
				rotation_movingvec.z() = 0.0f;
				rotation_movingvec.w() = 0.0f;
			}

			if ((rotation_list_element & RLIST_GENROT_MASK) == 0) // If rotating around rotatable bond
			{
				uint rotbond_id = (rotation_list_element & RLIST_RBONDID_MASK) >> RLIST_RBONDID_SHIFT;

				float rotation_angle = pGenotype[6+rotbond_id] * DEG_TO_RAD * 0.5f;
				float s = SYCL_SIN(rotation_angle);
				rotation_unitvec.x() = s * (cData->pKerconst_conform)->rotbonds_unit_vectors_const[3 * rotbond_id];
				rotation_unitvec.y() = s * (cData->pKerconst_conform)->rotbonds_unit_vectors_const[3 * rotbond_id + 1];
				rotation_unitvec.z() = s * (cData->pKerconst_conform)->rotbonds_unit_vectors_const[3 * rotbond_id + 2];
				rotation_unitvec.w() = SYCL_COS(rotation_angle);

				rotation_movingvec.x() = (cData->pKerconst_conform)->rotbonds_moving_vectors_const[3 * rotbond_id];
				rotation_movingvec.y() = (cData->pKerconst_conform)->rotbonds_moving_vectors_const[3 * rotbond_id + 1];
				rotation_movingvec.z() = (cData->pKerconst_conform)->rotbonds_moving_vectors_const[3 * rotbond_id + 2];

				// Performing additionally the first movement which
				// is needed only if rotating around rotatable bond
				atom_to_rotate.x() -= rotation_movingvec.x();
				atom_to_rotate.y() -= rotation_movingvec.y();
				atom_to_rotate.z() -= rotation_movingvec.z();
			}

			// Performing final movement and storing values
			sycl::float4 qt = quaternion_rotate(atom_to_rotate, rotation_unitvec);
			calc_coords[atom_id].x() = qt.x() + rotation_movingvec.x();
			calc_coords[atom_id].y() = qt.y() + rotation_movingvec.y();
			calc_coords[atom_id].z() = qt.z() + rotation_movingvec.z();
		} // End if-statement not dummy rotation

		//item_ct1.barrier(SYCL_MEMORY_SPACE);

	} // End rotation_counter for-loop
}
}
