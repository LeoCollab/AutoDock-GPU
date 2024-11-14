#ifndef XMX_CAPABILITIES
#define XMX_CAPABILITIES

using namespace sycl::ext::oneapi::experimental;

void query_xmx_capabilities (sycl::queue Q){
	// Device architecture
	architecture arch = Q.get_device().get_info<info::device::architecture>();
	printf("\nDetected architecture: ");
	switch (arch) {
		case architecture::x86_64:
			printf("Any CPU device with the x86_64 instruction set");
		break;
		case architecture::intel_gpu_pvc:
			printf("Ponte Vecchio Intel graphics");
		break;
		case architecture::intel_gpu_pvc_vg:
			printf("Ponte Vecchio VG Intel graphics");
		break;
		case architecture::nvidia_gpu_sm_70:
			printf("NVIDIA Volta (compute capability 7.0)");
		break;
		case architecture::nvidia_gpu_sm_72:
			printf("NVIDIA Volta (compute capability 7.2)");
		break;
		case architecture::nvidia_gpu_sm_75:
			printf("NVIDIA Turing (compute capability 7.5)");
		break;
		case architecture::nvidia_gpu_sm_80:
			printf("NVIDIA Ampere (compute capability 8.0)");
		break;
		case architecture::nvidia_gpu_sm_86:
			printf("NVIDIA Ampere (compute capability 8.6)");
		break;
		case architecture::nvidia_gpu_sm_89:
			printf("NVIDIA Ada Lovelace (compute capability 8.9)");
		break;
		case architecture::nvidia_gpu_sm_90:
			printf("NVIDIA Hopper (compute capability 9.0)");
		break;
//		case architecture::nvidia_gpu_sm_90a:
//			printf("NVIDIA Hopper (compute capability 9.0a)");
//		break;
		default:
			printf("Unknown");
		break;
	}

	// Runtime Query of matrix sizes and types configurations
	std::vector<matrix::combination> combinations = Q.get_device().get_info<info::device::matrix_combinations>();
	printf("\nSupported joint-matrix combinations on device: %2i\n", sizeof(combinations));

	for (int i = 0; i < sizeof(combinations); i++) {
		uint invalid = 0;

		printf("\tComb. #%2i: \t", i);

		if (combinations[i].max_msize == 0) {
			printf("m =  %2zu", combinations[i].msize);

			// *********************************************************
			if (combinations[i].max_nsize == 0) {
				printf("\t\tn =  %2zu", combinations[i].nsize);

				// ---------------------------------------------------------
				if (combinations[i].max_ksize == 0) {
					printf("\t\tk =  %2zu", combinations[i].ksize);
				}
				else {
					if (combinations[i].max_ksize <= 64) {
						printf("\t\tk <= %2zu", combinations[i].max_ksize);
					}
					else {
						//printf("[cont. size > 64 (k)!]");
						invalid++;
						goto invalid_combination;
					}
				}
				// ---------------------------------------------------------
			}
			else {
				if (combinations[i].max_nsize <= 64) {
					printf("\t\tn <= %2zu", combinations[i].max_nsize);
				}
				else {
					//printf("[cont. size > 64 (n)!]");
					invalid++;
					goto invalid_combination;
				}

				// ---------------------------------------------------------
				if (combinations[i].max_ksize == 0) {
					printf("\t\tk =  %2zu", combinations[i].ksize);
				}
				else {
					if (combinations[i].max_ksize <= 64) {
						printf("\t\tk <= %2zu", combinations[i].max_ksize);
					}
					else {
						//printf("[cont. size > 64 (k)!]");
						invalid++;
						goto invalid_combination;
					}
				}
				// ---------------------------------------------------------
			}
			// *********************************************************
		}
		else {
			if (combinations[i].max_msize <= 64) {
				printf("m <= %2zu", combinations[i].max_msize);
			}
			else {
				//printf("[cont. size > 64 (m)!]");
				invalid++;
				goto invalid_combination;
			}

			// *********************************************************
			if (combinations[i].max_nsize == 0) {
				printf("\t\tn =  %2zu", combinations[i].nsize);

				// ---------------------------------------------------------
				if (combinations[i].max_ksize == 0) {
					printf("\t\tk =  %2zu", combinations[i].ksize);
				}
				else {
					if (combinations[i].max_ksize <= 64) {
						printf("\t\tk <= %2zu", combinations[i].max_ksize);
					}
					else {
						//printf("[cont. size > 64 (k)!]");
						invalid++;
						goto invalid_combination;
					}
				}
				// ---------------------------------------------------------
			}
			else {
				if (combinations[i].max_nsize <= 64) {
					printf("\t\tn <= %2zu", combinations[i].max_nsize);
				}
				else {
					//printf("[cont. size > 64 (n)!]");
					invalid++;
					goto invalid_combination;
				}

				// ---------------------------------------------------------
				if (combinations[i].max_ksize == 0) {
					printf("\t\tk =  %2zu", combinations[i].ksize);
				}
				else {
					if (combinations[i].max_ksize <= 64) {
						printf("\t\tk <= %2zu", combinations[i].max_ksize);
					}
					else {
						//printf("[cont. size > 64 (k)!]");
						invalid++;
						goto invalid_combination;
					}
				}
				// ---------------------------------------------------------
			}
			// *********************************************************
		}
		invalid_combination:
		if (invalid > 0) {
			printf("\t\t-> Combination not usable!");
		}

		/*
		// Only for debugging
		printf("\tComb. #%2i: \t continuous-sizes (m,n,k): %2zu %2zu %2zu \t|\t discrete sizes (m,n,k): %2zu %2zu %2zu",
			i,
			combinations[i].max_msize, combinations[i].max_nsize, combinations[i].max_ksize,
			combinations[i].msize, combinations[i].nsize, combinations[i].ksize);
		*/

		// Checking support for specific types
		if (combinations[i].atype == matrix::matrix_type::fp16 &&
			combinations[i].btype == matrix::matrix_type::fp16 &&
			combinations[i].ctype == matrix::matrix_type::fp32 &&
			combinations[i].dtype == matrix::matrix_type::fp32) {
			printf(" <- A (half), B (half), C (float), D (float)");
		}
		if (combinations[i].atype == matrix::matrix_type::fp16 &&
			combinations[i].btype == matrix::matrix_type::fp16 &&
			combinations[i].ctype == matrix::matrix_type::fp16 &&
			combinations[i].dtype == matrix::matrix_type::fp32) {
			printf(" <- A (half), B (half), C (half), D (float)");
		}
		if (combinations[i].atype == matrix::matrix_type::fp16 &&
			combinations[i].btype == matrix::matrix_type::fp16 &&
			combinations[i].ctype == matrix::matrix_type::fp32 &&
			combinations[i].dtype == matrix::matrix_type::fp16) {
			printf(" <- A (half), B (half), C (float), D (half)");
		}
		if (combinations[i].atype == matrix::matrix_type::fp16 &&
			combinations[i].btype == matrix::matrix_type::fp16 &&
			combinations[i].ctype == matrix::matrix_type::fp16 &&
			combinations[i].dtype == matrix::matrix_type::fp16) {
			printf(" <- A (half), B (half), C (half), D (half)");
		}
		if (combinations[i].atype == matrix::matrix_type::tf32 &&
			combinations[i].btype == matrix::matrix_type::tf32 &&
			combinations[i].ctype == matrix::matrix_type::fp32 &&
			combinations[i].dtype == matrix::matrix_type::fp32) {
			printf(" <- A (tf32), B (tf32), C (float), D (float)");
		}

		// Checking types supported for a specific combination
		if (
			(combinations[i].msize == 32) &&
			(combinations[i].nsize == 64) &&
			(combinations[i].ksize == 16)
		) {
			switch (combinations[i].atype)	{
				case matrix::matrix_type::bf16:
					printf(" <- A (bf16)"); // Supported on PVC 11.11.2024
					if (combinations[i].btype == matrix::matrix_type::bf16) {
						printf(", B (bf16)"); // Supported on PVC 11.11.2024
						if (combinations[i].ctype == matrix::matrix_type::fp32) {
							printf(", C (float)"); // Supported on PVC 11.11.2024
							if (combinations[i].dtype == matrix::matrix_type::fp32) {
								printf(", D (float)"); // Supported on PVC 11.11.2024
							} else {
								printf("\t D (other types than fp32)");
							}
						}
						else {
							printf("\t C (other types than fp32)");
						}
					}
					else {
						printf("\t B (other types than bf16)");
					}
					break;
				case matrix::matrix_type::fp16:
					printf("\n\t A (fp16)");
					break;
				case matrix::matrix_type::tf32:
					printf("\n\t A (tf32)");
					break;
				case matrix::matrix_type::fp32:
					printf("\n\t A (fp32)");
					break;
				case matrix::matrix_type::fp64:
					printf("\n\t A (fp64)");
					break;
				case matrix::matrix_type::sint8:
					printf("\n\t A (sint8)");
					break;
				case matrix::matrix_type::sint16:
					printf("\n\t A (sint16)");
					break;
				case matrix::matrix_type::sint32:
					printf("\n\t A (sint32)");
					break;
				case matrix::matrix_type::sint64:
					printf("\n\t A (sint64)");
					break;
				case matrix::matrix_type::uint8:
					printf("\n\t A (uint8)");
					break;
				case matrix::matrix_type::uint16:
					printf("\n\t A (uint16)");
					break;
				case matrix::matrix_type::uint32:
					printf("\n\t A (uint32)");
					break;
				case matrix::matrix_type::uint64:
					printf("\n\t A (uint64)");
					break;
			}

		}


		printf("\n");
	}
}

#endif // XMX_CAPABILITIES


