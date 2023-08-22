#!/bin/bash

#set -o xtrace

pdb=7cpa
input_path=input/${pdb}/derived
input_protein=${input_path}/${pdb}_protein.maps.fld
input_ligand=${input_path}/${pdb}_ligand.pdbqt

init_tool() {
	# Initializing oneAPI tools
	printf "\n"
	source "/opt/intel/oneapi/setvars.sh"
}

compile_for_roofline() {
#	printf "\nmake DEVICE=XeGPU PLATFORM=PVC CONFIG=LDEBUG_VTUNE"
	printf "\nmake DEVICE=XeGPU PLATFORM=PVC CONFIG=RELEASE"
	printf "\n"
#	make DEVICE=XeGPU PLATFORM=PVC CONFIG=LDEBUG_VTUNE
	make DEVICE=XeGPU PLATFORM=PVC CONFIG=RELEASE
}

choose_codeversion() {
	printf "\nSolis-Wets will be run by default. Do you __also__ want to run ADADELTA?"
	printf "\n"
	read -p "Type either [y] or [n]: " RUN_ALSO_ADADELTA

	if [ "${RUN_ALSO_ADADELTA}" == "y" ]; then
		printf "\nWe will profile both Solis-Wets and ADADELTA \n"
	elif [ "${RUN_ALSO_ADADELTA}" == "n" ]; then
		printf "\nWe will profile only Solis-Wets \n"
	else
		printf "\nWrong selection. Type either [y] or [n] -> terminating!"
		printf "\n" && echo $0 && exit 1
	fi
	sleep 1
}

define_executable() {
	adgpu_binary=bin/autodock_xegpu_64wi

	if [ -f "${adgpu_binary}" ]; then
		printf "${adgpu_binary} exists!\n"
	else
		printf "${adgpu_binary} does NOT exist -> terminating!\n"
		printf "\n" && echo $0 && exit 1
	fi
	sleep 1

	adgpu_cmd_sw="${adgpu_binary} -ffile ${input_protein} -lfile ${input_ligand} -nrun 100 -lsmet sw --heuristics 0 --autostop 0"
	adgpu_cmd_ad="${adgpu_binary} -ffile ${input_protein} -lfile ${input_ligand} -nrun 100 -lsmet ad --heuristics 0 --autostop 0"

	printf "\nAutoDock-GPU commands: "
	printf "\n${adgpu_cmd_sw}"
	if [ "${RUN_ALSO_ADADELTA}" == "y" ]; then
		printf "\n${adgpu_cmd_ad}"
	fi
	sleep 1
}

print_cmd () {
	printf "\n$1\n"
}

run_cmd () {
	print_cmd "$1"
	$1
	if [ "${RUN_ALSO_ADADELTA}" == "y" ]; then
		print_cmd "$2"
		$2
	fi
}

run_gpu_roofline() {
	printf "\n"
	printf "\n------------------------------------------------\n"
	printf "run_gpu_roofline() ..."
	printf "\n------------------------------------------------\n"
	output_folder_sw="r_gpu-roofline_${pdb}_sw"
	output_folder_ad="r_gpu-roofline_${pdb}_ad"

	cmd_roofline_shorcut="advisor --collect=roofline --profile-gpu"
	cmd_perfmodeling="advisor --collect=projection --profile-gpu --model-baseline-gpu"

	cmd_roofline_sw_1="${cmd_roofline_shorcut} --project-dir=${output_folder_sw} -- ${adgpu_cmd_sw}"
	cmd_roofline_sw_2="${cmd_perfmodeling} --project-dir=${output_folder_sw}"
	
	cmd_roofline_ad_1="${cmd_roofline_shorcut} --project-dir=${output_folder_ad} -- ${adgpu_cmd_ad}"
	cmd_roofline_ad_2="${cmd_perfmodeling} --project-dir=${output_folder_ad}"

	run_cmd "${cmd_roofline_sw_1}"
	run_cmd "${cmd_roofline_sw_2}"

	if [ "${RUN_ALSO_ADADELTA}" == "y" ]; then
		run_cmd "${cmd_roofline_ad_1}"
		run_cmd "${cmd_roofline_ad_2}"
	fi
}


init_tool
compile_for_roofline
choose_codeversion
define_executable
run_gpu_roofline


