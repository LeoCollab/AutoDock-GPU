#!/bin/bash

#set -o xtrace

pdb=1mzc
input_path=ad-gpu_miniset_20/data/${pdb}
input_protein=${input_path}/protein.maps.fld
input_ligand=${input_path}/rand-0.pdbqt

warning_message() {
	printf "\nMake sure that:"
	printf "\n - AutoDock-GPU was compiled for profiling on PVC"
	printf "\n - make DEVICE=XeGPU PLATFORM=PVC CONFIG=LDEBUG_VTUNE"
	printf "\n"
	sleep 1
}

init_tool() {
	# Initializing oneAPI tools
	printf "\n"
	source "/opt/intel/oneapi/setvars.sh"
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
	adgpu_binary=bin/autodock_gpu_64wi

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

run_gpu_offload() {
	printf "\n"
	printf "\n------------------------------------------------\n"
	printf "run_gpu_offload() ..."
	printf "\n------------------------------------------------\n"
	cmd_offload="vtune -collect gpu-offload"
	output_folder=${output_mainfolder}/r_gpu-offload_${pdb}

	local cmd_local_sw="${cmd_offload} -r ${output_folder}_sw -- ${adgpu_cmd_sw}"
	local cmd_local_ad="${cmd_offload} -r ${output_folder}_ad -- ${adgpu_cmd_ad}"

	run_cmd "${cmd_local_sw}" "${cmd_local_ad}"
}

run_characterization_globallocalacceses() {
	printf "\n"
	printf "\n------------------------------------------------\n"
	printf "run_characterization_globallocalacceses() ... "
	printf "\n------------------------------------------------\n"
	cmd_characterization_globallocalacceses="vtune -collect gpu-hotspots -knob profiling-mode=characterization -knob characterization-mode=global-local-accesses"
	output_folder=${output_mainfolder}/r_gpu-hotspots_characterization_globallocalaccesses_${pdb}

	local cmd_local_sw="${cmd_characterization_globallocalacceses} -r ${output_folder}_sw -- ${adgpu_cmd_sw}"
	local cmd_local_ad="${cmd_characterization_globallocalacceses} -r ${output_folder}_ad -- ${adgpu_cmd_ad}"

	run_cmd "${cmd_local_sw}" "${cmd_local_ad}"
}

run_characterization_instructioncount() {
	printf "\n"
	printf "\n------------------------------------------------\n"
	printf "run_characterization_instructioncount() ..."
	printf "\n------------------------------------------------\n"
	cmd_characterization_instructioncount="vtune -collect gpu-hotspots -knob profiling-mode=characterization -knob characterization-mode=instruction-count"
	output_folder=${output_mainfolder}/r_gpu-hotspots_characterization_instructioncount_${pdb}

	local cmd_local_sw="${cmd_characterization_instructioncount} -r ${output_folder}_sw -- ${adgpu_cmd_sw}"
	local cmd_local_ad="${cmd_characterization_instructioncount} -r ${output_folder}_ad -- ${adgpu_cmd_ad}"

	run_cmd "${cmd_local_sw}" "${cmd_local_ad}"
}

run_sourceanalysis_bblatency() {
	printf "\n"
	printf "\n------------------------------------------------\n"
	printf "run_sourceanalysis_bblatency() ..."
	printf "\n------------------------------------------------\n"
	cmd_sourceanalysis_bblatency="vtune -collect gpu-hotspots -knob profiling-mode=source-analysis -knob source-analysis=bb-latency"
	output_folder=${output_mainfolder}/r_gpu-hotspots_sourceanalysis_bblatency_${pdb}

	local cmd_local_sw="${cmd_sourceanalysis_bblatency} -r ${output_folder}_sw -- ${adgpu_cmd_sw}"
	local cmd_local_ad="${cmd_sourceanalysis_bblatency} -r ${output_folder}_ad -- ${adgpu_cmd_ad}"

	run_cmd "${cmd_local_sw}" "${cmd_local_ad}"
}

run_sourceanalysis_memlatency() {
	printf "\n"
	printf "\n------------------------------------------------\n"
	printf "run_sourceanalysis_memlatency() ..."
	printf "\n------------------------------------------------\n"
	cmd_sourceanalysis_memlatency="vtune -collect gpu-hotspots -knob profiling-mode=source-analysis -knob source-analysis=mem-latency"
	output_folder=${output_mainfolder}/r_gpu-hotspots_sourceanalysis_memlatency_${pdb}

	local cmd_local_sw="${cmd_sourceanalysis_memlatency} -r ${output_folder}_sw -- ${adgpu_cmd_sw}"
	local cmd_local_ad="${cmd_sourceanalysis_memlatency} -r ${output_folder}_ad -- ${adgpu_cmd_ad}"

	run_cmd "${cmd_local_sw}" "${cmd_local_ad}"
}

run_characterization() {
	run_characterization_globallocalacceses
#	run_characterization_instructioncount
}

run_sourceanalysis() {
	run_sourceanalysis_bblatency
	run_sourceanalysis_memlatency
}

run_gpu_hotspots() {
	run_characterization
	run_sourceanalysis
}

warning_message
init_tool
choose_codeversion
define_executable
run_gpu_offload
run_gpu_hotspots


