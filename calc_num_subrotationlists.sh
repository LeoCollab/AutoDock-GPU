#!/bin/bash

# ./calc_num_subrotationlists.sh |& tee calc_num_subrotationlists.txt

# Source: https://gitlab.com/L30nardoSV/ad-gpu_miniset_20.git

# Subset of 5 inputs
EXPERIMENTS_DATASET=(1u4d 1oyt 1mzc 3s8o 2d1o)

# Set of 20 inputs
#EXPERIMENTS_DATASET=(1u4d 1xoz 1yv3 1owe 1oyt 1ywr 1t46 2bm2 1mzc 1r55 5wlo 1kzk 3s8o 5kao 1hfs 1jyq 2d1o 3drf 4er4 3er5)

# LSMET set
#LSMET_SET=(sw ad)
LSMET_SET=(sw)

# Execution parameters
LSIT=300
LSRAT=100.000000
SMOOTH=0.500
NEV=2500000
NGEN=99999
NRUN=100
PSIZE=150
#INPUTS_DIR=./ad-gpu_miniset_20/data
INPUTS_DIR=./data

function numwi() {
	for ipdb in ${EXPERIMENTS_DATASET[@]}; do
		printf '%s\n' " "
		for ilsmet in ${LSMET_SET[@]}; do
			printf '%s\n' " "
			( # Starting a subshell to print only the command
				set -x; \
				$1 \
				-lsmet ${ilsmet} \
				-lsit ${LSIT} \
				-lsrat ${LSRAT} \
				-smooth ${SMOOTH} \
				-nev ${NEV} \
				-ngen ${NGEN} \
				-nrun ${NRUN} \
				-psize ${PSIZE} \
				-lfile ${INPUTS_DIR}/${ipdb}/rand-0.pdbqt \
				-xraylfile ${INPUTS_DIR}/${ipdb}/flex-xray.pdbqt \
				-ffile ${INPUTS_DIR}/${ipdb}/protein.maps.fld \
				-xmloutput 0 \
				-autostop 1 \
				-heuristics 1 \
				-resnam ${ipdb}_${ilsmet}_"`date +"%Y-%m-%d-%H:%M"`"
			)
		done
	done
}

numwi "./bin/autodock_xegpu_64wi"
