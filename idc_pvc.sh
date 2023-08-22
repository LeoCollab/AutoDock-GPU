#!/bin/bash
#SBATCH -A u102810
#SBATCH -p pvc-shared
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

source /opt/intel/oneapi/setvars.sh

echo " "
echo "-------------->>>> Testing functionality"
make DEVICE=XeGPU PLATFORM=PVC test

echo " "
echo "-------------->>>> Compiling for profiling"
make DEVICE=XeGPU PLATFORM=PVC CONFIG=LDEBUG_VTUNE

echo " "
echo "-------------->>>> Starting actual profiling"
vtune \
-collect gpu-hotspots \
-knob profiling-mode=characterization \
-knob characterization-mode=global-local-accesses \
-r "profiling_AD-GPU" \
./bin/autodock_xegpu_64wi \
-ffile ./input/7cpa/derived/7cpa_protein.maps.fld \
-lfile ./input/7cpa/derived/7cpa_ligand.pdbqt \
-nrun 100 \
-ngen 27000 \
-psize 150 \
-resnam test \
-gfpop 0 \
-lsmet sw

# Execution command on IDC head node: sbatch idc_pvc.sh
