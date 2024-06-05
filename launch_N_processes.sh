#! /bin/bash
prog="./bin/autodock_xegpu_64wi"
prog_options="-ffile ./input/${2}/derived/${2}_protein.maps.fld -lfile ./input/${2}/derived/${2}_ligand.pdbqt -nrun 100 -ngen 27000 -psize 150 -resnam test -gfpop 0 -lsmet sw"
cmd="${prog} ${prog_options}"
echo ${prog}
echo ${prog_options}
echo ${cmd}

function run_adgpu () {
	numactl --interleave=2,3 $cmd
}

# https://unix.stackexchange.com/questions/392951/how-to-write-a-for-loop-which-runs-an-asynchronous-command-in-each-iteration
#run_adgpu
#run_adgpu & run_adgpu & run_adgpu
#for i in {1..2}; do { run_adgpu & }; done

re='^[0-9]+$'
if ! [[ ${1} =~ ${re} ]] ; then
	printf "\n\n>>> Error: not an number!\n\n"; exit 1
else
	printf "\n\n>>>> Number of process to be run simultaneously: ${1}\n\n"
fi

# Running processes asynchronously and storing pids in array
# https://www.hostinger.com/tutorials/bash-for-loop-guide-and-examples
# https://stackoverflow.com/a/356154
for i in `seq 1 $1`
do
	run_adgpu &
	pids[${i}]=$!
done

# Waiting for all pids
for pid in ${pids[*]}
do
	wait $pid
done

# ./launch_N_processes.sh 10 1stp
