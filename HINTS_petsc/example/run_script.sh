#!/bin/bash

mkdir out_files


for num_samples_total  in 25000 50000 250000 500000;
do 	
	for num_basis_functions  in 64 128 256;
	do 			
		for dofs_don  in 8 16 32 64;
		do 			
			for k_sigmas  in 1 3 6;
			do 						
				sbatch --output=out_files/out_${num_samples_total}_${num_basis_functions}_${dofs_don}_${k_sigmas}.out --job-name=Helm_${num_samples_total}_${num_basis_functions}_${dofs_don}_${k_sigmas} submission_script_daint.job $num_samples_total $num_basis_functions $dofs_don $k_sigmas; 
			done
		done
	done
done


