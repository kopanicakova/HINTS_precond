[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# HINTS: Preconditioning of Krylov methods (The linear-systems are assembled and solved using Firedrake and PETSc.)

## This repository contains the code used to generate large-scale results in the paper: <br> **"Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers" by Zhang, Kahana, Kopanicakova, Turkel, Ranade, Pathak, and Karniadakis.**<br> 

If you use the developed code/its components for your research, please use the following bibtex entries (or equivalent) to cite us
```bibtex
@article{zktrpkk_24,
title = {Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers},
author = {Enrui Zhang and Adar Kahana and Alena Kopani{\v{c}}{\'a}kov{\'a} and Eli Turkel and Rishikesh Ranade and Jay Pathak and George Em Karniadakis},
journal = {},
volume = {0},
number = {0},
pages = {},
year = {},
doi = {},
URL = {},
note = {Accepted for the publication in Nature Machine intelligence.},
}
```


### Depedencies
Firedrake=0.13.0+6118.g149f8fda6 <br>
petsc=3.20.5 <br>
torch=2.2.2 <br>
numpy=1.24.0 <br>
matplotlib=3.9.0 <br>
pandas=2.2.2 <br>


### Hardware requirements
GPU is not necessary but it is highly desirable for training the DeepONets efficiently.


### Installation guide (building Petsc might take more than 1 hour)
1. Make sure to deactivate any conda enviroment you might have!
2. Install Firedrake - official guidance can be found at https://www.firedrakeproject.org/download.html.
   
	We have followed these steps:  
	2.1. mkdir my_path/DonPrecond;  <br>
 	2.2. cd my_path/DonPrecond  <br>
	2.3. curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install <br>
	2.4. add support for exodus meshes, i.e., add  "petsc_options.add("--download-exodusii")" to line 745 of firedrake-install script <br>
	2.5. python3 firedrake-install --disable-ssh --no-package-manager <br>


### Instructions to run the code
3.1. source the firedrake enviroment, i.e., <br> 
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  . /my_path/DonPrecond/firedrake/bin/activate <br>
3.2. copy HINTS_petsc folder to /my_path/DonPrecond/ folder <br> 
3.3. cd HINTS_petsc <br>
3.4. Export path to HINTS_petsc code, i.e.,   <br> 
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; export PYTHONPATH=$PYTHONPATH:my_path/DonPrecond/HINTS_petsc <br>
3.5. cd example <br>


##### Running numerical experiment: 
We have uploaded to Zenodo an instance of the dataset used to generate the large-scale results reported in the paper. 
This dataset can be downloaded using the following link: https://zenodo.org/records/10904349/files/NonNestedHelm3D_5000_1_32_1_0.0001.pkl?download=1. 
Using the provided code, you can generate larger or smaller datasets. The HINTS preconditioner typically performs better if the DeepOnet is trained using a larger number of samples. <br>
To execute the different experiments, one can use the following commands: 
			

	3.5.1. HYPRE-AMG preconditioner: 
		- using uploaded samples for k=6, you can test the code as
			python3 -u hints_test_hypre_sampled_k.py  --num_samples_total 10000 --num_samples 100000 --k_sigma 6.0 



	3.5.1. HINTS-MG preconditioner, e.g., for the example where DON is trained with only 10,000 on cube 8x8x8 
		- python3 -u hints_test_HINTSgmg_sampled_k.py   --epochs 50000000 --force_retrain false --recreate_data false --only_train false --num_samples_total 10000 --num_samples 10000 --dofs_don 8  --num_basis_functions 128 --k_sigma 6


To obtain alternative configurations of HINTS-Jacobi-MG, please explore different values of parameters for --dofs_don, --num_samples, --num_basis_functions, and --k_sigmas. This will require you to create different datasets as well as to train deepOnets.

## Dislaimer
This code was developed for research purposes only. The authors make no warranties, express or implied, regarding its suitability for any particular purpose or its performance.

## License
The software is realized with NO WARRANTY and it is licenzed under [BSD 3-Clause license](https://opensource.org/licenses/BSD-3-Clause)

# Copyright
Copyright (c) 2024 Brown University and Università della Svizzera Italiana (Euler Institute)


## Contact
Alena Kopaničáková (<alena_kopanicakova@brown.edu>, <alena.kopanicakova@usi.ch>)


## Contributors: 
* Alena Kopanicakova (Brown, Providence; USI, Lugano; ANITI, Toulouse)

