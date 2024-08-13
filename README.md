[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# HINTS: Firedrake and petsc4py

# Preconditioning of Krylov methods using HINTS. The problems are assembled and solved using Firedrake and petsc4py. 
This repository contains the code used to generate results in the paper: <br> 
**Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers by Zhang, Kahana, Kopanicakova, Turkel, Ranade, Pathak, and Karniadakis.**<br> 

If you use the developed code/its components for your research, please use the following bibtex entries (or equivalent) to cite us
```bibtex
@article{zktrpkk_24,
title = {Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers},
author = { Enrui Zhang and Adar Kahana and Alena Kopani{\v{c}}{\'a}kov{\'a} and Eli Turkel and Rishikesh Ranade and Jay Pathak and George Em Karniadakis},
journal = {},
volume = {0},
number = {0},
pages = {},
year = {},
doi = {},
URL = {},
note = {Accepted for publication in Nature Machine intelligence.},
}
```


### Depedencies
Firedrake=0.13.0+6118.g149f8fda6
petsc=3.20.5
torch=2.2.2
numpy=1.24.0
matplotlib=3.9.0
pandas=2.2.2


### Hardware requirements
GPU is not necessary but it is highly desirable for training the DeepONets


### Installation guide (building, including Petsc, might take more than 1 hour)
1. Make sure to deactivate any conda enviroment you might have!
2. Install Firedrake (official guidance can be found at https://www.firedrakeproject.org/download.html). 
	We followed the following steps: 
	2.0. 	mkdir my_path/DonPrecond 
			cd my_path/DonPrecond
	2.1. curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
	2.2. add support for exodus meshes, i.e., add  "petsc_options.add("--download-exodusii")" to line 745 of firedrake-install script
	2.3. python3 firedrake-install --disable-ssh --no-package-manager


### Instructions to run the code
	3.0.  source the firedrake enviroment, i.e., 
			. /my_path/DonPrecond/firedrake/bin/activate
	3.1.  copy HINTS_petsc folder to /my_path/DonPrecond/ folder
	3.2.  cd HINTS_petsc
	3.3.  Export path to HINTS_petsc code, i.e., 
			export PYTHONPATH=$PYTHONPATH:my_path/DonPrecond/HINTS_petsc
	3.4   cd example


##### 3.5. 	Testing different examples: We have uploaded a dataset with 10,000 samples to test the devised HINTs preconditioners. However, a larger dataset was used to produce the results in the paper, in particular with 250,000 samples. Unfortunately, we are not able to upload that dataset due to its large size. You can generate the data by themselves and train the DeepOnets accordingly. Note, by training DeepOnet using larger datasets, the HINTS preconditioner typically performs better (on average by 20-35%). To run the different examples, execute: 
			

	3.5.1. HYPRE-AMG preconditioner: 
		- using uploaded samples for k=6, you can test the code as
			python3 -u hints_test_hypre_sampled_k.py  --num_samples_total 10000 --num_samples 100000 --k_sigma 6.0 



	3.5.1. HINTS-MG preconditioner, e.g., for the example where DON is trained with only 10,000 on cube 8x8x8 
		- python3 -u hints_test_HINTSgmg_sampled_k.py   --epochs 50000000 --force_retrain false --recreate_data false --only_train false --num_samples_total 10000 --num_samples 10000 --dofs_don 8  --num_basis_functions 128 --k_sigma 6


		- To obtain alternative configurations of HINTS-Jacobi-MG, please explore different values of parameters for --dofs_don, --num_samples, --num_basis_functions, and --k_sigmas. This will require you to create different datasets as well as to train deepOnets, which might of course take a considerable amount of time. 


The typical run time for the examples with 74,712 dofs (sampled dataset and pre-trained DeepONet) is < 1 min per experiment. 
The bigger problems might take a longer time to run. 



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

