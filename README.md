[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/841994693.svg)](https://zenodo.org/doi/10.5281/zenodo.13321073)

# HINTS: This repository contains the code used to generate large-scale results in the paper: <br> **"Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers" by Zhang, Kahana, Kopanicakova, Turkel, Ranade, Pathak, and Karniadakis.**<br> 

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

## Numpy-based code: HINTS methodology and small-scale examples

### Depedencies
scipy>=1.4.1 <br>
torch>=1.8 <br>
tqdm>=4.46.0 <br>
numpy>=1.21 <br>
matplotlib>=3.1.2 <br>


### Hardware requirements
GPU is not necessary but it is highly desirable for training the DeepONets efficiently.
If that case, a proper installation of the GPU drivers (such as CUDE integration, etc.) is expected.


### Installation guide 
1. cd HINTS_numpy <br>
2. Make sure you have python 3.7 installed (or newer, newest version may be inconsistent). <br>
3. Make sure you have pip installed. <br>
4. Open a command line interface and switch to the project folder. <br>
5. Run 'pip install -r requirements.txt' from within the project folder. (Typical installation time on "normal" desktop computer is < 2 mins.) <br>


### Instructions to run the code
The usage of the code is through a file called 'configs.py' located in the HINTS_NP project folder. <br>
1. To train a DeepONet, choose the desired parameters (such as the dimension, the problem, the domain size, etc.). <br>
2. Set the variable 'ITERATION_METHOD' to 'DeepONet' and execute python3 main.py <br>
* This will automatically create data, train a DeepONet and finally throw an error. <br>
The reason for that is the code is able to do 'Only DeepONet' mode, which is not intended in most cases of HINTS. <br>
3. During training, the folder 'debug_figs' will start logging images based on the plotting interval config. <br>
These help monitor the training and see that the network trains well. <br>
4. After training is done, a model is saved into models folder. <br>
5. Set the variable 'ITERATION_METHOD' to 'Numerical_DeepONet_Hybrid' to run the HINTS as: python3 main.py <br>
6. The outputs are logged into the outputs folder. (Typical run time for 1D examples is < 10 mins, including training of DeepONet. Expected output is reported in manuscript, see for example convergence plots on Figure 3B for 1D Poisson equation and HINTS-Jacobi solver.) <br>

To train another DeepONet (change the problem/scenario) the results.npz file in the output folder needs to be deleted. <br>
Alternatively, the flag 'FORCE_RETRAIN' can be set to True. <br>

* We recommend changing the name of the models and the datasets at the bottom of the configs file as per the simulation one wishes to run. 

## PETSc-based code: large-scale hybrid preconditioning for Krylov methods and demonstration of interfacing with state-of-the-art linear algebra
This code of HINTS uses Firedrake for assembly of finite element systems and PETSc for linear algebra, including standard stationary and Krylov methods

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
   
	We have followed these steps: <br>
	2.1. mkdir my_firedrake <br>
 	2.2. cd my_firedrake <br>
	2.3. curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install <br>
	2.4. add support for exodus meshes, i.e., add  "petsc_options.add("--download-exodusii")" to line 745 of firedrake-install script <br>
	2.5. python3 firedrake-install --disable-ssh --no-package-manager <br>


### Instructions to run the code
3.1. source the firedrake enviroment, i.e., <br> 
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  . /my_path/my_firedrake/firedrake/bin/activate <br>
3.2. cd HINTS_petsc <br>
3.3. Export path to HINTS_petsc code, i.e.,   <br> 
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; export PYTHONPATH=$PYTHONPATH:my_path/HINTS_petsc <br>
3.4. cd example <br>


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
Alena Kopaničáková (<alena.kopanicakova@usi.ch>), 
Adar Kahana (<adar.kahana@brown.edu>)


## Contributors: 
* Alena Kopanicakova (Brown, Providence; USI, Lugano; ANITI, Toulouse)
* Adar Kahana (Brown, Providence)
* Enrui Zhang (Brown, Providence)

