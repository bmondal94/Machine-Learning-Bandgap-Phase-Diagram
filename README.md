# Machine Learning Bandgap Phase Diagram

Please visit here for more details:

* https://bmondal94.github.io/Bandgap-Phase-Diagram/

   <img src="./ImageFolder/BandgapPhaseDiagram.png" style="width:300px;height:300px;">
   
## Introduction
The tuning of the type and size of bandgaps of III-V semiconductors is a major goal for optoelectronic applications. 
Varying the relative composition of several III- or V-components in compound semiconductors is one of the major approaches here. 
Alternatively, straining the system can be used to modify the bandgaps. By combining these two approaches, 
bandgaps can be tuned over a wide range of values, and direct or indirect semiconductors can be designed. However, an optimal choice of composition and 
strain to a target bandgap requires complete material-specific composition, strain, and bandgap knowledge. Exploring the vast chemical space of all 
possible combinations of III- and V-elements with variation in composition and strain is experimentally not feasible. We thus developed a 
density-functional-theory-based predictive computational approach for such an exhaustive exploration. This enabled us to construct the 
'bandgap phase diagram' by mapping the bandgap in terms of its magnitude and nature over the whole composition-strain space. 
Further, we have developed efficient machine-learning models to accelerate such mapping in multinary systems. We show the application and great 
benefit of this new predictive mapping on device design. 

## General density-functinal-theory computational setup
General computational setup.

* Periodic DFT using VASP-5.4.4
* Geometry optimization: PBE-D3 (BJ), PAW basis set 
* Electronic properties: TB09 [[1]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.102.226401), PAW basis set, spin-orbit coupling 
* Super cell : 6x6x6, 10 SQS [[2]](https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node74.html), 
Γ-only, band unfolding [[3]](https://github.com/rubel75/fold2Bloch-VASP),[[4]](https://github.com/band-unfolding/bandup)

## Machine learning (ML) models

* Supervised model
	* ML model: Support Vector Machine with Radial Basis Function kernel (SVM-RBF) 
		* Bandgap magnitude prediction: Support Vector Regression
		* Bandgap nature prediction: Support Vector Classification
		* Classification type:  binary (direct or indirect bandgap)
## Softwares

The ML models are implemented in `python scikit-learn` package. To run the above script you need the following packages:

* python >3.6
* matplotlib
* scikit-learn
* tensorflow (if NN section is used)
* pandas
* sqlite3
* numpy
* pickle
* python-ternary
* tensorflow_docs
* bokeh (if html plot section is used)
* holoview (if html plot section is used)

## References
* III-V semiconductors bandgap phase diagram
    *  Binary compounds: [arXiv](http://arxiv.org/abs/2208.10596), [NOMAD repository](https://doi.org/10.17172/NOMAD/2022.08.20-2)
    *  Ternary compounds: [arXiv](http://arxiv.org/abs/2302.14547), [NOMAD repository](https://doi.org/10.17172/NOMAD/2023.02.27-1)

Please contact to [Badal Mondal](mailto:badalmondal.chembgc@gmail.com) for further details.
