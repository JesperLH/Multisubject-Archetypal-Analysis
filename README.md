# MSAA - MultiSubject Archetypal Analysis
The Multisubject Archetypal Analysis Toolbox holds several extensions to
ordinary archetypal analysis. Their algorithms implemented in Matlab™ and 
support the use of graphical processing units (GPUs) for high performance 
computing. All code can be used freely in research and other non-profit 
applications. If you publish results obtained with this toolbox we kindly 
ask that our and other relevant sources are properly cited. 

This toolbox has been developed at:

The Technical University of Denmark, 
Department for Applied Mathematics and Computer Science,
Section for Cognitive Systems.

The toolbox was developed in connection with the Brain Connectivity project 
at DTU (https://brainconnectivity.compute.dtu.dk/) .

## Algorithms:

* MultiSubjectAA
	- MSAA with heteroscedastic noise in the first dimension.
* MultiSubjectAA_T   
	- MSAA with heteroscedastic noise in the second dimension.

Common algorithm properties

* Finds archetypes for multisubject data.
* The second dimension can have different length for each subject.
* Ability to individually turn off heteroscedastic noise modeling.
* The log likelihood is calculated.


Demonstrators:
* demoMSAA,demoMSAA_T
	- Demostrates the algorithms and their optional parameters.
* demoVisualizeAA
	- Demostrates how to visualize the found archetypes (REQUIRES VISUALIZATION TOOLBOX).
