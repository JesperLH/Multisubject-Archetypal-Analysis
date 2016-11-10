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

## Demonstrators:
* demoMSAA,demoMSAA_T
	- Demostrates the algorithms and their optional parameters.
* demoVisualizeAA
	- Demostrates how to visualize the found archetypes (Visualizations requires the VITLAB toolbox, avaliable at https://github.com/JesperLH/VITLAM).

## References
Archetypal analysis was first proposed by Cutler and Brieman [1]. The extension to heteroscedastic noise and ability to model multiple subjects was introduced in by Hinrich et al. [2]. The solution of AA using projected gradient descent and the FurthestSum initialization was proposed by Mørup and Hansen[3]. While these reference provides the basis for the implementation of MSAA there are many other interesting approaches in AA and its application.

 * [1] Cutler, A., & Breiman, L. (1994). Archetypal analysis. Technometrics, 36(4), 338-347.
 * [2] Hinrich, J. L., Bardenfleth, S. E., Røge, R. E., Churchill, N. W., Madsen, K. H., & Mørup, M. (2016). Archetypal Analysis for Modeling Multisubject fMRI Data. IEEE Journal of Selected Topics in Signal Processing, 10(7), 1160-1171.
 * [3] Mørup, M., & Hansen, L. K. (2012). Archetypal analysis for machine learning and data mining. Neurocomputing, 80, 54-63.
