%% ############ README ################
% This script is made to facility easy reproducability of the synthetic
% experiments done in [1].

%Run order
%% 1. Construct the synthetic dataset 
construct_synthetic_data

%% 2. Run MSAA analysis (also adds noise to the data and save noise for use in ICA)
Run_MSAA_Analysis

%% 3. Compare different noise settings and assumptions
Analyse_Synthetic_Runs


%% For Group ICA
% Export each problem as a .mat file, which is then analysed sepeately. Run
Prepare_synthetic_data
% The synthetic datasets will then be in the folder
% './paperMSAA_SyntheticStudy/synthetic_ica'
% The Group-ICA analysis performed in [1] uses the GIFT toolbox (available
% at http://mialab.mrn.org/software/gift/) using the GICA3 method with
% default settings  

%% References
% [1] Hinrich, J. L., Bardenfleth, S. E., Røge, R. E., Churchill, N. W.,
%     Madsen, K. H., & Mørup, M. (2016). Archetypal Analysis for Modeling
%     Multisubject fMRI Data. IEEE Journal of Selected Topics in Signal
%     Processing, 10(7), 1160-1171.   
