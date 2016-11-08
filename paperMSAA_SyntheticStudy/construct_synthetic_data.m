%% Constructs a synthetic fMRI dataset
% The sources and mixing matrices (A and S) have been simulated by the
% fMRI Simulation Toolbox (SimTB, % http://mialab.mrn.org/software/simtb/).
% This script multiplies A and S for each subject, then add a subject
% specific mean value, in order to obtain the noisefree data for each
% simulated subject.
%
clear
file_folder = '.\paperMSAA_SyntheticStudy\data\';
load([file_folder, 'synthetic_fmri']);
%% Construct data
% 400 timepoints, 23 components, 12 subjects
[T,D,B] = size(A);
% One slice with 148 x 148 voxels = 21904 voxels for each subject
[~,V,~] = size(S);
AS = zeros(T,V,B);
for b = 1:B
    AS(:,:,b) = A(:,:,b)*S(:,:,b);
end
% Permute and reshape into 148 x 148 x 1 x 400 x 12
AS = permute(AS,[2,1,3]);
AS = reshape(AS,[sqrt(V),sqrt(V),1,T,B]);

% Add mean
data = bsxfun(@plus,AS,meanim);
clear AS;
%%
save([file_folder, 'noisefree'],'A','S','meanim','data')