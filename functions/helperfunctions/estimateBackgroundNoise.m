function noise_threshold = estimateBackgroundNoise(file,fileFilt)
% file is the raw fMRI data, used to find the mask
% fileFilt is the filtered/preprocessed fMRI data, where the background
% noise is calculated. 
% Note: Both files are expected to be .nii files

%file = 'T:\Data\fMRI\motor_smooth\ID19_4D_motor.nii'
nii=load_nii(file); %load_nii requires the NIfTI toolbox
[x y z T] = size(nii.img);

X = reshape(double(nii.img),x*y*z,T);
not_brain=mean(X')<mean(X(:))*.8; 
%--- Comment these two lins out to estimate the background noise in the raw fMRI data
nii=load_nii(fileFilt); 
[x y z T] = size(nii.img);
%---
filtered_data = reshape(double(nii.img),x*y*z,T);

var_X=var(filtered_data(not_brain,:)'); 
noise_threshold=mean(var_X(var_X>0));

%% Code for showing voxels, that were NOT used to estimate background noise
% figure('position',[10,800,600,400]);
% m = ~not_brain;
% pa = patch(isosurface(reshape(m,[53 63 46])));
% set(pa,'edgecolor','none','facecolor',[0.5 0.5 0.5],'facealpha',0.5);
% axis off;
% axis equal;
% axis tight;
% view(30,30);
% figure; montage(reshape(not_brain,[53 63 1 46]))
