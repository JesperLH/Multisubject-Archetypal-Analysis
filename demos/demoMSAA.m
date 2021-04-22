clear; close all;
%% Problem setting
noc = 10; %number of components
B = 3; % number of subjects
Timesteps = 40; %Observations
Voxels = 300; % Dimensions

% Data generation
for b=B:-1:1
   X = randn(Timesteps,Voxels).^3; % Zero mean, double precission
   X = bsxfun(@minus,X,mean(X)); % Voxels have zero mean over time, 
   subj(b).X = X;
   %sX does not have to be different from X. However it is designed to
   %allow comparison with a smoothed version of X, or to be used as a
   %search/spot-light approach. 
   subj(b).sX = X;
   %subj(b).sX = X(:,1:ceil(Voxels/2));
end

%% Optimization parameters
opts.maxiter = 100; 
opts.conv_crit = 1e-8;
opts.fix_var_iter = 5; 
opts.use_gpu = false; 
opts.heteroscedastic = true;
opts.rngSEED = 12345; %Ability to fix the seed for the random generator

%% Finding a solution
[output_subj,C,cost_fun,varexpl,time_taken]=MultiSubjectAA(subj,noc,opts);
% C are the shared archetypal generators
% output_subj(b).sXC are the archetypes for subject b
% output_subj(b).S are the archetypal mixing matrix for subject b
% output_subj(b).sigmaSq are the voxel specific noise for subject b

%%
figure; plot(cost_fun); title('Evolution of Negative Log-likelihood')
xlabel(sprintf('Iterations (The sharp dip around iteration %i is due to \nthe voxel variance nolonger being fixed)',opts.fix_var_iter))