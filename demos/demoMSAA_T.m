clear; close all;
%% Problem setting
noc = 10; %number of components
B = 3; % number of subjects
Timesteps = 40; %Observations
Voxels = 300; % Dimensions

%% Data generation
%%Archetypes XC are spatial with spatial noise (X is Voxels x Time)
for b=B:-1:1
   % X is transposed compared to MSAA
   X = randn(Voxels,Timesteps); % double precission
   X = bsxfun(@minus,X,mean(X,2)); % Voxels have zero mean over time, 
   subj(b).X = X;
   %sX does not have to be different from X. However it is designed to
   %allow comparison with a smoothed version of X, or to be used as a
   %search/spot-light approach. 
   subj(b).sX = X; 
   subj(b).sX = X(:,1:ceil(Timesteps/2));
end

%%Archetypes XC are temporal with temporal noise (X is Time x Voxels)
% In contrast to regular MS-AA, which has temporal archetypes, but spatial
% noise.
% for b=B:-1:1
%    % X is transposed compared to MSAA
%    X = randn(Timesteps,Voxels); % double precission
%    X = bsxfun(@minus,X,mean(X)); % Voxels have zero mean over time, 
%    subj(b).X = X;
%    %sX does not have to be different from X. However it is designed to
%    %allow comparison with a smoothed version of X, or to be used as a
%    %search/spot-light approach. 
%    subj(b).sX = X; 
%    subj(b).sX = X(:,1:ceil(Voxels/2)); 
% end

%% Optimization parameters
opts.maxiter = 100; 
opts.conv_crit = 1e-8;
opts.fix_var_iter = 5; 
opts.use_gpu = false; 
opts.heteroscedastic = true;
opts.rngSEED = 12345; %Ability to fix the seed for the random generator

%% Run the algorithm
[results_subj,C,cost_fun,varexpl,time_taken]=MultiSubjectAA_T(subj,noc,opts);
% C are the shared archetypal generators
% output_subj(b).sXC are the archetypes for subject b
% output_subj(b).S are the archetypal mixing matrix for subject b
% output_subj(b).sigmaSq is the voxel specific noise for subject b

%%
figure,plot(cost_fun); title('Evolution of Negative Log-likelihood')
xlabel(sprintf('Iterations (The sharp dip around iteration %i is due\n to the voxel variance nolonger being fixed)',opts.fix_var_iter))
ylabel('Negative Log-likelihood')