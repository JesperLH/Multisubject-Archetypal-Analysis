%% Visualization demo; Multi-Subject Archetypal Analysis
% Requires the VITLAB toolbox, avaliable at https://github.com/JesperLH/VITLAM
clear
load('./demos/example_data_fmri')

%%
nSubs = length(results_subj);
[T, D] = size(results_subj(1).sXC);
V = size(results_subj(1).S,2);
S = reshape([results_subj.sXC],T,D,nSubs);
A = reshape([results_subj.S],D,V,nSubs);

%% Permute dimensions
% So they it fits the expected input of the visualization toolbox
S = permute(S,[2,1,3]); % Now a D x T x nSubs array
A = permute(A,[2,1,3]);

% Average spatial maps
Am = mean(A,3);

%% Plotting a subset or using variable arguments
subset = 1:10;
plotComponents(Am(:,subset),S(subset,:,:),mask,...,)
            'inConvention','Radiological',... %Input convention (i.e. data)
            'outConvention','Neurological',...%Output convention
            'TR',2.49,... %Time resolution in secounds
            'FontSize',18,... % Self-explanatory. Note titles has Fontsize+2 
            'LineWidth',2,... % Width of the temporal activation
            'save','',... %Saves .png and .eps to the given directory, if '' figures aren't saved
            'Position',[0 0 800 450],... %Figure position lower left and upper right cornor
            'threshold',[0.2:0.1:0.9],... 
            'Scaling','absmax');
% Note it is good practise to let the threshold be positive, but it is not necessary to specify, as the AA maps are strictly positive.
%% Plotting only spatial maps
plotComponents(Am(:,1),[],mask,'Position',[50 50 800 350])

%% Plotting with custome coloring
rng('default')
custom_color = rand(1,3,2);
plotComponents(Am(:,2:3),S(2:3,:,:),mask,'threshold',0.1:0.1:0.9,'color',custom_color)

%% Visualizing the noise map
plotNoisemap([results_subj.sigmaSq],mask,mask_affine_mat)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualise seed generators
plotBloatedComponents(C,mask,'threshold',0)

%% Or if only a single plot is needed
figure
[clab,cticks] = plotBrainBloatedVoxels(C,mask,'threshold',0);

%% Alternatively, it can also be used on the (average) spatial maps
% By varying the threshold (between 0 and 1) interesting patterns can
% appear
figure('Units','normalized','position',[0.2,0.2,0.7,0.3])
subplot 131
plotBrainBloatedVoxels(Am,mask,'threshold',0,'colorbar',true)
subplot 132
plotBrainBloatedVoxels(Am,mask,'threshold',0.25,'colorbar',true)
subplot 133
plotBrainBloatedVoxels(Am,mask,'threshold',0.5,'colorbar',true)
