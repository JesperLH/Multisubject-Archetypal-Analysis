%% Add noise to synthetic data, then save it for analysis with ICA.
% Note, the noise is identical to that used in the AA runs. Such that
% different noise generation does not influence the final results.

%Load SimTB generated data
load('./paperMSAA_SyntheticStudy/data/noisefree.mat')
save_folder = './paperMSAA_SyntheticStudy/synthetic_ica/';
load_folder = './paperMSAA_SyntheticStudy/synthetic_results/';
mkdir(save_folder);

sx = size(data,1);
sy = size(data,2);
T=size(data,4);
B= size(data,5);
%Determine active voxels. (i.e. mask the data)
mask = mean(mean(data(:,:,:,:,:),4),5)>mean(data(:))*.2;

noise_db = {'+4db','0db','-4db','-8db','-16db'};
%% For each noise SNR_dB setting, load the generated noise, add it to data
% and save the results to file
for n = 1:length(noise_db)
    %%
    fprintf('Loading %s noise ... ',noise_db{n}); tic
    load(strcat([load_folder,'noise'],noise_db{n},'.mat'));
    toc
    
    fprintf('Adding noise... '); tic
    for b=B:-1:1
        subj(b).X = reshape(data(:,:,:,:,b),sx*sy,T);
        subj(b).X = subj(b).X+reshape(noise(:,:,:,b),sx*sy,T); %Add noise at zero mean

        subj(b).X = reshape(subj(b).X,sx,sy,T);
    end
    toc
    %%
    fprintf('Rearranging... '); tic
    dataNoise = nan(sx,sy,1,T,B);
    for b = 1:B
       dataNoise(:,:,1,:,b) = subj(b).X; 
    end
    toc
    clear subj
    %%
    fprintf('Saving... '); tic
    save(strcat([save_folder, 'dataNoise'],noise_db{n}),'A','dataNoise','meanim','S')%,'noise')
    toc
    fprintf('Done\n')
    clear dataNoise
end
