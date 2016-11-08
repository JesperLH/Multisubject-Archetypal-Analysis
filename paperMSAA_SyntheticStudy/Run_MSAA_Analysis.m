clear; close all

SEED = 64531168;
rng(SEED);

load_folder = '.\paperMSAA_SyntheticStudy\data\';
load([load_folder,'noisefree'])

save_folder = '.\paperMSAA_SyntheticStudy\';
save_folder_mask = [save_folder, 'data\'];
save_folder_noiseforICAanalysis = [save_folder, 'synthetic_results\'];
save_folder = [save_folder, 'synthetic_results\'];
mkdir(save_folder);
mkdir(save_folder_mask);
mkdir(save_folder_noiseforICAanalysis);

%Data size
%data = data(:,:,:,:,1:2);
sx = size(data,1);
sy = size(data,2);
T=size(data,4);
B= size(data,5);

%Find mask
mask = mean(mean(data(:,:,:,:,:),4),5)>mean(data(:))*.2;
save(strcat(save_folder_mask,'synth-mask'),'mask')
figure; imagesc(mask)

%% Make noise variance maps
noise_db = {'+4db','0db','-4db','-8db','-16db'};
noise_var_max = [0.125, 0.325, 0.82, 2.05, 13];

% Heteroscedastic noise (DIFFERENT noisemap for each noise level)
%noise_var = randn(sx,sy,length(noise_var_max));

% Heteroscedastic noise (SAME noisemap for each noise level)
noise_var = repmat(0.9*rand(sx,sy)+0.1,1,1,length(noise_var_max)); %

model_hetero = true;
for i = 1:length(noise_var_max)
    if model_hetero
        noise_var(:,:,i) = noise_var(:,:,i)*noise_var_max(i);
    else %Homoscedastic noise
        noise_var(:,:,i) = mask*noise_var_max(i);
    end
end

initTypes = {'FurthestSum','Random'}; %Types of initializations
replicates = 3; % Number of repeated analysis for a given setting
noc = 23; % Number of components to look for in the MSAA model.

%Generate a random seed for each setting
total_runs = replicates*length(initTypes)*2*size(noise_var,3);
rngSeeds=randi([0 10^6],total_runs,1);
%% Opt
opts.maxiter=150; 
opts.conv_crit=1e-6;
opts.fix_var_iter = 25; 
opts.use_gpu = true;
if ~opts.use_gpu, warning('******** NOT USING GPU ********'); end

%% Order data for AA analysis
seed_num = 1; %random seed index
% For each noise setting (i.e. snrdB = +4,0,-4,-8,-16)
for noise_level = 1:size(noise_var,3)
    %% Generate  noise
    noise = nan(sx,sy,T,B);
    P_sig = 0;
    %Add heteroscedastic noise to every subject.
    for b=B:-1:1
        %Draw homoscedastic noise for each voxel at each timepoint, then
        %multiply by the standard deviation of the noise variance (noise_var)
        noise(:,:,:,b) = bsxfun(@times,randn(sx,sy,T),sqrt(noise_var(:,:,noise_level)));
        
        X = reshape(squeeze(data(:,:,:,:,b)),sx*sy,T);
        %Calculate power of the noisefree signal for SNR 
        X = bsxfun(@minus,X,mean(X,2)); %zero mean
        P_sig = P_sig+norm(X(mask(:),:),'fro')^2; %Power of signal
        
        % Add noise and store the data in a struct
        subj(b).X = X+reshape(noise(:,:,:,b),sx*sy,T); %Add noise
        subj(b).X = subj(b).X(mask(:),:); %Apply mask
        subj(b).X = bsxfun(@minus,subj(b).X,mean(subj(b).X,2))'; % Substract mean
        
    end
    clear X;
    
    %Calculate signal-to-noise ratio
    P_sig = P_sig/numel([subj.X]);
    P_noise = reshape(noise,sx*sy,T,B); 
    P_noise = P_noise(mask(:),:,:); %Only take noise inside the mask
    P_noise = sum(P_noise(:).^2)/(numel(P_noise));
    snr_db = 10*log10(P_sig)-10*log10(P_noise);
    fprintf('Noise variance (mask only) %3.2f , log SNR %2.4f\n',noise_var_max(noise_level),snr_db)
    
    %Save the noise, such that it can be used for ICA analysis in the GIFT
    %toolbox.
    fprintf('Saving noise for use in the ICA analysis...'); tic
    save(strcat(save_folder_noiseforICAanalysis,'noise',noise_db{noise_level}),'noise','snr_db')
    clear noise;
    toc;
    
    %% Setup "modified" X for each subject as required by MSAA
    for b = 1:B
        subj(b).sX = subj(b).X;
    end
    
    %% Run simulations
    for isheteroscedastic = [false, true] %[Homo, Hetero] noise assumption
        opts.with_voxel_var = isheteroscedastic;
        for init = initTypes;
            init = init{1}; %#ok<FXSET>
            opts.init = init;
            for run = 1:replicates
                opts.rngSEED=rngSeeds(seed_num);
                seed_num = seed_num+1;
                fprintf('Run %i of %i : noise lvl %3.2e : Init: %s , noise is hetero = %i\n'...
                    ,run,replicates,noise_level,init,isheteroscedastic)
                
                % Run MSAA algorithm
                [final_brain,C,cost_fun,varexplained,time_taken]=...
                    MultiSubjectAA(subj,noc,opts);
                
                %Save results
                if isheteroscedastic == 0
                    save(sprintf('%s%i-%i-fullbrain%iiterations-homo%s-%s.mat',...
                        save_folder,noc,run,opts.maxiter,noise_db{noise_level},opts.init)...
                        ,'final_brain','time_taken','varexplained','C','cost_fun')
                else
                    save(sprintf('%s%i-%i-fullbrain%iiterations-hetero%s-%s.mat',...
                        save_folder,noc,run,opts.maxiter,noise_db{noise_level},opts.init)...
                        ,'final_brain','time_taken','varexplained','C','cost_fun')
                end
                
            end
            
 
        end
        
    end
    
end
