%% Visualize synthetic results
clear; close all
% Options
noise_db = {'+4db','0db','-4db','-8db','-16db'};
f_size=18;
l_size = 2; %linewidth
maxiter=150;
replicates = 3;
noc = 23;
num_noise = length(noise_db);
num_init = 2;
num_noise_type = 2; %Hetero and homo

%Setting for permutation test
numPermutations = 100;

%% Load and save folders
load_folder = './paperMSAA_SyntheticStudy/';
%Load mask
load([load_folder,'data/synth-mask.mat'])

%Load true sources and mixing matrices
load([load_folder, 'data/synthetic_fmri.mat'])
[T,~,B] = size(A);
Vmasked = sum(mask(:));
S_true = S(:,mask(:),1:B);
A_true = A(:,:,1:B);

clear A meanim S

%
load_folder = [load_folder,'/synthetic_results/'];

results_folder = strcat(load_folder,'analysis/');
mkdir(results_folder);

%% Combine results from the different runs
all_NLL = nan(num_noise_type,num_noise,num_init,replicates);
all_NLL_dev = nan(num_noise_type,num_noise,num_init,replicates,maxiter);
all_time = nan(num_noise_type,num_noise,num_init,replicates);
all_best = nan(num_noise_type,num_noise,num_init);
%Testing
all_nmiS = nan(num_noise_type,num_noise,num_init,sum(1:(replicates-1)));
all_nmiSperm = nan(num_noise_type,num_noise,num_init,numPermutations*sum(1:(replicates-1)));
all_mean_corr = nan(num_noise_type,num_noise,num_init,replicates);

type_init = {'FurthestSum','Random'};
num_instances = num_noise_type*num_noise*num_init;
method_name = cell(num_instances,1);
name_id = 1;

for noise_level = 1:num_noise % For each noise level
    for noise_type = [0 1] %For [Homo, Hetero] noise assumption in MSAA
        opts.with_voxel_var = noise_type;
        for init = 1:2;
            %Determine naming
            if noise_type == 0
                method_name{name_id} = strcat('Homo-',type_init{init});
            else
                method_name{name_id} = strcat('Hetero-',type_init{init});
            end
            
            method_name{name_id} = strcat(noise_db{noise_level},'XxX',method_name{name_id});
            method_name{name_id} = strrep(method_name{name_id},'XxX',' ');
            name_id = name_id+1;
            
            %All spatial maps, to do NMI calculation
            Smatrices = nan(noc,B*Vmasked,replicates); %d x B*V x runs
            allC = nan(noc,Vmasked,1,replicates);
            %Load data
            for run = 1:replicates
                fprintf('Loading Run %i of %i : Max noise lvl %i : Init: %s , noise is hetero = %i\n',...
                    run,replicates,noise_level,type_init{init},noise_type)
                % Load data
                if noise_type == 0
                    load(sprintf('%s%i-%i-fullbrain%iiterations-homo%s-%s.mat',...
                        load_folder,noc,run,maxiter,noise_db{noise_level},type_init{init}))
                else
                    load(sprintf('%s%i-%i-fullbrain%iiterations-hetero%s-%s.mat',...
                        load_folder,noc,run,maxiter,noise_db{noise_level},type_init{init}))
                end

                %Gather
                all_NLL(noise_type+1,noise_level,init,run) = cost_fun(end);
                all_NLL_dev(noise_type+1,noise_level,init,run,1:length(cost_fun)) = cost_fun;
                all_time(noise_type+1,noise_level,init,run) = time_taken;
                
                Smatrices(:,:,run) = [final_brain.S];
                
                %% Calcuate mean correlation between true and estimated
                %components.
                S_est = reshape([final_brain.S],noc,Vmasked,B);
                all_mean_corr(noise_type+1,noise_level,init,run) = ...
                    calcMatchedCorrelation(permute(S_true,[2,1,3]),...
                    permute(S_est,[2,1,3]));
            end
            
            [~,all_best(noise_type+1,noise_level,init)] = min(squeeze(all_NLL(noise_type+1,noise_level,init,:)));
            
            %% NMI and permutation test
            fprintf('NMI perm. ')
            tic
            [nmi_S,nmi_Sperm]=nmiPermutationTest(gpuArray(Smatrices),numPermutations);
            toc
            all_nmiS(noise_type+1,noise_level,init,:) = gather(nmi_S);
            all_nmiSperm(noise_type+1,noise_level,init,:) = gather(nmi_Sperm);
            
        end
        
    end
    
end

method_name_old = method_name;

save(strcat(results_folder,'analysis_data_combined'),'all_best','all_NLL','all_NLL_dev','all_nmiS','all_nmiSperm','all_mean_corr','method_name_old')
clear Smatrices

%% Order NMI and correlation measures 
nmi = nan(replicates,num_instances);
nmiPerm = nan(size(all_nmiSperm,4),num_instances);
mean_corr = nan(num_instances,replicates);
counter = 1;
names = cell(num_instances,1);
for noise_level = 1:num_noise
    for noise_type = [0 1] %[Homo, Hetero]
        opts.with_voxel_var = noise_type;
        for init = 1:2;
            %Sort NMI and mean_corr according to noise_level and noise_type
            nmi(:,counter) = squeeze(all_nmiS(noise_type+1,noise_level,init,:));
            nmiPerm(:,counter) = squeeze(all_nmiSperm(noise_type+1,noise_level,init,:));
            mean_corr(counter,:) = all_mean_corr(noise_type+1,noise_level,init,:);
            
            %Make sure names are unfolded correctly
            s=sprintf('%s',noise_db{noise_level});
            if noise_type == 0
                s=sprintf('%s Homo',s);
            else
                s=sprintf('%s Hetero',s);
            end
            s=sprintf('%s %s',s,type_init{init});
            fprintf('%s\n',s)
            names{counter}=s;
            
            counter=counter+1;
        end
    end
end
%Use short names
names = strrep(names,'FurthestSum','FS');
names = strrep(names,'Random','Rand');

%% Make NMI permutation boxplot
figure('Position',[0 0 800 500]);
boxplot(nmi,'colors','k'); hold on
boxplot(nmiPerm,'colors','b'); hold off
ylim([-.1 1.1])
%title('NMI for 5 Solutions and Permutation Test','FontSize',f_size+2)
ylabel('NMI','FontSize',f_size)
%xlabel('Number of archetypes','FontSize',f_size)
%set(gca,'XTickLabels',method_name,'XTickLabelRotation',60,'Fontsize',f_size)
set(gca,'XTickLabels',names,'XTickLabelRotation',90,'Fontsize',f_size)

tightfig();
pbaspect([4 1 1])
print(strcat(results_folder,'nmi'),'-dpng')
print(strcat(results_folder,'nmi'),'-depsc')

%% Calc and print avg and sd corr over replications for each instance
m = mean(mean_corr,2); sd = sqrt(var(mean_corr,[],2));
fprintf('%20s :\t mean coor. \t sd%4s\n','','') 
for i = 1:length(names)
   %fprintf('%20s :\t mean=%2.6e |\t sd=%2.6e\n',names{i},m(i),sd(i)) 
   fprintf('%20s :\t %2.4e |\t %2.4e\n',names{i},m(i),sd(i)) 
end
