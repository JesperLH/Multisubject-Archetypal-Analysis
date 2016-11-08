function [results_subj,C,cost_fun,varexpl,time_taken]=MultiSubjectAA(subj,noc,varargin)
%% Spatial Multi-Subject Archetypal Analysis (Spatial MS-AA)
%  (with heteroscedastic noise on 2. dimension of X (i.e. voxels).
%
%   Solves the AA problem for multiple subjects, with a heteroscedastic or 
%    homoscedastic noise model (default: heteroscedastic). The model was
%    originally developed for high dimensional functional magnetic
%    resonance imaging (fMRI) data, as described in [1].
%
%   results_subj = MultiSubjectAA(subj,noc) returns a struct with the
%   corresponding archetypal scores S, archetypal loadings (archetypes)
%   sXC, noise variance sigmaSq and some error statistics (described in the
%   output section) for each subjects data matrix subj.X reconstructed by
%   subj.sX . The number of archetypes are given by input parameter noc.
%
%   [results_subj,C] = MultiSubjectAA(subj,noc) returns the shared
%   archetypal generator matrix C, describing which timepoints from sX are
%   used to create the archetypes sXC. 
%
%   [results_subj,C,cost_fun] = MultiSubjectAA(subj,noc) returns the
%   negative log-likelihood for the model at each iteration.
%
%   [results_subj,C,cost_fun,varexpl,time_taken]=MultiSubjectAA(subj,noc)
%   returns the algorithm runtime in seconds. 
%
%   [...] = MultiSubjectAA(subj,noc,varargin) allows for OPTIONAL input
%   parameters which are described in the section below.
%
%% Input:
% subj         A structure of data arrays
% subj.X       Data array of size [T x V], T: Time, V: Voxels
% subj.sX      Data array of size [T x sV] 
%              (can be a modification of X, or an entirely different matrix,
%               from which the archetypes are based on.) 
% noc           number of components
% opts         (Struct containing:
%   'maxiter'             maximum number of iterations (default: 100 iterations)
%   'conv_crit'           The convergence criteria (default: 10^-6 relative change in NLL)
%   'fix_var_iter'        Number of iterations to keep the voxel variance
%                           fixed to avoid overfitting the noise model during initial training
%                           (default: 5)
%   'use_gpu'             Accelerate calculations by using GPU (default:
%                           'false', a double precision GPU is recommended)
%   'heteroscedastic'     Model individual voxel noise (default: 'true')
%   'numCstep'            Steps in optimisation of C (default: 10)
%   'numSstep'            Steps in optimisation of S (default: 20)
%   'sort_crit'           Criteria for sorting archetypes (default: 'corr')
%   'init_method'         Method for initializing C (default: 'FurthestSum')
%   'initSstep'           No. of line searches to initialize S(default: 250)
%
%% Output:
% final_subj               Structure of data arrays
% final_subj.S             A [noc x V] archetypal scores matrix, S>=0 |S_j|_1=1
% final_subj.sXC           A [sV x noc] archetypal loadings matrix (i.e.
%                           sXC = sX*C forming the archetypes) 
% final_subj.sigmaSq       A [V x 1] vector with each voxels variance
% final_subj.NLL           Negative Log-likelihood
% final_subj.SSE           Sum of Squares Error (Reconstruction error) 
% final_subj.SST           Sum of Squares Total 
% final_subj.SST_sigmaSq   Sum of Squares Total scaled by fitted noise
%                           variance sigmaSq .
% C                        The [V x noc] archetype generator matrix, 
%                           $C>=0 |C_j|_1=1$, where $C_vj>0$ indicate that
%                           voxel v is part of archetype j .
% cost_fun                 A [iterations x 1] vector of the negative log
%                           likelihood over at each iteration. 
% varexpl                  Percent variation explained by the model 
%                           (Does not account for influence by noise modeling)
%
%% References
% [1] Hinrich, J. L., Bardenfleth, S. E., Røge, R. E., Churchill, N. W.,
%     Madsen, K. H., & Mørup, M. (2016). Archetypal Analysis for Modeling
%     Multisubject fMRI Data. IEEE Journal of Selected Topics in Signal
%     Processing, 10(7), 1160-1171.  
%
% Written by Jesper L. Hinrich, Sophia E. Bardenfleth and Morten Mørup
%
% Copyright (C) 2016 Technical University of Denmark - All Rights Reserved
% You may use, distribute and modify this code under the
% terms of the Multisubject Archetypal Analysis Toolbox license.
% You should have received a copy of the Multisubject Archetypal Analysis Toolbox
% license with this file. If not, please write to: jesper dot hinrich at gmail dot com, 
% or visit : https://brainconnectivity.compute.dtu.dk/ (under software)

warning('off','MATLAB:dispatcher:InexactMatch')

if nargin>=3, opts = varargin{1}; else opts = struct; end
conv_crit=mgetopt(opts,'conv_crit',10^-6);
maxiter=mgetopt(opts,'maxiter',100);
fix_var_iter = mgetopt(opts,'fix_var_iter',5);
runGPU = mgetopt(opts,'use_gpu',false);
voxelVariance = mgetopt(opts,'heteroscedastic',true);
numCstep = mgetopt(opts,'numCstep',10);
numSstep = mgetopt(opts,'numSstep',20);
sort_crit = mgetopt(opts,'sort_crit','corr');
init_type = mgetopt(opts,'init','FurthestSum');
initial_S_steps = mgetopt(opts,'initSstep',250);

if ~isempty(mgetopt(opts,'rngSEED',[]))%Fix random seed
    rng(mgetopt(opts,'rngSEED',[]));
end

V = size(subj(1).X,2); %Voxels
sV= size(subj(1).sX,2); %Tilte Voxels
B = length(subj); % Number of subjects
for i = 1:B,
    %Timesteps for subj i
    subj(i).T = size(subj(i).sX,1);
end

if strcmpi(init_type,'FurthestSum')
    % Initialize C, from all subjects combined
    fprintf('Stacking X matrices\n')
    t0 = [0 cumsum([subj.T])];
    Xcombined = zeros(t0(end),sV);
    for j = 1:B,
        Xcombined((t0(j)+1):t0(j+1),:) = subj(j).sX;
    end
    fprintf('Running furthestSum\n')
    % Initialize C by furthest sum, always on CPU (memory)
    %TODO: Calc on GPU, problem should be resolved
    i=FurthestSum(Xcombined,noc,ceil(sV*rand));
    fprintf('Completed FurthestSum\n')
    clear Xcombined
    clear t0;
    
    if (runGPU),
        fprintf('Moving data to GPU\n')
        C=gpuArray(full(sparse(gather(i),1:noc,ones(1,noc),sV,noc)));
        muC = gpuArray(1);
    else
        C=sparse(i,1:noc,ones(1,noc),sV,noc);
        muC=1;  %Step size
    end
else %random initialise
    if (runGPU),
        muC = gpuArray(1);
    else
        muC=1;  %Step size
    end
    C=rand(sV,noc,'like',muC);
    C=bsxfun(@rdivide,C,sum(C));
end

%Initialize SST
for i=1:B
    if runGPU, %Convert to GPU arrays
        subj(i).sX = gpuArray(subj(i).sX);
        subj(i).X = gpuArray(subj(i).X);
        subj(i).T = gpuArray(subj(i).T);
    end
    
    %Sum of Squares Total (SST) for i'th subj (sigma == 1)
    subj(i).SST = sum(sum(bsxfun(@times,subj(i).X,subj(i).X)));
end
SST = sum([subj.SST]);

% Initialise S and sufficient statistics for each subject
for i = 1:B
    %Initialise variance
    if voxelVariance
        subj(i).sigmaSq = ones(V,1,'like',muC)*SST/(sum([subj.T])*V);
    else
        subj(i).sigmaSq = ones(V,1,'like',muC);
    end
    
    % Initialise S
    subj(i).S = -log(rand(noc,V,'like',muC));
    subj(i).muS=ones('like',muC); %Step size
    subj(i).S = bsxfun(@rdivide,subj(i).S,sum(subj(i).S));
    
    %Initialise sufficient statistics
    subj(i).sXC = subj(i).sX*C;
    subj(i).XCtX=subj(i).sXC'*subj(i).X; % (X*C)^T*X
    subj(i).CtXtXC=subj(i).sXC'*subj(i).sXC;
    subj(i).SSt = subj(i).S*subj(i).S';
    subj(i).XSt = subj(i).X*subj(i).S';
    
    % Perform update of S
    [subj(i).S,~,subj(i).SSt]=SupdateIndiStep(subj(i).S,...
        subj(i).XCtX,...
        subj(i).CtXtXC,...
        subj(i).muS*ones(1,size(subj(i).S,2)),...
        subj(i).T,initial_S_steps,...
        subj(i).sigmaSq);
    
    subj(i).muS=ones(1,size(subj(i).S,2),'like',muC);
end

NLL = 0; SST_sigmaSq =0;
% Initialise with variance term
for i = 1:B
    % Update sufficient statistics
    subj(i).XSt = subj(i).X*bsxfun(@rdivide,subj(i).S,subj(i).sigmaSq')';
    subj(i).SSt = subj(i).S*bsxfun(@rdivide,subj(i).S,subj(i).sigmaSq')';
    
    % Update cost function terms
    subj(i).SST_sigmaSq=sum(sum(subj(i).X.*bsxfun(@rdivide,subj(i).X,subj(i).sigmaSq')));
    subj(i).NLL = 0.5*subj(i).SST_sigmaSq...
        -sum(sum(subj(i).sXC.*subj(i).XSt))...
        +0.5*sum(sum(subj(i).CtXtXC.*subj(i).SSt))...
        +subj(i).T/2*(V*log(2*pi)+sum(log(subj(i).sigmaSq)));
    
    SST_sigmaSq = SST_sigmaSq+subj(i).SST_sigmaSq;
    NLL = NLL+subj(i).NLL;
end
% Set PCHA parameters
iter=0;
dNLL=inf;
t1=cputime;
time_taken = cputime;

%% Display algorithm profile
fprintf(['Principal Convex Hull Analysis / Archetypal Analysis\n'...
         'A ' num2str(noc) ' component model will be fitted\n'...
         'To stop algorithm press control C\n']);
dheader = sprintf('%12s | %12s | %12s | %12s | %16s | %19s|%12s ','Iteration','Cost func.','Delta NLLf.','muC','min(median(muS))','min(median(sigmaSq))',' Time(s)   ');
dline = sprintf('-------------+--------------+--------------+--------------+------------------+---------------------+--------------+');

% Threshold for sigmaSq (numerical stability)
if ~isfield(opts,'noise_threshold')
    var_threshold = SST/(sum([subj.T])*V)*1e-3;
else
    var_threshold = opts.noise_threshold;
end
cost_fun = zeros(maxiter,1);
told=t1;
while (abs(dNLL)>=conv_crit*abs(NLL) || fix_var_iter >= iter) && iter<maxiter
    if mod(iter,100)==0
        disp(dline); disp(dheader); disp(dline);
    end
    told=t1;
    iter=iter+1;
    NLL_old=NLL;
    cost_fun(iter) = gather(NLL);
    
    %% C update
    [C,muC]=CupdateMultiSubjects(subj,C,muC,NLL,numCstep);
    
    % Update terms effected by new C
    for i = 1:B
        %Update sufficient statistics
        subj(i).sXC = subj(i).sX*C;
        subj(i).XCtX = subj(i).sXC'*subj(i).X; %For S update
        subj(i).CtXtXC = subj(i).sXC'*subj(i).sXC;
        
        %Update cost function
        subj(i).NLL =  0.5*subj(i).SST_sigmaSq...
            -sum(sum(subj(i).XCtX.*bsxfun(@rdivide,subj(i).S,subj(i).sigmaSq')))...
            +0.5*sum(sum(subj(i).CtXtXC.*subj(i).SSt))...
            +subj(i).T/2.0*(V*log(2*pi)+sum(log(subj(i).sigmaSq)));
    end
    
    %% Update S and sigmaSq for each subject
    NLL = 0;
    SST_sigmaSq = 0;
    for i=1:B,
        % Update S
        [subj(i).S,subj(i).muS,subj(i).SSt]=SupdateIndiStep(subj(i).S,...
            subj(i).XCtX,...
            subj(i).CtXtXC,...
            subj(i).muS,...
            subj(i).T,numSstep,...
            subj(i).sigmaSq);
        
        % Update voxels specific variance
        if (voxelVariance && (iter > fix_var_iter) )
            %Sigma update, threshold nessecary due to numerical precision
            subj(i).sigmaSq = sum(bsxfun(@power,subj(i).X-subj(i).sXC*subj(i).S,2))'/subj(i).T;
            subj(i).sigmaSq(subj(i).sigmaSq < var_threshold) = var_threshold;
            
            %Update sufficient statistics
            subj(i).XSt = subj(i).X*bsxfun(@rdivide,subj(i).S,subj(i).sigmaSq')';
            subj(i).SSt = subj(i).S*bsxfun(@rdivide,subj(i).S,subj(i).sigmaSq')';
            
            %Update cost function terms
            subj(i).SST_sigmaSq = sum(sum(subj(i).X.*bsxfun(@rdivide,subj(i).X,subj(i).sigmaSq')));
        else
            subj(i).XSt = subj(i).X*bsxfun(@rdivide,subj(i).S,subj(i).sigmaSq')';
        end
        subj(i).NLL = 0.5*subj(i).SST_sigmaSq...
            -sum(sum(subj(i).sXC.*subj(i).XSt))...
            +0.5*sum(sum(subj(i).CtXtXC.*subj(i).SSt))...
            +subj(i).T/2.0*(V*log(2*pi)+sum(log(subj(i).sigmaSq)));
        
        SST_sigmaSq = SST_sigmaSq+subj(i).SST_sigmaSq;
        NLL = NLL+subj(i).NLL;
    end
    
    % Evaluate and display iteration
    dNLL=NLL_old-NLL;
    if rem(iter,5)==0
        t1=cputime;
        fprintf('%12.0f | %12.4e | %12.4e | %12.4e | %16.4e | %19.4e |%12.4f\n'...
            ,iter,NLL,dNLL/abs(NLL),muC,min(median([subj(:).muS])),min(min([subj.sigmaSq])),t1-told);
    end
    
    %Check for convergence issues
    if (dNLL/abs(NLL)<0) && (abs(dNLL/NLL)>conv_crit)
        warning(['Negative Log-likelihood did not monotonically descrease. '...
                'A relative increase of %6.4e was observed.'],dNLL/abs(NLL))
    end
end
% display final iteration
time_taken = cputime - time_taken;

%Sum of squares error for each subj
SSE = zeros(B,1);
for i = 1:B,
    SSE(i) = norm(subj(i).X-(subj(i).sX*C)*subj(i).S,'fro')^2;
end
varexpl=(SST-sum(SSE))/SST;

disp(dline);
disp(dheader);
disp(dline);
fprintf('%12.0f | %12.4e | %12.4e | %12.4e | %16.4e | %19.4e |%12.4f\n',iter,NLL,dNLL/abs(NLL),muC,min(median([subj(:).muS])),min(min([subj.sigmaSq])),t1-told);
fprintf('Algorithm converged in %f sec. . Reconstruction covers %3.2f%% of data variance.\n',time_taken,varexpl*100)

%% Sort components by selected criteria
if strcmpi(sort_crit,'corr') && gather(sum([subj.T]) == max([subj.T])*B)
    % Mean correlation of each archetype
    arch = zeros(gather(subj(1).T),B);
    mean_corr = zeros(noc,1);
    for j = 1:noc
        for i = 1:B
            arch(:,i) = gather(subj(i).sXC(:,j));
        end
        comp_corr = triu(corr(arch),1);
        mean_corr(j) = mean(comp_corr(comp_corr(:)~=0));
    end
    [~,ind] = sort(mean_corr,'descend');
    
else % Sort by usage/energy in S
    energy = sum([subj.S],2);
    [~,ind]=sort(energy,'descend');
    
    if strcmpi(sort_crit,'corr')
        warning(['Second dimension of the data matrix X varied between '...
                 'subjects. Sorting by average correlation is not supported,'...
                 ' sorted by ''energy'' instead.'])
    end
end
% Apply new indexing
C=C(:,ind);
for i = 1:B,
    subj(i).S=subj(i).S(ind,:);
    subj(i).sXC = subj(i).sXC(:,ind);
end

%% Collect data from GPU memory
C = gather(C);
for i = B:-1:1,
    results_subj(i).S = gather(subj(i).S);
    results_subj(i).sXC = gather(subj(i).sXC);
    results_subj(i).sigmaSq = gather(subj(i).sigmaSq);
    results_subj(i).NLL = gather(subj(i).NLL);
    results_subj(i).SSE = gather(SSE(i));
    results_subj(i).SST = gather(subj(i).SST);
    results_subj(i).SST_sigmaSq = gather(subj(i).SST_sigmaSq);
end

cost_fun = cost_fun(1:iter);
end

%% --------------------------------------------------------------------
% ------- Updating the shared generator matrix C
function [C,muC]=CupdateMultiSubjects(subj,C,muC,NLL,niter)
if nargin<6, niter=1; end

[sV,noc]=size(C);
V = size(subj(1).X,2);

%Constant term
for i = length(subj):-1:1,
    temp(i).XtXSt = subj(i).sX'*subj(i).XSt;
end

for k=1:niter %Number of line searches to perform
    NLL_old=NLL;
    
    %Calculate gradient
    g=zeros(sV,noc);
    for i = 1:length(subj),
        g=g+(subj(i).sX'*(subj(i).sXC*subj(i).SSt)-temp(i).XtXSt);
    end
    g=g/(sum([subj.T])*sV);
    g=bsxfun(@minus,g,sum(bsxfun(@times,g,C)));
    
    stop=0;
    Cold=C;
    while ~stop %Line search
        % Update C and do back projection into correct subspace
        C=Cold-muC*g;
        C(C<0)=0;
        C=bsxfun(@rdivide,C,sum(C)+eps);
        
        NLL = 0;
        for i = 1:length(subj),
            subj(i).sXC=subj(i).sX*C; %Smooth
            subj(i).CtXtXC=subj(i).sXC'*subj(i).sXC;
            
            subj(i).NLL = 0.5*subj(i).SST_sigmaSq...
                -sum(sum(subj(i).sXC.*subj(i).XSt))...
                +0.5*sum(sum(subj(i).CtXtXC.*subj(i).SSt))...
                +subj(i).T/2.0*(V*log(2*pi)+sum(log(subj(i).sigmaSq)));  %Can be moved to SST_sigmaSq
            
            NLL=NLL+subj(i).NLL;
        end
        
        if NLL<=NLL_old*(1+1e-9)
            muC=muC*1.2;
            stop=1;
        else
            muC=muC/2;
        end
    end
end
end
%--------------------------------------------------------------------