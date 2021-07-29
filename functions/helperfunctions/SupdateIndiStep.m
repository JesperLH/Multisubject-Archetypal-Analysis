%% -------------------------------------------------------------------------
%Individual subject voxels step size
% -------------------------------------------------------------------------
% In MS-AA with spatial archetypes (i.e. X is Time x Voxels) and spatial
% (voxels) noise, the inputs are:
%   S:          components x Voxels
%   XCtX:       components x Voxels
%   CtXtXC:     components x components
%   muS:        1 x Voxels
%   numObs:     Is a scalar with the number of timesteps (i.e. Time)
%   niter:      Is the number of line searches to perform
%   sigmaSq:    Voxels x 1 (the noise variance is part of the SSt 
%                           sufficient statistic)
%
% In MS-AA with temporal archetypes (i.e. X is Voxels x Time) and spatial
% (voxels) noise, the inputs are:
%   S:          components x Time
%   XCtX:       components x Time
%   CtXtXC:     components x components
%   muS:        1 x Time
%   numObs:     Is a scalar with the number of voxels (i.e. Voxels)
%   niter:      Is the number of line searches to perform
%   sigmaSq:    Should not be passed (as it not part of the SSt sufficient
%                                     statistic)
%Note sigmaSq is not needed during the update, as it can be considered a
%scale difference, which is irrelevant when using individual stepsize for S
%
%% Written by Jesper L. Hinrich, Sophia E. Bardenfleth and Morten Mørup
%
% Copyright (C) 2016 Technical University of Denmark - All Rights Reserved
% You may use, distribute and modify this code under the
% terms of the Multisubject Archetypal Analysis Toolbox license.
% You should have received a copy of the Multisubject Archetypal Analysis Toolbox
% license with this file. If not, please write to: jesper dot hinrich at gmail dot com, 
% or visit : https://brainconnectivity.compute.dtu.dk/ (under software)
function [S,muS,SSt]=SupdateIndiStep(S,XCtX,CtXtXC,muS,numObs,niter,sigmaSq)

if isscalar(muS), muS=ones(1,numObs); else, muS=muS(:)'; end

[~,numFeature]=size(S);
cost = -2*(sum(S.*XCtX))+sum(S.*(CtXtXC*S));
for k=1:niter
    g=(CtXtXC*S-XCtX)/(numObs*numFeature); %(U_s2) gradient
    g=bsxfun(@minus,g,sum(bsxfun(@times,g,S)));
    Sold=S;
    S = Sold-bsxfun(@times,g,muS);
    S(S<0)=0; %(U_s1) S>= 0 constraint
    S=bsxfun(@rdivide,S,sum(S)); %(U_s2) l1-normalisering
    
    cost_new = -2*(sum(S.*XCtX))+sum(S.*(CtXtXC*S));
    idx = cost_new <= cost;
    S(:,~idx) = Sold(:,~idx);
    muS(idx) = 1.2*muS(idx);
    muS(~idx) = 0.5*muS(~idx);
    cost(idx) = cost_new(idx);
end

%Calculate sufficient statistic. With or without sigmaSq.
if nargin == 7
    SSt = S*bsxfun(@rdivide,S,sigmaSq')';
else
    SSt = S*S';
end

end