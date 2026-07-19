function [d,delta] = selectDistributionAIC(t,varargin)
t = reshape(t,length(t),1);

if nargin > 1
    censorFlags = varargin{1};
else
    censorFlags = zeros(length(t),1);
end

% possible distributions
dists = {'Weibull','Normal','Lognormal','ev','exponential','Rayleigh'};

%% Fit all distributions, compute AICc
warning('off','all');
aicc = zeros(length(dists),1);
fittedDists = cell(1,length(dists));
n = length(t);
for i = 1:length(dists)
   D = fitdist(t,dists{i},'Censoring',censorFlags);
   K = D.NumParameters;
   
   % Compute small-sample AICc (asymptotically equivalent to AIC)
   aicc(i) = 2*D.negloglik + 2*K + 2*K*(K+1)/(n-K-1);
   fittedDists{i} = D;
end

[aicc,mask] = sort(aicc);
delta = aicc-min(aicc);
d = fittedDists(mask);
