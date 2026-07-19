function [D,acceptable] = selectDistributionAD(t,varargin)
t = reshape(t,length(t),1);

% possible distributions
dists = {'Weibull','Normal','Lognormal','ev'};

%% Compute p-values
warning('off','all');
for i = 1:length(dists)
   [~,pval(i)] = adtest(t,'Distribution',dists{i}); 
end

if all(pval<0.05) % no distribution is suitable!
    D =[];
    acceptable = {'none'};
    return
end

%% Fit the best distribution
bdist = dists{ pval==max(pval) };
D = fitdist(t,bdist);

%% Check for Exponential/Rayleigh if Weibull
if strcmpi(bdist,'Weibull')
    [HE,pv] = adtest(t,'Distribution','Exponential');
    DR = fitdist(reshape(t,length(t),1),'Rayleigh');
    [HR,pv] = adtest(t,'Distribution',DR);
    UL = D.B + 1.96*sqrt(D.ParameterCovariance(2,2));
    LL = D.B - 1.96*sqrt(D.ParameterCovariance(2,2));
    if HE == 0 && (UL>1 && LL<1)  % exponential
        bdist = 'Exponential';
        D = fitdist(t,bdist);
    elseif HR==0 && (UL>2 && LL<2)  % rayleigh
        bdist = 'Rayleigh';
        D = fitdist(t,bdist);
    end
end
warning('on','all');

acceptable = dists(pval>0.05);
if ~ismember(bdist,acceptable)
    acceptable{end+1} = bdist;
end