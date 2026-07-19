clear, close all
%% "Real" Distribution
eta = 200;
beta = 2; % beta == 1 is Exponential, beta==2 is Rayleigh
N = 100;
numSims = 100;
Dreal = makedist('Weibull','a',eta,'b',beta);
% Dreal = makedist('uniform','lower',10,'upper',20);

dists = {'Weibull','Normal','Lognormal','ev','Exponential',...
    'Rayleigh','None'};
freq = zeros(1,length(dists)); % [weibull,exponential,rayleigh,lognormal,ev,none]
for n = 1:numSims
    t = random(Dreal,1,N);
    [D,A] = selectDistributionAD(t);
    if isempty(D)
        freq = freq + strcmpi('None',dists);
    else
        freq = freq + strcmpi(D.DistributionName,dists);
    end
end
bar(freq/numSims)
set(gca,'Xticklabel',dists)
ylabel('Relative Frequency')