clear, close all
%% "Real" Distribution
eta = 200;
beta = 2; % beta == 1 is Exponential, beta==2 is Rayleigh
N = 1000;
numSims = 100;
freq = zeros(1,6); % [weibull,exponential,rayleigh,lognormal,ev,none]
Dreal = makedist('Weibull','a',eta,'b',beta);
% Dreal = makedist('uniform','lower',10,'upper',20);

dists = {'Weibull','Normal','Lognormal','ev','Exponential',...
    'Rayleigh'};
for n = 1:numSims
    t = random(Dreal,1,N);
    [D,A] = selectDistributionAIC(t);
    D = D(A<2); % AIC differences of 2 are considered statistically equivalent!
    for j = 1:length(D)
        if ~isempty(D)
            freq = freq + ...
                strcmpi(D{j}.DistributionName,dists);
        end
    end
end
bar(freq/numSims)
set(gca,'Xticklabel',dists)
ylabel('Relative Frequency')