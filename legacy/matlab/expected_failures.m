function [t,H] = expected_failures(dist,DT,T,varargin)
% [T,H] = EXPECTED_FAILURES(PDF,DT,T,PARAM1,PARAM2,...) 
% Computes the expected number of failures for the time
% interval [0,T] using the assumption of one failure per DT interval.
% The string PDF is the name of a distribution where PARAM1,... are the
% parameters of the distribution.  The syntax for the common distributions
% is as follows
%
% Normal PDF:  
%       [t,H] = expected_failures('normal',DT,T,mu,sigma)
% Weibull PDF:  
%       [t,H] = expected_failures('weibull',DT,T,eta,beta)
% Exponential PDF:  
%       [t,H] = expected_failures('exponential',DT,T,lambda)
% Uniform PDF:  
%       [t,H] = expected_failures('uniform',DT,T,upperlimit,lowerlimit)
%
% where the meaning of the parameters should be obvious from class, but if
% you forgot, consult the MATLAB documentation regarding each PDF.
%
% Written by Michael E. Cholette
% April 19, 2022

k = floor(T/DT);
t = 0:DT:k*DT;
if strcmpi(dist(1),'w') % weibull
   f = @wblcdf;
elseif strcmpi(dist(1),'n') % normal
   f = @normcdf;
elseif strcmpi(dist(1),'e') % exponential
   varargin{1} = 1/varargin{1};
   f = @expcdf;
elseif strcmpi(dist(1),'u')
   f = @unifcdf;
end

% CDF and probabilities of failures in itervals
F = f(t,varargin{:});
p = diff(F); % F(i)-F(i-1)

% compute H recursively
H = zeros(1,k+1);
for r = 1:k
   H(r+1) = (1 + H(1:r))*fliplr(p(1:r))'; 
end