function [res] = getGaussianParams(X)  
    %CREATEFIT(TIME,X)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : time
%      Y Output: X
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 25-Oct-2022 19:02:13 自动生成


%% Fit: 'untitled fit 1'.
 time = 1:1:length(X);
[xData, yData] = prepareCurveData( time, X );

% Set up fittype and options.
ft = fittype( 'gauss2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0 -Inf -Inf 0];
opts.Robust = 'Bisquare';
opts.StartPoint = [0.673308239378563 55 7.95189272026727 0.490986110325115 41 12.535997769999];

% Fit model to data.
[fitresult, ~] = fit( xData, yData, ft, opts );
    
res = [];
res(end + 1) = fitresult.a1;
res(end + 1) = fitresult.b1;
res(end + 1) = fitresult.c1;

res(end + 1) = fitresult.a2;
res(end + 1) = fitresult.b2;
res(end + 1) = fitresult.c2;

    
    
end