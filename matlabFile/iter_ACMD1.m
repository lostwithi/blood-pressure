function compset = iter_ACMD1(Sig,SampFreq,alpha0,tol,re)
% Iteratively execute the ACMD to extract all the signal modes 
% re denotes the terminal threshold, namely, when the ratio of the residual energy to the total energy is less than re, stop the algorithm 
% When using this code, please cite our papers:
% -----------------------------------------------
% Chen S, Wang K, Chang C, et al. A two-level adaptive chirp mode decomposition method for the railway wheel flat detection under variable-speed conditions, Journal of Sound and Vibration, 2021.
% Chen S, Yang Y, Peng Z, et al, Detection of Rub-Impact Fault for Rotor-Stator Systems: A Novel Method Based on Adaptive Chirp Mode Decomposition, Journal of Sound and Vibration, 2018.
% Chen S, Yang Y, Peng Z, et al, Adaptive chirp mode pursuit: Algorithm and applications, Mechanical Systems and Signal Processing, 2018.
% Chen S, Dong X, Peng Z, et al, Nonlinear Chirp Mode Decomposition: A Variational Method, IEEE Transactions on Signal Processing, 2017.
% Chen S, Peng Z, Yang Y, et al, Intrinsic chirp component decomposition by using Fourier Series representation, Signal Processing, 2017.
% Chen S, Dong X, Xing G, et al, Separation of Overlapped Non-Stationary Signals by Ridge Path Regrouping and Intrinsic Chirp Component Decomposition, IEEE Sensors Journal, 2017.
Sig1 = Sig;
N = length(Sig);
maxnum = 15;  % maximum iteration
compset = zeros(maxnum,N);
SampFreq = double(SampFreq);

for ii = 1:maxnum
% the algorithm gets the initial frequency for the ACMD by finding the peak frequency based on FFT
Spec = 2*abs(fft(Sig))/length(Sig);
Spec = Spec(1:round(end/2));
Freqbin = linspace(0,SampFreq/2,length(Spec));
subindex = Freqbin>0 & Freqbin<0.475*SampFreq; 
Freqbin = Freqbin(subindex);
Spec = Spec(subindex);

[~,findex1] = max(Spec);
peakfre1 = Freqbin(findex1); % IF initialization by finding peak frequency of the Fourier spectrum
iniIF1 = peakfre1*ones(1,length(Sig)); % initial IF vector

[~, ~, Sigtemp] = ACMD1(Sig,SampFreq,iniIF1,alpha0,tol);

compset(ii,:) = Sigtemp;

Sig = Sig - Sigtemp; % update the residual signal 


if (norm(Sig)/norm(Sig1))^2 < re
    break
end
end
compset = [compset(1:ii,:);Sig]; % output all the signal modes and the residual signal








