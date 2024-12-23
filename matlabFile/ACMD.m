function [IFest IAest sest] = ACMD(s,fs,eIF,alpha0,beta,tol)
% Adaptive Chirp Mode Decomposition (ACMD)
% Authors: Shiqian Chen and Zhike Peng
% mailto:chenshiqian@sjtu.edu.cn; z.peng@sjtu.edu.cn;
% https://www.researchgate.net/profile/Shiqian_Chen2   https://www.researchgate.net/profile/Z_Peng2
%
%%%%%%%%%%%%%%%%%%%%%%%  input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% s: measured signal,a row vector
% fs: sampling frequency
% eIF: initial instantaneous frequency (IF) time series of a certain signal mode,a row vector
% alpha0: penalty parameter controling the filtering bandwidth of ACMD;the smaller the alpha0 is, the narrower the bandwidth would be
% beta: penalty parameter controling the smooth degree of the IF increment during iterations;the smaller the beta is, the more smooth the IF increment would be
% tol: tolerance of convergence criterion; typically 1e-7, 1e-8, 1e-9...
%%%%%%%%%%%%%%%%%%%%%%% output %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IFest: the finally estimated IF
% sest: the finally estimated signal mode
% IAest: the finally estimated instantaneous amplitude (IA) of the signal mode
% When using this code, please cite our papers:
% -----------------------------------------------
% Chen S, Yang Y, Peng Z, et al, Detection of Rub-Impact Fault for Rotor-Stator Systems: A Novel Method Based on Adaptive Chirp Mode Decomposition, Journal of Sound and Vibration, 2018.
% Chen S, Yang Y, Peng Z, et al, Adaptive chirp mode pursuit: Algorithm and applications, Mechanical Systems and Signal Processing, 2018.
% Chen S, Dong X, Peng Z, et al, Nonlinear Chirp Mode Decomposition: A Variational Method, IEEE Transactions on Signal Processing, 2017.
% Chen S, Peng Z, Yang Y, et al, Intrinsic chirp component decomposition by using Fourier Series representation, Signal Processing, 2017.
% Chen S, Dong X, Xing G, et al, Separation of Overlapped Non-Stationary Signals by Ridge Path Regrouping and Intrinsic Chirp Component Decomposition, IEEE Sensors Journal, 2017.
%% initialize
N = length(eIF);%N is the number of the samples
fs = double(fs);
t = (0:N-1)/fs;%time
%t = 0:1/fs:N-1
e = ones(N,1);
e2 = -2*e;
oper = spdiags([e e2 e], 0:2, N-2, N);%the second-order difference matrix
spzeros = spdiags([zeros(N,1)], 0, N-2, N);
opedoub = oper'*oper;%
phim = [oper spzeros;spzeros oper];
phidoubm = phim'*phim;
iternum = 300; %the maximum allowable iterations
IFsetiter = zeros(iternum,N);  %the collection of the obtained IF time series of the signal modes at each iteration
ssetiter = zeros(iternum,N); %the collection of the obtained signal modes at each iteration
ysetiter = zeros(iternum,2*N);
%% iterations 
iter = 1;% iteration counter
sDif = tol + 1;%
alpha = alpha0;
while ( sDif > tol &&  iter <= iternum ) % 
       
    cosm = cos(2*pi*(cumtrapz(t,eIF)));
    sinm = sin(2*pi*(cumtrapz(t,eIF)));
    Cm = spdiags(cosm(:), 0, N, N);
    Sm = spdiags(sinm(:), 0, N, N);
    Kerm = [Cm Sm]; %kernel matrix
    Kerdoubm = Kerm'*Kerm;
%%%%%%%%%%%%%%%%update demodulated signals%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    ym = (1/alpha*phidoubm + Kerdoubm)\(Kerm'*s(:));  %the demodulated target signal
    si = Kerm*ym; %the signal component
    ssetiter(iter,:) = si;
    ysetiter(iter,:) = ym;
 %%%%%%%%%%%%%  update the IFs  %%%%%%%%%%%%%%%%%%%%%%%%           
    ycm = (ym(1:N))'; ysm = (ym(N+1:end))'; % the two demodulated quadrature signals
    ycmbar = Differ(ycm,1/fs); ysmbar = Differ(ysm,1/fs);%compute the derivative of the functions
    deltaIF = (ycm.*ysmbar - ysm.*ycmbar)./(ycm.^2 + ysm.^2)/2/pi;% obtain the frequency increment by arctangent demodulation
    deltaIF = (1/beta*opedoub + speye(N))\deltaIF';% smooth the frequency increment by low pass filtering
    eIF = eIF - deltaIF';% update the IF
    IFsetiter(iter,:) = eIF;    
%%%%%%%%%%%%%  compute the convergence index %%%%%%%%%%%%%%%%%%  
  if iter>1
   sDif = (norm(ssetiter(iter,:) - ssetiter(iter-1,:))/norm(ssetiter(iter-1,:))).^2;
  end
    iter = iter + 1;
end
    iter = iter -1; % maximum iteration
    IFest = IFsetiter(iter,:); %estimated IF
    sest =  ssetiter(iter,:);  %estimated signal mode
    ycm = ysetiter(iter,1:N);ysm = ysetiter(iter,N+1:end);
    IAest = (sqrt(ycm.^2 + ysm.^2));  % estimated IA
end