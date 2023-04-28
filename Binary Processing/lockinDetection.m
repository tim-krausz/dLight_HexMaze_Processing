function [sig1,sig2]=lockinDetection(input,exc1,exc2,Fs,varargin)

filterorder = 5;
tau=10;
deTrend=false;
full = false;

for i = 1:2:length(varargin)
    switch lower(varargin{i})
        case 'tau'   % Filter DEcay constant(ms)
            tau = varargin{i+1};
        case 'filterorder'
            filterorder = varargin{i+1};
        case 'detrend'
            deTrend = varargin{i+1};
        case 'full'
            full = varargin{i+1};
        otherwise
            disp('invalid optional argument passed to lockinDetection');
    end
end


tau = tau/1000;
Fc = 1/(2*pi*tau);
fL = .01;
[b,a] = butter(filterorder,Fc/(Fs/2),'high');

input = filter(b,a,input);


demod1 = input.*exc1;
demod2 = input.*exc2;

if deTrend
    [b,a] = butter(filterorder,[fL Fc]/(Fs/2));
else
    [b,a] = butter(filterorder,Fc/(Fs/2));
end

% [z,p,k] = butter(4,Fc/(Fs/2));
% [sos,g] = zp2sos(z,p,k);


if ~full
    
    sig1 = filter(b,a,demod1);
    sig2 = filter(b,a,demod2);
    
else
    
    sig1x = filter(b,a,demod1);
    sig2x = filter(b,a,demod2);
    
    
    exc1  =hilbert(exc1);exc1=imag(exc1);
    exc2  =hilbert(exc2);exc2=imag(exc2);
    
    demod1 = input.*exc1;
    demod2 = input.*exc2;
    
    if deTrend
        [b,a] = butter(filterorder,[fL Fc]/(Fs/2));
    else
        [b,a] = butter(filterorder,Fc/(Fs/2));
    end
    
    
    sig1y = filter(b,a,demod1);
    sig2y = filter(b,a,demod2);
    
    sig1 = (sig1x.^2+sig1y.^2).^.5;
    sig2 = (sig2x.^2+sig2y.^2).^.5;
end








