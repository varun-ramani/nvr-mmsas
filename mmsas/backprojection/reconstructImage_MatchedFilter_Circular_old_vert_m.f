%% Copyright(C) 2018 Muhammet Emin Yanik
%  Advisor: Prof. Murat Torlak
%  The University of Texas at Dallas
%  Department of Electrical and Computer Engineering

%  Redistributions and use of source must retain the above copyright notice
%  Redistributions in binary form must reproduce the above copyright notice
%%

function [sarImage, dx,x ,y, z] = ...
    reconstructImage_MatchedFilter_Circular_old1opti1(sarData,...
                                                        frequency,...
                                                        Num_actuator_step,...
                                                        Num_rotor_step,...
                                                        txAntPos,...
                                                        rxAntPos,...
                                                        xySizeI,...
                                                        zTarget_center)

% For single tone applications: 
% -------------------------------------------------------------------------
% sarData: should be nRx x nTx x yPointM x xPointM x nSample

% If wideband processing is used:
% -------------------------------------------------------------------------
% frequency: [fStart,fSlope,fSample,adcStart]
% fStart: Start frequency
% fSlope: Slope const (Hz/sec)
% fSample: Sample ps
% adcStart: ADC start time
% Example: [77e9,63.343e12,9121e3,4.66e-6]

% For single tone applications:
% -------------------------------------------------------------------------
% frequency: start frequency

% Variables
% -------------------------------------------------------------------------
% xStepM: measurement step size at x (horizontal) axis in mm
% yStepM: measurement step size at y (vertical) axis in mm
%
% txAntPos,rxAntPos: Tx and Rx antenna locations
% % Optional (To get the antenna locations from the radar type)
% % [rxAntPos,txAntPos,virtualChPos,~] = getAntennaLocations(radarType);
% % [rxAntPos,txAntPos,~] = cropAntennaLocationsSetArrayCenter(rxAntPos,txAntPos,virtualChPos,activeTx,activeRx);
%
% xySizeI: size of image in xy-axis in mm
% zTarget: target distance in mm
% nFFTkXY: number of FFT points, should be greater than xStepM and yStepM


%% Zero Padding to sarData to Locate Target at Center
%-------------------------------------------------------------------------%
sarData = single(conj(sarData)); % Take the conjugate to make consistent with the paper

% sarData = single(conj(sarData));

[nRx,nTx,Num_actuator_step,Num_rotor_step,nSample] = size(sarData);
% if (nFFTkX<xPointM)
%     error('# of FFT points must be greater than the # of measurement points.')
% else
%     sarData = padarray(sarData,[0 0 0 floor((nFFTkX-xPointM)/2) 0],0,'pre');
%     sarData = padarray(sarData,[0 0 0 ceil((nFFTkX-xPointM)/2) 0],0,'post');
% end


%% Define Frequency Spectrum
%-------------------------------------------------------------------------%
if (length(frequency)==4) && (nSample>1)
    frequency = num2cell(frequency);
    [f0,K,fS,adcStart] = frequency{:};
    f0 = f0 + adcStart*K; % This is for ADC sampling offset
    f = f0 + (0:nSample-1)*K/fS; % wideband frequency
elseif (length(frequency)==1) && (nSample==1)
    f = frequency; % single frequency
else
    error('Please correct the configuration and data for 3D processing')
end


%% Define Fixed Parameters
%-------------------------------------------------------------------------%
c = 299792458; % physconst('lightspeed'); in m/s
lambda = c./f;
k = 2*pi./lambda;
fc = 7.9e10;

totsteps = 20;
steplength = 7.6;
steps = 20;
startval = totsteps - steps+1;
count = 1;
finalangle = Num_rotor_step - 1; %359;
measurement_grid = zeros(Num_rotor_step*steps,3);
% center = [0,0,zTarget_center];

for xi = startval:totsteps
    for theta=0:finalangle
        z = 153 - (xi)*steplength;
        x = zTarget_center*sind(theta);
        y = zTarget_center - (zTarget_center*cosd(theta));
        measurement_grid(count,:) = [x y z]; 
        count = count + 1;
    end
end


% Define the range for each dimension
% dx = 15;
% x = -100:dx:150;
% y = -200:dx:0;
% z = 900:dx:1100;

% dx = 5; %- trophy
% x =  -60:dx:60; %-200:dx:140;
% y = 375:dx:925;
% z = -110:dx:0;

% dx = 5; %- trophy
% x =  -50:dx:50; %-200:dx:140;
% y = 510:dx:1200;
% z = -160:dx:300;

dx = 5; %- CFrot2
x =  -160:dx:160; %-200:dx:140;
y = 270:dx:550; %:dx:1200;
z = -110:dx:300;

% Create a grid of points
[X, Y, Z] = meshgrid(x, y, z);

% Reshape and concatenate the coordinates into a single matrix
voxelCoordinates = [X(:), Y(:), Z(:)];

sarImage = zeros(length(voxelCoordinates),1);

matchedFilter_k = single(zeros(nRx,nTx,length(voxelCoordinates)));
% sarImage = single(zeros(length(y),length(x),length(k)));

N = 16*2048; % Number of FFT Point

deld = (fS*c/(2*K*N));
deld2 = N*K/c;

% rawData_RangeFFT = fft(rawData,N,2);

processBar = waitbar(0,'Processing...');
% for nK = 1:length(k)
%     disp(nK);

% fftsarData = fft(sarData,N,5);
% distall = [];
num_workers = 4;
poolobj = check_my_parpool(num_workers);


parfor mG = 1:length(measurement_grid)
    lin = floor((mG-1)/Num_rotor_step) + startval; %1;
    rot = (mod((mG-1),Num_rotor_step) + 1);
    for nT = 1:nTx
        for nR = 1:nRx
            distTx = voxelCoordinates.* 1e-3 - (measurement_grid(mG,:)* 1e-3 + txAntPos(nT,:));
            distRx = voxelCoordinates.* 1e-3 - (measurement_grid(mG,:)* 1e-3 + rxAntPos(nR,:));
            disttot = sqrt((distRx(:,1)).^2 + (distRx(:,2)).^2 + (distRx(:,3)).^2) + sqrt((distTx(:,1)).^2 + (distTx(:,2)).^2 + (distTx(:,3)).^2);

            matchedFilter = exp((disttot*k).*1i);
%                 disp(matchedFilter);
%                     disp(floor(disttot * deld2));
            val = matchedFilter*squeeze(sarData(nR,nT,lin,rot,:));%fftsarData(nR,nT,1,mG,floor(disttot/deld));
            sarImage = sarImage + val;
        end
    end     
end



% sarImage = sarImage.';

% figure;plot(distall);

[xval,yval,zval] = size(X);

sarImage3D = reshape(sarImage,xval,yval,zval);
% 
sarImage3DAbs = abs(sarImage3D);

volumeViewer(sarImage3DAbs);
    
    
% end
delete(processBar)
end

% %% Coherently combine sarImage accross k
% sarImage = sum(sarImage,3);

function [poolobj] = check_my_parpool(num_workers)
    poolobj = gcp('nocreate'); % get the pool object.
    if ~isempty(poolobj) % check if there IS actually a pool:
        if poolobj.NumWorkers ~= num_workers % it has a different number of workers???:
            delete(poolobj); % delete the current pool object.
        end
    end
    if isempty( gcp('nocreate')) % finally, if there is not a pool:
        poolobj = parpool(num_workers); % create a new pool.
    end
end

