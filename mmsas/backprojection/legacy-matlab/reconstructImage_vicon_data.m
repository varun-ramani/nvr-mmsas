function [sarImage, dx,x ,y, z] = ...
    reconstructImage_vicon_data(sarData,frequency,txAntPos,...
    rxAntPos,...
    zTarget_radius, ...
    markerlocs)

[nRx,nTx,Num_actuator_step,Num_rotor_step,nSample] = size(sarData);

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

c = 299792458; % physconst('lightspeed'); in m/s
lambda = c./f;
k = 2*pi./lambda;
startval = 1;

measurement_grid = getMeasurementGrid(markerlocs, zTarget_radius, Num_rotor_step, Num_actuator_step);

disp('saving measurement grid')

filepath = strcat("./measurement_grid.mat");
save(filepath, "measurement_grid");

dx = 5; %- CFrot2
x = -50:dx:50; %-200:dx:140;
y = -50:dx:50; %10:dx:40; %:dx:1200;
z = 90:dx:220; %75:dx:220;
% Create a grid of points
[X, Y, Z] = meshgrid(x, y, z);
% Reshape and concatenate the coordinates into a single matrix
voxelCoordinates = [X(:), Y(:), Z(:)];
sarImage = zeros(length(voxelCoordinates),1);

voxelCoordinates = gpuArray(voxelCoordinates);
measurement_grid = gpuArray(measurement_grid);
txAntPos = gpuArray(txAntPos);
rxAntPos = gpuArray(rxAntPos);
sarData = gpuArray(sarData);
sarImage = gpuArray(zeros(size(sarImage))); % Assuming sarImage is defined

startval = gpuArray(startval);
Num_rotor_step = gpuArray(Num_rotor_step);

% Pre-compute grid indexes and rotations to reduce overhead in loop
lin = floor((gpuArray(1:length(measurement_grid))-1)/Num_rotor_step) + startval;
rot = mod((gpuArray(1:length(measurement_grid))-1),Num_rotor_step) + 1;

num_workers = 4;
poolobj = check_my_parpool(num_workers);

parfor mG = 1:length(measurement_grid)
    for nT = 1:nTx
        for nR = 1:nRx
            distTx = bsxfun(@minus, voxelCoordinates .* 1e-3, (measurement_grid(mG,:) .* 1e-3 + txAntPos(nT,:)));
            distRx = bsxfun(@minus, voxelCoordinates .* 1e-3, (measurement_grid(mG,:) .* 1e-3 + rxAntPos(nR,:)));
            disttot = sqrt(sum(distRx.^2, 2)) + sqrt(sum(distTx.^2, 2));
            matchedFilter = exp((disttot * k) .* -1i);
            val = matchedFilter * squeeze(sarData(nR, nT, lin(mG), rot(mG), :));
            val = sum(val,2);
            sarImage = sarImage + val;
        end
    end
end

sarImage = gather(sarImage);

figure;plot3(voxelCoordinates(:,1),voxelCoordinates(:,2),voxelCoordinates(:,3),'o');
hold on;plot3(measurement_grid(:,1),measurement_grid(:,2),measurement_grid(:,3),'*');
axis equal;

[xval,yval,zval] = size(X);

sarImage3D = reshape(sarImage,xval,yval,zval);
sarImage3DAbs = abs(sarImage3D);

% volumeViewer(sarImage3DAbs);
end

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