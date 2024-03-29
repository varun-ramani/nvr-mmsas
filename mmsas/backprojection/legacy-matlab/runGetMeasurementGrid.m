load('./inputData_run21.mat')

sarData = rawDataCal;
markerlocs = recentered_marker_locs;
[nRx,nTx,Num_actuator_step,Num_rotor_step,nSample] = size(sarData);

mgrid = getMeasurementGrid(markerlocs, zTarget_radius, Num_rotor_step, Num_actuator_step);

save('measurement_grid.mat', "mgrid");