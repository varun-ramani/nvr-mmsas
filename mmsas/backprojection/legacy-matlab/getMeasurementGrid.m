function measurement_grid = getMeasurementGrid(markerlocs, radius, Num_rotor_step, Num_actuator_step)
    sensorMarkerheight = markerlocs(:,3,4);
    sensorMarkerheight = reshape(sensorMarkerheight, Num_rotor_step, []);
    heights = mean(sensorMarkerheight, 1)';
    heights = heights - ones(30,1)*15;

    % table1 = markerlocs(:,:,2);
    % angles1 = getAnglesMatrix(table1, Num_rotor_step, Num_actuator_step);

    % table2 = markerlocs(:,:,1);
    % angles2 = getAnglesMatrix(table2, Num_rotor_step, Num_actuator_step);

    % finalangles = (angles1 + angles2)/2; % we're intentionally not using this

    finalangles = transpose(linspace(0,2*pi - pi/(720),1440));
    finalangles = repmat(finalangles, 1, 30);

    measurement_grid = zeros(length(finalangles)*length(heights),3);
    count = 1;

    for xi = 1:length(heights)
        for theta=1:length(finalangles)
            x = radius*cos(finalangles(theta,xi));
            y = radius*sin(finalangles(theta,xi));
            z = heights(xi);
            measurement_grid(count,:) = [x y z];
            count = count + 1;
        end
    end
end