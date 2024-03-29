function [relative_angles] = getAnglesMatrix(coords, Num_rotor_step, Num_actuator_step)
    coords = reshape(coords, Num_rotor_step, Num_actuator_step, 3);
    %     coords = squeeze(median(coords,2));

    angles = atan2(coords(:,:,2), coords(:,:,1));
    initial_angle = angles(1,:);
    relative_angles = angles - initial_angle;
    relative_angles = mod(relative_angles, 2*pi);
end