function joint_positions_3d = extractJointPositions3d(metadata_Depth)
    % Extract 25 3d joint positions from metadata 
    joint_positions_3d = zeros(25, 3, length(metadata_Depth));
    for i = 1:length(metadata_Depth)
        joint_positions_3d(:,:,i) = metadata_Depth(i).JointPositions(:,:,metadata_Depth(i).IsBodyTracked==1);
    end
    
end