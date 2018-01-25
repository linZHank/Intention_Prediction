%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           record_pitch_data.m                           %
%                                                                         %  
% This code enables a Kinect V2 camera record trials of ball pitching     %
% activities. Meanwhile, output skeleton model, color and depth image     %
% data for further applications.                                          % 
%                                                                         %
%                                                   Created by: LinZ,     %
%                                                   05/14/2017            %
%                                                                         %
%                                               Mod on: 01/24/2018        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize
clc; clear;
LOCALDIR = 'D:\Data\pit2d9blk';
% Get experiment information
ymdhms = clock;
yr = num2str(ymdhms(1), '%04i');
mon = num2str(ymdhms(2), '%02i');
date = num2str(ymdhms(3), '%02i');
hr = num2str(ymdhms(4), '%02i');
min = num2str(ymdhms(5), '%02i');
TIME_STAMP = [yr, mon, date, hr, min];
SCALE = 1/3;
% Initial settings
NUM_TARGET = 9;
ID = input('Declare yourself: ', 's'); % Attribute to Baldur's Gate II
target_pool = linspace(1,NUM_TARGET,NUM_TARGET);
counter = 0;

%% Recording
while ~isempty(target_pool)
    actual_target = 0;
    [intent_target, new_target_pool] = arrangeTarget(target_pool);
    fprintf('Your boss picked No. %d as your target.\n', intent_target);
    while intent_target ~= actual_target
        disp('COPY TARGET TO THE PITCHER!!!')
%         pause;
        disp('GET READY !!!')
        data_capture;
        actual_target = input('What was the actual target? ');
        % message about last pitch
        while ~ismember(actual_target, linspace(1,NUM_TARGET,NUM_TARGET))
            actual_target = input('What was the actual target? ');
        end
        if actual_target == intent_target
            disp('Strike! You hit the target!')
        else
            fprintf('Oops, you missed the target... Please aim at target %d.\n', intent_target)
        end
        counter = counter+1;
        
        %% Reshape data
        % extract 3d joint positions
        joint_positions_3d = extractJointPositions3d(metadata_Depth);
        % resize color images
        color_img = imresize(imgColor, SCALE);
        % resize depth images
        depth_img = imgDepth; 
        
        %% Save_date;
        % save joint positions
        dest_dir_joint = [LOCALDIR, '\joint\intent', num2str(intent_target,'%02i'),...
            '\', ID, '\trial', '_', TIME_STAMP, '\actual', num2str(actual_target,'%02i')];
        if ~exist(dest_dir_joint, 'dir')
            mkdir(dest_dir_joint);
        end
        save([dest_dir_joint, '\joint_positions_3d.mat'], 'joint_positions_3d')
        % save color images 
        dest_dir_color = [LOCALDIR, '\color\intent', num2str(intent_target,'%02i'),...
            '\', ID, '\trial', '_', TIME_STAMP, '\actual', num2str(actual_target,'%02i')];
        saveImages(dest_dir_color, color_img, TIME_STAMP);
        % save depth images
        dest_dir_depth = [LOCALDIR, '\depth\intent', num2str(intent_target,'%02i'),...
            '\', ID, '\trial', '_', TIME_STAMP, '\actual', num2str(actual_target,'%02i')];
        saveImages(dest_dir_depth, depth_img, TIME_STAMP);        
    end
    target_pool = new_target_pool;
end
fprintf('Congrats! you are done, your have attemped %d times to finish the test\n', counter)




        
    
    
    
    
    
    
    




