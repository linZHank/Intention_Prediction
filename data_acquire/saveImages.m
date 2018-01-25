function saveImages(destiny_directory, images, ts)
% Save images from tensor to .png files
    % ts: time stamp
    if ~exist(destiny_directory, 'dir')
        mkdir(destiny_directory);
    end
    
    for i = 1:size(images,4)
        filename = ['\trial', '_', ts, '_', 'frame', '_', num2str(i,'%04i'), '.png'];
        imwrite(images(:,:,:,i), [destiny_directory, filename]);
    end
