imaqreset %deletes any image acquisition objects existing in memory


%% Setup the Kinect V2 for color and depth acquisition
% Create color and depth kinect videoinput objects.
colorVid = videoinput('kinect', 1);
depthVid = videoinput('kinect', 2);
% pause;

% Look at the device-specific properties on the depth source device,
% which is the depth sensor on the Kinect V2.
% Set 'EnableBodyTracking' to on, so that the depth sensor will
% return body tracking metadata along with the depth frame.
depthSource = getselectedsource(depthVid);
depthSource.EnableBodyTracking = 'on';

% Acquire 100 color and depth frames.
framesPerTrig = 90;
colorVid.FramesPerTrigger = framesPerTrig;
depthVid.FramesPerTrigger = framesPerTrig;

% Start the depth and color acquisition objects.
% This begins acquisition, but does not start logging of acquired data.
preview(colorVid)

% Countdown
load gong.mat
disp('5');beep;
pause(1)
disp('4');beep;
pause(1)
disp('3');beep;
pause(1)
disp('2');beep;
pause(1)
disp('1');beep;
pause(1)
sound(y);disp('Go');

start([depthVid colorVid]);

%% Access Image and Skeletal Data
% Get images and metadata from the color and depth device objects.
[imgColor, ts_color, metadata_Color] = getdata(colorVid);
[imgDepth, ts_depth, metadata_Depth] = getdata(depthVid);

stop([depthVid colorVid]);
closepreview(colorVid);
beep;