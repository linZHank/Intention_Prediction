# Notes

## Plot joint position distance change
Run `plotjposchange.py` to pick and specify intent, pitcher and trial. Follow the prompt instructions.\
### How to locate pitching initiation
Define the pitcher's pose at frame *t1* by joint positions p_t1=(x_1_t, y_1_t, z_1_t, ..., x_25_t, y_25_t, z_25_t), then distance between p_t1 and p_t2 can be calculated by applying L2-norm to p_t1 - p_t2. 
1. Set p_t0 as the reference pose. 
  > Since in every pitch, the pitcher starts the trial from static rest state, we set the joint positions in very first frame,
2. Calculate distance between reference pose and pose at every frame to obtain a curve that describes the pitcher's pose movement againt his initial pose.
3. Create a window with size of 20 frames.
4. Sweep the pose distance chaning curve with the window starting at first frame. In current window: \
- &nbsp;&nbsp;If all distance change was greater than 0.16 (heuristic) and the curve was monotomically increasing, the very first frame of current windown was denoted as the initiation of this pitch trial.
- &nbsp;&nbsp;If neither the distance change was  greater than 0.16 nor the curve was monotomically increasing, keep sweeping the window until these two conditions both satisfied.

In practice, we are able to locate the initiating of the pitch in each trial using this method.

