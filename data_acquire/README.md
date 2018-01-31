# Experiment Protocol
## Getting Involved
- Control Tower: Human conductor who coordinating the experiment, issue command to *Pitcher*, operating computer, etc..
- Pitcher: Human participant who throws the *Ball* to the *Cubes*. 
- Cubes: Targets of *Ball*.
- Ball: Thrown by human *Pitcher* toward the *Cubes*.
- Kinect V2: Record data.

## Procedures
- ***Humans warm up:*** *Pitcher* practices throwing ball to the cube he\/she intend to. Make sure the pitcher succeeded at least one time for each cube. *Pitcher* practices for his\/her initiating and ending pose. *Control Tower* practices communication with the *Pitcher*.
- ***(Optional) Computer warm up:*** run `data_capture.m` because first trial usually takes longer to initiate. 
- **Data collection:** run `record_pitch_data.m` to starting collect data. 3D joint position, color images and depth images 
will automatically stored.

## Descriptions
Computer picks the target randomly. *Control Tower* inform the target to the *Pitcher*. *Pitcher* throws the *Ball* toward the target *Cube*. Save the data.
- A set of experiment comprises 9 succeeded trials. *Pitcher* has to throw the *Ball* into every cube once.
- Each *Pitcher* was expected to complete 10 sets of experiment \(90 succeeded trials\). \(The experiment can be seperated 
into serveral subsets if participants fatigued or lacked enough time to finish all sets in one time slot.\)
- Looking forward to collect 10 participants' data. Varied gender and age. So, ask the girl you are interested in to 
come by and play this game:bowtie:

## Data Collection
- **Important: Make sure the computer time is correct!**
- *Control Tower* confirm with the *Pitcher* to get ready of recording experiment trials.
    1. *Control Tower* inform the *Pitcher* which cube was his\/her target
    2. *Pitcher* listen to the auditory cue, and initiate throwing \"duang\" was heard. **Important! *Pitcher* has to stand 
    still before he\/she initiating.** The initiating pose was not mandatory, hence can be determined at *Pitcher's* preference.
    3. *Pitcher* throwing the ball toward the intended cube within 3 seconds. **Important! *Pitcher* is not allowed to perform 
    any irrelevent action after he\/she finished throwing.** We recommend *Pitcher* to return to the state of standing still 
    with his\/her prefered pose.
    4. Unless the ball was blocked out by the border frame of the target cube, or *Control Tower* and *Pitcher* 
    not satisfied with current trial, recorded data will be saved. 
        - if the ball was thrown into any cube include the target, label the trial with actual target number.
        - if the ball was thrown out of the range of cube storage, label the trial with '100'.
        - if the ball was thrown onto the border frame which was not belong to the target cube, label the trial with the 
        farthest cube number that has the border frame \(Manhatan distance\). 
- Repeat step i through iv, untill the *Pitcher* succeeded every cube intentionally.
- *(Optional) Water break.*
