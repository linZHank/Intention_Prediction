"""Plot all models test accuracies against number of frames
"""

import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

# Read in accuracies
csv_dir = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/result20180607"
csvfiles = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
scores = np.zeros([len(csvfiles), 8])
num_frames = 5*np.arange(1,9)

for i, cf in enumerate(csvfiles):
  scores[i] = pd.read_csv(cf)["accuracy_disc"].values

# Plot bars
x_ticks = 10*np.arange(scores.shape[1])  # the x locations for the groups
y_ticks = 0.1*np.arange(11)
#y_ticks = y_ticks.astype("|S4")
width = 1.2  # the width of the bars
rects = []
clr = ["#ff9b1a"]*4 + ["white"]*3
pattern = [""] + ["/", ".", "O"]*2

fig, ax = plt.subplots()
rects1 = ax.bar(
  x_ticks + range(-3, 4)[0]*width,
  scores[0],
  width,
  align="center",
  color=clr[0],
  edgecolor="black",
  hatch=pattern[0],       
  label='color_cnn'
)
rects1 = ax.bar(
  x_ticks + range(-3, 4)[1]*width,
  scores[1],
  width,
  align="center",
  color=clr[1],
  edgecolor="black",
  hatch=pattern[1],       
  label='color_knn'
)
rects1 = ax.bar(
  x_ticks + range(-3, 4)[2]*width,
  scores[2],
  width, 
  align="center",
  color=clr[2],
  edgecolor="black",
  hatch=pattern[2],       
  label='color_mlp'
)
rects1 = ax.bar(
  x_ticks + range(-3, 4)[3]*width,
  scores[3],
  width, 
  align="center",
  color=clr[3],
  edgecolor="black",
  hatch=pattern[3],       
  label='color_svm'
)
rects1 = ax.bar(
  x_ticks + range(-3, 4)[4]*width,
  scores[4],
  width, 
  align="center",
  color=clr[4],
  edgecolor="black",
  hatch=pattern[4],       
  label='joint_knn'
)
rects1 = ax.bar(
  x_ticks + range(-3, 4)[5]*width,
  scores[5],
  width, 
  align="center",
  color=clr[5],
  edgecolor="black",
  hatch=pattern[5],       
  label='joint_mlp'
)
rects7 = ax.bar(
  x_ticks + range(-3, 4)[6]*width,
  scores[6],
  width, 
  align="center",
  color=clr[6],
  edgecolor="black",
  hatch=pattern[6],       
  label='joint_svm'
)


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(('5', '10', '15', '20', '25', '30', '35', '40'))
# ax.set_yticks(y_ticks)
# ax.legend()
ax.legend(
  loc=9, # upper center
  bbox_to_anchor=(0.5, 1.1),
  ncol=7,
  fancybox=True,
  shadow=True
) 
plt.xlabel('Number of frames', fontsize=20)
plt.ylabel('Test Accuracy', fontsize=20)
plt.xlim(-5, 75)
plt.ylim(0, 1.01)
plt.xticks(x_ticks, num_frames.astype("|S4"), fontsize=16)
plt.yticks(y_ticks, y_ticks.astype("|S3"), fontsize=16)
plt.grid(axis="y")
plt.show()
