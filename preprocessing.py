import cv2
import os
import matplotlib.pyplot as plt

# path to where the dataset got downloaded
dataset_path = r"C:\Users\sijip\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"

# just checking what folders are in there
folders = os.listdir(dataset_path)
print(folders)

# grabbing one image to test my pipeline on first
folder1 = os.path.join(dataset_path, folders[0])
img_name = os.listdir(folder1)[0]
img_path = os.path.join(folder1, img_name)
print("using this image:", img_path)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 loads as BGR by default, fixing that

# converting to HSV - this makes the diseased/discolored spots easier to isolate
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# trying to separate the leaf from the background using green color range
lower = (25, 40, 40)
upper = (90, 255, 255)
mask = cv2.inRange(hsv, lower, upper)
leaf_only = cv2.bitwise_and(img, img, mask=mask)

# blurring a bit to get rid of noise/graininess
blurred = cv2.GaussianBlur(leaf_only, (5, 5), 0)

# boosting contrast so the CNN has an easier time picking up features
ycc = cv2.cvtColor(blurred, cv2.COLOR_RGB2YCrCb)
ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
result = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

# putting it all in one image so I can see each step
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(img); ax[0].set_title("original")
ax[1].imshow(hsv); ax[1].set_title("hsv")
ax[2].imshow(leaf_only); ax[2].set_title("leaf segmented")
ax[3].imshow(result); ax[3].set_title("final result")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("pipeline_demo.png")
print("done, check pipeline_demo.png")