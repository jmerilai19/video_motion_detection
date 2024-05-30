import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('test.mp4')

# first frame
ret, img0_color = cap.read()

# grayscale
img0 = cv2.cvtColor(img0_color, cv2.COLOR_BGR2GRAY)/255
height, width = img0.shape

# store a few frames
plt.figure(figsize=(15,10))
images = np.zeros((height,width,9), dtype=np.float32)

for i in range(9):
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20*i)
    ret, img_color = cap.read()
    if ret == False:
        break
        
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)/255
    images[:,:,i] = img

sigma_cam = 0.005
mean_img = np.median(images, axis=2)
var_img = sigma_cam * np.ones_like(img0)

state_initial = {"alpha": 0.99,
                 "sigma_cam": sigma_cam,
                 "mean_img": mean_img,
                 "var_img": var_img,
                 "K" : 2.0}

alpha = state_initial["alpha"]

def updateBackgroundModel(img, state):
    old_state = state["mean_img"]
    state["mean_img"] = alpha * state["mean_img"] + (1 - alpha) * img

    state["var_img"] = alpha * (state["var_img"] + (state["mean_img"] - old_state) ** 2) + \
                       (1 - alpha) * (img - state["mean_img"]) ** 2

    return state

def thresholdFrame(img, state):
    thresh = np.abs(img - state["mean_img"]) > state["K"] * np.maximum(np.sqrt(state["var_img"]), state["sigma_cam"])

    return thresh

N = 50
i = 0

state = state_initial.copy()

while i < N:
    ret, img_color = cap.read()
    if ret == False:
        break
        
    # convert to grayscale
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)/255
    
    # update
    state = updateBackgroundModel(img, state)
    
    i += 1
     
thresh = thresholdFrame(img, state)

# Overlay thresholded image on the frame
overlayed = np.stack((img,)*3, axis=-1)
red = img.copy()
red[thresh.astype("bool")] = 1
overlayed[:,:,0] = red

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))

N = 500
i = 0

state = state_initial.copy()

while i < N:

    ret, img_color = cap.read()
    if ret == False:
        break

    # Convert to grayscale
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)/255

    # Threshold
    thresh = thresholdFrame(img, state);

    # Update
    state = updateBackgroundModel(img, state);

    # Write output frame
    red = img_color[:,:,2]
    red[thresh.astype("bool")] = 255
    img_color[:,:,2] = red
    out.write(img_color)

    i += 1
    
cap.release()
out.release()
