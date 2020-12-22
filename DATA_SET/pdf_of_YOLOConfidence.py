'''
Author: rghods@cs.cmu.edu, durkin.98@osu.edu

plot histograms for probability density function of object detection as a
function of confidence measure of YOLOv3
'''

import os
from darknet_tools import Darknet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import time

image_dir = 'stats_images'
p = 2
image_list = []
for index in range(0,256):
    for d in range(4): # four angles
        image_name = (str(index)+'_' +str(d) + "_" + str(p) + '.png')
        image_list.append(os.path.join(image_dir,image_name))
dn = Darknet(image_list)
# dn.run_darknet()
#
# # For example, print the results of the second image
# print(dn.image_results[1])
#
# # now save to a pickle
# dn.dump_to_pickle(os.path.join('Stats_Results',"dn_results.pkl"))
#
# print('saved results!')

res_dicts = dn.load_from_pickle(os.path.join('Stats_Results',"dn_results.pkl"))

TrueDet = [] # the instances where the detector correctly detected a person
FalseDet = [] # the instances where a detector mistakenly detected something as a person
FalseUndet = []
for i in range(45,len(image_list)):
    dict = res_dicts[i]
    items = dict["item"]
    bounds = dict["bounds"]
    acc = dict["accuracy"]
    img_PIL = Image.open(dict["img"])
    img_name = os.path.basename(dict["img"])
    depth_name = "depth_"+img_name.split("_")[0]+"_"+img_name.split("_")[1]+"_0.npy"
    depth = np.load(os.path.join(image_dir,depth_name))
    width,height = img_PIL.size
    for j, item in enumerate(items):
        if(items[j]=="person"):
            (x1,y1,x2,y2) = bounds[j]
            print(bounds[j])
            depth_box = depth[int(y1):int(y2),int(x1):int(x2)]
            d = np.median(depth_box)
            fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2, 2)
            im1 = ax1.imshow(depth_box,vmin=0,vmax=90)
            fig.colorbar(im1,ax=[ax1])
            crop_bounds = (np.maximum(0,int(x1)-50),np.maximum(int(y1)-50,0),\
                np.minimum(width,int(x2)+50),np.minimum(int(y2)+50,height))
            ax0.imshow(img_PIL.crop(crop_bounds))
            ax2.imshow(img_PIL.crop((int(x1),int(y1),int(x2),int(y2))))
            fig.show()
            print("Is the object a "+items[j]+"?\n")
            Det = input("Please Type Y/N")
            if(Det == "y"):
                TrueDet.append((d,acc[j]))
                print('Truly detected: d=%3.0f,accuracy=%2.2f',d,acc[j])
            else:
                FalseDet.append((d,acc[j]))
                print('Falsely detected: d=%3f,accuracy=%2.2f',d,acc[j])
            plt.close()

    try:
        pred_PIL = Image.open(dict["img_predict"])
        plt.imshow(pred_PIL)
        plt.show()
    except:
        print('prediction not found')
    text = input("Are there any people here that are not detected by YOLOv3?")
    if(text == "y"):
        num_undetected = int(input("how many are there?"))
        print("please click them on the image:/n \
        add points: left click or space/n \
        remove the last point: right click or backspace/n \
        terminate input: middle mouse or enter")
        plt.imshow(pred_PIL)
        points = plt.ginput(n=num_undetected,timeout=30)
        for k in range(num_undetected):
            (x,y) = points[k]
            d = np.median(depth[np.maximum(int(y)-10,0):np.minimum(int(y)+10,height)\
            ,np.maximum(int(x)-10,0):np.minimum(int(x)+10,height)])
            FalseUndet.append((d,0.0))
            print('Undetected: d=%3f,accuracy=%2.2f',d,0.0)
        plt.close()
        print('done with',img_name)
    pickle.dump([TrueDet,FalseDet,FalseUndet,image_list],open(os.path.join('Results',"stats.pkl"), 'wb'))

[TrueDet,FalseDet,FalseUndet,image_list] = pickle.load(open(os.path.join('Results',"stats.pkl"), 'rb'))

print(TrueDet)
print(FalseDet)
print(FalseUndet)




















# kjii
