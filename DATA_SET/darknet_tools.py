import numpy as np
import os
import subprocess
import pickle
import time

class Darknet:
    def __init__(self, image_list):
        #image list is a list containing the paths to the images.
        self.image_list = image_list

        # I don't think these will change, so they are hard coded in
        self.yolov3_weights = "~/Programs/darknet_bounding_box/yolov3.weights"

        self.image_results = []

        self.threshold = None

    def run_darknet(self, threshold=0.25,dir="Results"):

        # Threshold is the detection probability that will be printed
        # It should be between "0" and 1. Darknet doesn't like to use 0, so
        # just use a very low number. If you use a percentage like 25 instead of
        # 0.25, the code divides it by 100.
        if threshold > 1.:
            threshold /= 100.
        if threshold == 0:
            threshold += 0.01
        self.threshold = threshold
        self.dir = dir


        for img in self.image_list:
            self.image_results.append(self.__run_darknet_on_img__(img))
        #self.img_results = np.array(self.img_results)

    def __run_darknet_on_img__(self, img):
        cmd ="darknet detect cfg/yolov3.cfg "+self.yolov3_weights + " "+ img + " -thresh "+ str(self.threshold)

        #s = subprocess.call([cmd], shell=True)
        s = subprocess.check_output([cmd], shell=True)
        s = s.decode() # change it from binary to string
        print(s)
        img_results = s.split("\n")

        items = []
        bounds = []
        accuracy = []
        count = 1
        for idx in np.arange(1, len(img_results)):
            line = img_results[idx].split()
            if (len(line) == 2 or len(line) == 3):
                item = line[0].split(":")[0]
                items.append(item)

                acc = float(line[-1].split("%")[0])*1.e-2
                accuracy.append(acc)

                if(idx>1):
                    if(len(prev_line) == 2 or len(prev_line) == 3):
                        count += 1

            elif len(line) == 6:
                x1 = float(line[2].split("=")[1].split(",")[0])
                x2 = float(line[4].split("=")[1].split(",")[0])
                y1 = float(line[3].split("=")[1].split(",")[0])
                y2 = float(line[5].split("=")[1].split(",")[0])
                bounding_box = (x1, y1, x2, y2)
                for ii in range(count):
                    bounds.append(bounding_box)
                count = 1

            prev_line = line

        # Finally, rename the predictions.jpg file
        img_base = os.path.basename(img)
        img_dir = os.path.dirname(img)
        pred_name = os.path.join(self.dir, img_base.split(".")[0]+"_predictions.jpg")
        # cmd = ""
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        notMoved = True
        while(notMoved):
            try:
                os.replace('predictions.jpg', pred_name)
                notMoved = False
            except:
                time.sleep(0.1)
        # cmd += "mv predictions.jpg "+ pred_name
        # subprocess.call([cmd], shell=True)
        img_dict = {"img":img, 'item':items, 'bounds':bounds, "accuracy":accuracy, "img_predict":pred_name}
        return img_dict

    def dump_to_pickle(self, pkl_path):
        pickle.dump(self.image_results, open(pkl_path, 'wb'))

    def load_from_pickle(self, pkl_path):
        return pickle.load(open(pkl_path, 'rb'))

if __name__ == "__main__":

    #image_dir = 'images'
    #image_list = [os.path.join(image_dir,'1041240_2.png'), os.path.join(image_dir,'254241_2.png')]
    #image_list = ["low_test.jpg", "low_test.jpg"]
    image_dir = "."
    image_list = ['test.jpg']
    dn = Darknet(image_list)
    dn.run_darknet()

    # For example, print the results of the second image
    print(dn.image_results[0])

    # now save to a pickle
    dn.dump_to_pickle(os.path.join('Results',"dn_results.pkl"))
