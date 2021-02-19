import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from darknet import Darknet

#IMAGE_PATH = 'main_test/test3_fix.jpg'
#!wget https://pjreddie.com/media/files/yolov3.weights

cfg_file = 'yolov3.cfg'
weight_file = 'yolov3.weights'

Model = Darknet(cfg_file)
Model.load_weights(weight_file)

#NMS
def IOU_AREA(box1, box2):


    wid_b1 = box1[2]
    ht_b1 = box1[3]
    wid_b2 = box2[2]
    ht_b2 = box2[3]

    area_b1 = wid_b1 * ht_b1
    area_b2 = wid_b2 * ht_b2

    inner = min(box1[0] - wid_b1/2.0, box2[0] - wid_b2/2.0)
    outer = max(box1[0] + wid_b1/2.0, box2[0] + wid_b2/2.0)

    UNION_WIDTH = outer - inner

    inner = min(box1[1] - ht_b1/2.0, box2[1] - ht_b2/2.0)
    outer = max(box1[1] + ht_b1/2.0, box2[1] + ht_b2/2.0)

    UNION_HEIGHT = outer - inner

    INTERSECTION_WIDTH = wid_b1 + wid_b2 - UNION_WIDTH
    INTERNSECTION_HEIGHT = ht_b1 + ht_b2 - UNION_HEIGHT

    if INTERNSECTION_HEIGHT <=0 or INTERSECTION_WIDTH <=0:
        return 0.0

    INTERSECTION_AREA = INTERSECTION_WIDTH * INTERNSECTION_HEIGHT
    UNION_AREA = UNION_HEIGHT * UNION_WIDTH

    IOU = INTERSECTION_AREA / UNION_AREA

    return IOU


def Non_Maximal_Supression(boxes,Threshold):

    L = len(boxes)

    if(L == 0):
        return 0

    Confidence = torch.zeros(L)

    for i in range(L):
        Confidence[i] = boxes[i][4]

    _,sortIds = torch.sort(Confidence, descending = True)

    get_best_bound = []

    for i in range(L):

        Ibox = boxes[sortIds[i]]

        if Ibox[4] > 0:
            get_best_bound.append(Ibox)

            for j in range(i+1, L):
                Jbox = boxes[sortIds[j]]
                if IOU_AREA(Ibox,Jbox) > Threshold:
                    Jbox[4] = 0

    return get_best_bound

def Detection(Model, Image, IOU_Threshold, NMS_Threshold):

    Model.eval()
    Image = torch.from_numpy(Image.transpose(2,0,1)).float().div(255.0).unsqueeze(0)

    Initial_Boxes = Model(Image,NMS_Threshold)
    Return_Boxes = Initial_Boxes[0][0] + Initial_Boxes[1][0] + Initial_Boxes[2][0]
    Return_Boxes = Non_Maximal_Supression(Return_Boxes, IOU_Threshold)

    print('Total Objects : ',len(Return_Boxes))

    return Return_Boxes

def print_objects(Boxes, Classes):
    total_humans = 0
    L = len(Boxes)
    for i in range(L):
        I_box = Boxes[i]
        if len(I_box) >= 7 and Classes:
            Id = I_box[6]
            if(Classes[Id]=='person'):
                total_humans+=1

    return(total_humans)

def plot_boxes(Image, Boxes, Classes, plot_labels, color = None):

    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])

    def get_color(c, x, max_val):
        R = float(x) / max_val * 5
        i = int(np.floor(R))
        j = int(np.ceil(R))

        R = R - i
        r = (1 - R) * colors[i][c] + R * colors[j][c]

        return int(r * 255)

    width = Image.shape[1]
    height = Image.shape[0]

    fig, a = plt.subplots(1,1)
    a.imshow(Image)

    for i in range(len(Boxes)):

        Ibox = Boxes[i]

        x1 = int(np.around((Ibox[0] - Ibox[2]/2.0) * width))
        y1 = int(np.around((Ibox[1] - Ibox[3]/2.0) * height))
        x2 = int(np.around((Ibox[0] + Ibox[2]/2.0) * width))
        y2 = int(np.around((Ibox[1] + Ibox[3]/2.0) * height))

        rgb = (1, 0, 0)

        if len(Ibox) >= 7 and Classes:
            cls_conf = Ibox[5]
            cls_id = Ibox[6]
            L = len(Classes)
            offset = cls_id * 123457 % L
            red   = get_color(2, offset, L) / 255
            green = get_color(1, offset, L) / 255
            blue  = get_color(0, offset, L) / 255

            if color is None:
                rgb = (red, green, blue)
            else:
                rgb = color

        width_x = x2 - x1
        width_y = y1 - y2

        rect = patches.Rectangle((x1, y2),
                                 width_x, width_y,
                                 linewidth = 2,
                                 edgecolor = rgb,
                                 facecolor = 'none')

        a.add_patch(rect)

        if plot_labels:
            conf_tx = Classes[cls_id] + ': {:.1f}'.format(cls_conf)

            X_Offset = (Image.shape[1] * 0.266) / 100
            Y_Offset = (Image.shape[0] * 1.180) / 100

            a.text(x1 + X_Offset, y1 - Y_Offset, conf_tx, fontsize = 24, color = 'k',
                   bbox = dict(facecolor = rgb, edgecolor = rgb, alpha = 0.8))

    #plt.figure(figsize=(24,14))
    #plt.rcParams['figure.figsize'] = [24.0, 14.0]
    plt.savefig('OUTPUT.jpg',bboxes='tight')
    #plt.show()

def Load(path):

    class_names = []
    with open(path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.rstrip()
        class_names.append(line)

    return class_names
