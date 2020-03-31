from __future__ import division, print_function
from cv2 import cv2
import numpy as np

import numpy as np
import pandas as pd

import numpy as np
from PIL import Image
import os
from itertools import chain
import csv


def run_length(label):
    x = label.transpose().flatten()
    y = np.where(x>0.5)[0]
    if len(y)<10:
        return []
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    leng = end - start
    res = [[s+1,l+1] for s,l in zip(list(start),list(leng))]
    res = list(chain.from_iterable(res))
    return res


if __name__ == '__main__':
    input_path = './realimage/'
    
    targ = ((10, 10), (10, 10))

    for i in range(1, 5509):
        img = cv2.imread('./data/real_test/mask/'+str(i)+'.png', 0)
        img = np.pad(img, targ, mode='constant', constant_values=0)
        mask_rle = run_length(img)
        print(str(mask_rle).replace(',', '').replace('[', '').replace(']', ''))
        mask_rle = str(mask_rle).replace(',', '').replace('[', '').replace(']', '')
        with open('submission.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, mask_rle])
