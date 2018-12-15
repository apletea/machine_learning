from __future__ import print_function
import os
from multiprocessing import Pool
import numpy as np
import tqdm
from PIL import Image
import sys

THREAD_NUM = 16

def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

classes = {}
file_tree = os.walk('/home/kirill.matsaberydze/Projects/whales/parsed')
next(file_tree)

def read_and_resize(path, class_):
   img = Image.open(path).resize((224,224))
   return class_, img

def flat_iter(arr):
    for i in arr:
        if isinstance(i, list):
            flat_iter_ = flat_iter(i)
            yield next(flat_iter)
        else:
            yield i

def read_entry(entry):
    eprint(f"reading {entry[0]}")
    return entry[0], [read_and_resize(f'{entry[0]}/{x}', entry[0].split("/")[-1]) for x in entry[2]]

if not os.path.isfile('/home/kirill.matsaberydze/Projects/whales/classes.pickle'):

    eprint("reading files")

    with Pool(THREAD_NUM) as p:
       classes =  p.map(read_entry, file_tree)
    
    eprint("readed")
    
    classes = dict(classes)
    
    with open('/home/kirill.matsaberydze/Projects/whales/classes.pickle', 'wb') as handle:
        pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:

    eprint("restoring dict")

    with open('/home/kirill.matsaberydze/Projects/whales/classes.pickle', 'rb') as handle:
        classes = pickle.load(handle)

def proc(image, path):
    min_v = 1000000000000000
    for test_img_name in os.listdir('/home/kirill.matsaberydze/Projects/whales/kaggle_data/train'):
         test_img = cv2.resize(cv2.imread(f'/home/kirill.matsaberydze/Projects/whales/kaggle_data/train/{test_iamge_name}'),(224,224))
         abs_diff = cv2.absdiff(test_img, np.array(image))
         diff = np.sum(abs_diff) 
         if (min_v > diff):
            min_v = diff
            if (min_v < 250000):
                 print(f"match for {path} : {str(min_v)}, image : {f'/home/kirill.matsaberydze/Projects/whales/kaggle_data/train/{test_iamge_name}'}")
            else:
                 eprint(f"new min for {path} : {str(min_v)}, image : {f'/home/kirill.matsaberydze/Projects/whales/kaggle_data/train/{test_iamge_name}'}")
        
with Pool(THREAD_NUM) as p:
    p.map(lambda x: proc(x[0], x[1]), flat_iter(classes.values()))
