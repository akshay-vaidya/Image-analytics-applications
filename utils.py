#!/usr/bin/env python
"""
   common util funtions
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
os.environ['GLOG_minloglevel'] = '4'
import caffe
import sys
import os
import skimage
import openface
import logging
import json
from PIL import Image
from loggerfactory import *
import piexif

MAX_SIZE = float(1200)

def get_exif_rotation(iFile):
    logger=logging.getLogger(__name__)
    logger.debug("Entering exif")
    img = Image.open(iFile)
    orientation = 1
    if "exif" in img.info:
        logger.debug("found exif tags")
        exif_dict = piexif.load(img.info["exif"])

        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
            exif_bytes = piexif.dump(exif_dict)
            logger.debug("found exif orientation tags %s" % orientation)
    else:
        logger.debug("Image has no exif") 
    return orientation
 
def rotate_image(img, iFile):
    logger=logging.getLogger(__name__)
    logger.debug("Entering rotate image")
    orientation = get_exif_rotation(iFile)
    if orientation == 2:
        img = np.rot90(img, k=1, axes=(1,0))
        logger.debug("image rotated 90")
    elif orientation == 3:
        img = np.rot90(img, k=2, axes=(1,0))
        logger.debug("image rotated 180")
    elif orientation == 4:
        img = np.rot90(img, k=1, axes=(1,0))
        logger.debug("image rotated 180")
    elif orientation == 5:
        img = np.rot90(img, k=1, axes=(1,0))
        logger.debug("image rotated -90")
    elif orientation == 6:
        img = np.rot90(img, k=1, axes=(1,0))
        logger.debug("image rotated 90")
    elif orientation == 7:
        img = np.rot90(img, k=1, axes=(1,0))
        logger.debug("image rotated 90")
    elif orientation == 8:
        img = np.rot90(img, k=3, axes=(1,0))
        logger.debug("image rotated 90")
    skimage.io.imsave('testrot.jpg', img)
    return img

def format_image_bgr(img):
    logger=logging.getLogger(__name__)
    logger.debug("converting image to bgr")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def format_image_32(img):
    logger=logging.getLogger(__name__)
    logger.debug("converting image to 32")
    img32 = skimage.img_as_float(img).astype(np.float32)
    return img32

def load_image(iFile):
    logger=logging.getLogger(__name__)
    logger.debug("Entering load image")
    img = skimage.io.imread(iFile, as_grey=False)
    if len(img.shape) != 3:
        logger.error("needs to be a colour image - exiting")
        sys.exit()
    return img

def resize_image(img):
    logger=logging.getLogger(__name__)
    logger.debug("Entering resized")
    scale_factor = 1.000
    logger.debug("image size: %s,%s" % (img.shape[0],img.shape[1]))
    if img.shape[0] > img.shape[1]:
    # portrait
        if img.shape[0] > MAX_SIZE:
            scale_factor = img.shape[0] / MAX_SIZE
            img = cv2.resize(img,(int(img.shape[1]/scale_factor),int(img.shape[0]/scale_factor)))
            logger.debug("potrait image resized: %s,%s" % (img.shape[0],img.shape[1]))
    else:
        if img.shape[1] > MAX_SIZE:
            scale_factor = img.shape[1] / MAX_SIZE
            img = cv2.resize(img,(int(img.shape[1]/scale_factor),int(img.shape[0]/scale_factor)))
            logger.debug("landscape image resized: %s,%s" % (img.shape[0],img.shape[1]))
    return img

def infer2(model, inputs, names, multi_crops=False, name="model", oversample=False):
    logger=logging.getLogger(__name__)
    caffe.set_mode_gpu()
    logger.debug("Entered infer2")
    # Classify.
    item_list = []
    item={}
    results = {}
    start = time.time()
    predictions = model.predict(inputs, oversample)
    logger.info("Done in %.2f s." % (time.time() - start))
    if multi_crops:
        for crop in range(0,len(predictions)):
            top = predictions[crop].argsort()[-30:][::-1]
        #top3 = predictions[0].argsort()[-3:][::-1]
            for i in top:
                if i in results:
                    prev = results[i][1]
                    results[i] = (crop, predictions[crop][i]+prev)
                    
                else:
                    results[i] = (crop, predictions[crop][i])
        for tag in results:
            if results[tag][1] > 0.1:
                #logger.info( "%s : Probability : %.3f : crop no. : %d" % (names[tag], results[tag][1], results[tag][0])) 
                item['category']=name
                item['category_names']=names[tag]
                item['probability']=str(results[tag][1])
                item_list.append(dict(item))
    else:
        if predictions[0].max() > 0.1:
            #logger.info('%s: %s, Probability: %.3f' % (name, names[predictions[0].argmax()], predictions[0].max()))
            item['category']=name
            item['category_names']=names[predictions[0].argmax()]
            item['probability']=str(predictions[0].max())
            item_list.append(dict(item))
    #logger.debug("item list %s" % item_list) 
    return item_list

def infer(model, inputs, names, multi_crops=False, oversample=False):
    logger=logging.getLogger(__name__)
    caffe.set_mode_gpu()
    logger.debug("Entered infer")
    # Classify.
    item_list = []
    item={}
    results = {}
    start = time.time()
    predictions = model.predict(inputs, oversample)
    logger.info("Done in %.2f s." % (time.time() - start))
    if multi_crops:
        for crop in range(0,len(predictions)):
            top = predictions[crop].argsort()[-30:][::-1]
        #top3 = predictions[0].argsort()[-3:][::-1]
            for i in top:
                if i in results:
                    prev = results[i][1]
                    results[i] = (crop, predictions[crop][i]+prev)
                    
                else:
                    results[i] = (crop, predictions[crop][i])
        for tag in results:
            if results[tag][1] > 0.1:
                #logger.info( "%s : Probability : %.3f : crop no. : %d" % (names[tag], results[tag][1], results[tag][0])) 
                item['category_names']=names[tag]
                item['probability']=str(results[tag][1])
                item_list.append(dict(item))
    else:
        if predictions[0].max() > 0.1:
            #logger.info('%s: %s, Probability: %.3f' % (name, names[predictions[0].argmax()], predictions[0].max()))
            item['category']=name
            item['category_names']=names[predictions[0].argmax()]
            item['probability']=str(predictions[0].max())
            item_list.append(dict(item))
    #logger.debug("item list %s" % item_list) 
    return item_list

def infer_multiple(model, inputs, names):
    logger=logging.getLogger(__name__)
    caffe.set_mode_gpu()
    logger.debug("Entered infer")
    # Classify.
    item_list = []
    item={}
    results = {}
    start = time.time()
    predictions = model.predict(inputs, False)
    logger.info("Done in %.2f s." % (time.time() - start))
    for crop in range(0,len(predictions)):
        top = predictions[crop].argsort()[-30:][::-1]
    #top3 = predictions[0].argsort()[-3:][::-1]
        for i in top:
            if i in results:
                prev = results[i][1]
                results[i] = (crop, predictions[crop][i]+prev)
                
            else:
                results[i] = (crop, predictions[crop][i])
    for tag in results:
        if results[tag][1] > 0.1:
            #logger.info( "%s : Probability : %.3f : crop no. : %d" % (names[tag], results[tag][1], results[tag][0])) 
            item['category_names']=names[tag]
            item['probability']=str(results[tag][1])
            item_list.append(dict(item))
    return item_list

def infer_single(model, inputs, names):
    logger=logging.getLogger(__name__)
    caffe.set_mode_gpu()
    logger.debug("Entered infer")
    # Classify.
    item_list = []
    item={}
    results = {}
    start = time.time()
    predictions = model.predict(inputs)
    logger.info("Done in %.2f s." % (time.time() - start))
    for crop in range(0,len(predictions)):
        top = predictions[crop].argsort()[-30:][::-1]
        for i in top:
            if i in results:
                prev = results[i][1]
                results[i] = (crop, predictions[crop][i]+prev)
                
            else:
                results[i] = (crop, predictions[crop][i])
    max_seen = 0
    for tag in results:
        if results[tag][1] > max_seen:
            max_seen = results[tag][1]
            #logger.info( "%s : Probability : %.3f : crop no. : %d" % (names[tag], results[tag][1], results[tag][0])) 
            item['category_names']=names[tag]
            item['probability']=str(results[tag][1])
    
    return item
