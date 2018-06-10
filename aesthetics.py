#!/usr/bin/env python
"""
Class to detect place / scene of an image
Sends the image to a model that return results
"""

import os
from modelloader import ModelCache
import numpy as np
from loggerfactory import LoggerManager
import keras.backend as K
os.environ['KERAS_BACKEND'] = 'theano'
reload(K)
K.set_image_dim_ordering('th')

class Aesthetics(object):
    """ gives Aesthetic scores for images """
    def __init__(self, img):
        self.logger = LoggerManager().getLogger(__name__)
        self.image_input = img
        self.factor = 'Aesthetics'
        self.model_for_inference = None
        self.model_reference = None
    def get_prediction(self):
        """ get classifier prediction """
        self.logger.info("Entered get_prediction- aesthetics model")
        container = {}
        aesthetics_model = ModelCache().getModel(self.factor)
        classifier_aesthetics = aesthetics_model.getModelClassifier()
        self.model_reference = aesthetics_model.getWeightsReference()
        self.set_modelforinference(classifier_aesthetics)
        container = self.infer()
        return container

    def set_modelforinference(self, classifier_for_inference):
        """ set model to perform inference """
        self.model_for_inference = classifier_for_inference

    def infer(self):
        """ get inference """
        img_input = np.expand_dims(self.image_input, axis=0)
        softmax_output = self.model_for_inference.predict(img_input)
        print softmax_output
        return {'Model Reference' : self.model_reference, 'Aesthetics Score Prediction' :\
{'Good' : str(softmax_output[0][1]), \
'Bad' : str(softmax_output[0][0])}}
