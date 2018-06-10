import os
import imp
import numpy as np
from singleton import Singleton
import caffe
from loggerfactory import LoggerManager
from keras.models import model_from_json,model_from_yaml
import imp



class Modelloader(object):
    """This class loads all the models """
    def __init__(self, app_config):
        """ Initialize model configuration variables """
        self.modelConfig = app_config.get("models")
        self.logger = LoggerManager().getLogger("IMAGE_ANALYSER-MODELLOADER")
    def loadmodels(self):
        """ Load caffe and keras models """
        print self.modelConfig
        for modelName in self.modelConfig:
            #self.logger.debug("model name %s", modelName)
            modelProps = self.modelConfig.get(modelName)
            myModel = Model(modelName)
            if modelProps['framework'] == 'caffe':
                caffeModel = modelProps.get("caffemodel")
                prototxt = modelProps.get("prototxt")
                mean = modelProps.get("mean")
                modelMean = np.load(mean).mean(1).mean(1)
                image_dims = (256, 256)
                classifier = caffe.Classifier(str(prototxt), \
str(caffeModel), image_dims, modelMean, \
input_scale=None, raw_scale=255, channel_swap=(2, 1, 0))
                myModel.setModelClassifier(classifier)
                myModel.setModelMean(modelMean)
            if modelProps['framework'] == 'keras':
                kerasModel = modelProps.get("kerasmodel")
                input_dim = modelProps.get("input_dim")
                prototxt = modelProps.get("prototxt")
                if os.path.splitext(kerasModel)[1] == '.py':
                    model_file = imp.load_source('python_module', kerasModel)
                    loaded_model = getattr(model_file,'alexnet')
                    classifier = loaded_model()
                if os.path.splitext(kerasModel)[1] == '.json':
                    model_file = open(kerasModel, 'r')
                    loaded_model = model_file.read()
                    classifier = model_from_json(loaded_model)
                    model_file.close()
                classifier.load_weights(prototxt)
                dummy_object = np.random.rand(1, 3, input_dim, input_dim)
                print classifier.predict(dummy_object)
                myModel.setModelClassifier(classifier)
            myModel.setModelList(modelProps.get("list"))
            myModel.setWeightsReference(prototxt.split('/')[-1])
            ModelCache.loadModel(modelName, myModel)
                
                
class Model(object):
    """model getter and setter methods """
    def __init__(self, modelName):
        """ Initialize model configuration variables """
        self.m_ModelName = modelName
        self.m_ModelClassifier = None
        self.m_ModelMean = None
        self.m_ModelCategories = None
        self.m_ModelList = None
        self.m_WeightsReference = None
    def getModelClassifier(self):
        """get model classifier """
        return self.m_ModelClassifier
    def setModelClassifier(self, modelClassifier):
        """set model classifier """
        self.m_ModelClassifier = modelClassifier
    def getModelMean(self):
        """get model mean """
        return self.m_ModelMean
    def setModelMean(self, modelMean):
        """set model mean """
        self.m_ModelMean = modelMean
    def getModelName(self):
        """get model name """
        return self.m_ModelName
    def setModelName(self, modelName):
        """set model name """
        self.m_ModelName = modelName
    def getModelList(self):
        """get model list """
        return self.m_ModelList
    def setModelList(self, modelList):
        """set model list """
        self.m_ModelList = modelList
    def setWeightsReference(self, weights_name):
        """set weights file version number """
        self.m_WeightsReference = weights_name
    def getWeightsReference(self):
        """ get model reference """
        return self.m_WeightsReference

class ModelCache(object):
    """storing model objects """
    __metaclass__ = Singleton

    _modelMap = {}
    def __init__(self, *args, **kwargs):
        """.. """
        pass

    @staticmethod
    def getModel(name):
        """gets every model """
        return ModelCache._modelMap[name]

    @staticmethod
    def loadModel(name, model):
        """loads every model """
        ModelCache._modelMap[name] = model
