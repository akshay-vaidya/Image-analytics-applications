#!/usr/bin/env python


import os
from modelloader import ModelCache
import numpy as np
from loggerfactory import LoggerManager
import keras.backend as K
os.environ['KERAS_BACKEND'] = 'theano'
reload(K)
K.set_image_dim_ordering('th')
import dlib
import cv2

class Faces(object):
    def __init__(self, img):
        self.logger = LoggerManager().getLogger(__name__)
        self.image_input = img
        self.factor = 'gender'
        self.responselist = ['Female', 'Male']        
    @staticmethod
    def get_face_coordinates(im):
        face_detector = dlib.cnn_face_detection_model_v1('/home/ec2-user/akshayaa/age_gender_model/wiki/mmod_human_face_detector.dat')
        dets = face_detector(im, 1)
        return dets
    @staticmethod
    def coords_standardize(d):
        d1= [d.rect.top(),d.rect.bottom(),d.rect.left(),d.rect.right()]
        d2 = [0 if i < 0 else i for i in d1]
        return d2
    @staticmethod
    def check_face_position(det,img_pos,img_start,factor):
        if abs(img_start - det) > img_pos/2 : 
            return 4 * factor
        else:
            return factor
    @staticmethod
    def rows_occupied_by_face(image_with_face1):
        cords1 = Faces.get_face_coordinates(image_with_face1)
        pts1 = Faces.coords_standardize(cords1[0])
        rows= pts1[1]- pts1[0]
        return (float(rows)/float(image_with_face1.shape[0]))*100
    @staticmethod
    def cols_occupied_by_face(image_with_face2):
        cords2 = Faces.get_face_coordinates(image_with_face2)
        pts2 = Faces.coords_standardize(cords2[0])
        cols= pts2[3]-pts2[2]
        return (float(cols)/float(image_with_face2.shape[1]))*100
    @staticmethod
    def crop_face(img,dets, top_crop= True,bottom_crop=True,left_crop=True,right_crop=True):
        dets_list2= Faces.coords_standardize(dets[0])
        top_factor= 1.5
        bottom_factor= 1.5
        left_factor= 1.5
        right_factor=1.5
        crop_top=0
        crop_bottom=img.shape[0]
        crop_left=0
        crop_right=img.shape[1]
        if top_crop:
            top_factor= Faces.check_face_position(dets_list2[0],img_start=0,img_pos=img.shape[0],factor=1.5)
            crop_top= int(dets_list2[0] - (dets_list2[0]/top_factor))
        if bottom_crop:
            bottom_factor = Faces.check_face_position(dets_list2[1],img_start=img.shape[0],img_pos=img.shape[0],factor=1.5)
            crop_bottom= int(dets_list2[1] + ((img.shape[0]- dets_list2[1])/bottom_factor))
        if left_crop:
            left_factor= Faces.check_face_position(dets_list2[2],img_start=0,img_pos=img.shape[1],factor=1.5)            
            crop_left= int(dets_list2[2] - (dets_list2[2]/left_factor))
        if right_crop:
            right_factor= Faces.check_face_position(dets_list2[3],img_start=img.shape[1],img_pos=img.shape[1],factor=1.5)
            crop_right= int(dets_list2[3] + ((img.shape[1]- dets_list2[3])/right_factor))
        return img[crop_top:crop_bottom,crop_left:crop_right]
    @staticmethod
    def recrop_along_rows(f1):
        f=f1
        rows_pct_occupied= Faces.rows_occupied_by_face(f)
        cropped_face_cords = Faces.get_face_coordinates(f)
        cropped_face_pts = Faces.coords_standardize(cropped_face_cords[0])
        #print 'rows at starting'
        #print rows_pct_occupied
        j=0
        while rows_pct_occupied<40:
            #print rows_pct_occupied
            #print 'entered rows'
            if (cropped_face_pts[0] > f.shape[0]/2) and ((f.shape[0]-cropped_face_pts[1]) > f.shape[0]/2):
                f= Faces.crop_face(dets=cropped_face_cords, img=f, top_crop=True, bottom_crop=True,left_crop=False,right_crop=False)
            elif cropped_face_pts[0] > f.shape[0]/2:
                f= Faces.crop_face(dets=cropped_face_cords, img=f, top_crop=True, bottom_crop=False,left_crop=False,right_crop=False)
            elif (f.shape[0]-cropped_face_pts[1]) > f.shape[0]/2:
                f= Faces.crop_face(dets=cropped_face_cords, img=f, top_crop= False, bottom_crop=True,left_crop=False,right_crop=False)
            rows_pct_occupied= Faces.rows_occupied_by_face(f)
            cropped_face_cords = Faces.get_face_coordinates(f)
            cropped_face_pts = Faces.coords_standardize(cropped_face_cords[0])
            j+=1
            if j==5:
                break
        return f
    @staticmethod
    def recrop_along_cols(f1):
        f=f1
        cols_pct_occupied= Faces.cols_occupied_by_face(f)
        cropped_face_cords = Faces.get_face_coordinates(f)
        cropped_face_pts = Faces.coords_standardize(cropped_face_cords[0])
        #print 'cols at starting'
        #print cols_pct_occupied
        j=0
        while cols_pct_occupied<40:
            #print 'enetered cols'
            #print cols_pct_occupied
            if (cropped_face_pts[2] > f.shape[1]/2) and ((f.shape[1]-cropped_face_pts[3]) > f.shape[1]/2):
                #print 'yes'
                f= Faces.crop_face(dets=cropped_face_cords, img=f, top_crop=False, bottom_crop=False,left_crop=True,right_crop=True)
            elif cropped_face_pts[2] > f.shape[1]/2:
                #print 'tt'
                f= Faces.crop_face(dets=cropped_face_cords, img=f, top_crop=False, bottom_crop=False,left_crop=True,right_crop=False)
            elif (f.shape[1]-cropped_face_pts[3]) > f.shape[1]/2:
                #print 't'
                f= Faces.crop_face(dets=cropped_face_cords, img=f, top_crop=False, bottom_crop=False,left_crop=False,right_crop=True)
            cols_pct_occupied= Faces.cols_occupied_by_face(f)
            cropped_face_cords = Faces.get_face_coordinates(f)
            cropped_face_pts = Faces.coords_standardize(cropped_face_cords[0])
            j+=1
            if j==5:
                break
        return f
    @staticmethod
    def get_cropped_face(img):
        cords= Faces.get_face_coordinates(img)
        cropped_images=[]
        facial_points_response=[]
        if len(cords)==0:
            return cropped_images, facial_points_response
        if len(cords)==1:
            facial_points_response.append(Faces.coords_standardize(cords[0]))
            cropped_face= Faces.crop_face(img=img, dets=cords)
            cropped_face=Faces.recrop_along_rows(cropped_face)
            cropped_face=Faces.recrop_along_cols(cropped_face)
            cropped_images.append(cropped_face)
            return cropped_images, facial_points_response
        elif len(cords) > 1:
            ind=0
            left_most_order=[]
            right_most_order=[]
            order_index=[]
            facial_points_response=[]
            for d in cords:
                cropped_face_pts=Faces.coords_standardize(d)
                left_most_order. append (cropped_face_pts[2])
                right_most_order.append(cropped_face_pts[3])
                seq = sorted(left_most_order)
                order_index = [seq.index(v) for v in left_most_order]
            for d in cords:
                cropped_face_pts=Faces.coords_standardize(d)
                facial_points_response.append(cropped_face_pts)
                crop_top= int(cropped_face_pts[0] - (cropped_face_pts[0]/1.5))
                crop_bottom= int(cropped_face_pts[1] + ((img.shape[0]- cropped_face_pts[1])/1.5))
                if order_index[ind] == 0:
                    crop_left= cropped_face_pts[2] - (cropped_face_pts[2]/2)
                    next_right=sorted(left_most_order, key=lambda x:abs(x-left_most_order[ind]))[1]
                    crop_right= cropped_face_pts[3] + abs((cropped_face_pts[3]- next_right)/2)
                elif order_index[ind] == len(cords)-1:
                    #print 'right-most'
                    next_left= sorted(right_most_order, key=lambda x:abs(x-right_most_order[ind]))[1]
                    crop_left= cropped_face_pts[2] - abs((cropped_face_pts[2]-next_left)/2)
                    crop_right= cropped_face_pts[3] + ((img.shape[1]- cropped_face_pts[3])/2)
                else:
                    #print 'other'
                    next_right=sorted(left_most_order, key=lambda x:abs(x-left_most_order[ind]))[1]
                    next_left= sorted(right_most_order, key=lambda x:abs(x-right_most_order[ind]))[1]
                    crop_left= cropped_face_pts[2] - abs((cropped_face_pts[2]-next_left)/2)
                    crop_right= cropped_face_pts[3] + abs((cropped_face_pts[3]- next_right)/2)
                ind+=1
                cropped_face = img[crop_top:crop_bottom,crop_left:crop_right]
                cropped_face=Faces.recrop_along_rows(cropped_face)
                cropped_face=Faces.recrop_along_cols(cropped_face)
                cropped_images.append(cropped_face)
            return cropped_images, facial_points_response
    def get_prediction(self):
        container=[]
        facial_images, face_points= Faces.get_cropped_face(self.image_input)
        k=0
        if len(face_points)==0:
            return 'No face detected'
        else:
            for ims in facial_images:
                self.cropped_image= ims
                #self.cropped_image= cv2.resize(self.cropped_image, (250, 250))
                self.facial_points= face_points[k]
                faces = dlib.full_object_detections()
                dets1=Faces.get_face_coordinates(self.cropped_image)
                sp = dlib.shape_predictor('/home/ec2-user/akshayaa/age_gender_model/adience/shape_predictor_5_face_landmarks.dat')
                faces.append(sp(self.cropped_image, dets1[0].rect))
                images = dlib.get_face_chips(self.cropped_image, faces, size=250, padding=1)
                random_crop_left_top= images[0][0:227,0:227]
                cropped_img=images[0]
                random_crops_center=cropped_img[(cropped_img.shape[0]/2)-114:(cropped_img.shape[0]/2)+113,(cropped_img.shape[1]/2)-114: (cropped_img.shape[1]/2)+113]
                random_crop_left_bottom= cropped_img[cropped_img.shape[0]-227:cropped_img.shape[0],0:227]
                random_crop_right_top= cropped_img[0:227,cropped_img.shape[1]-227:cropped_img.shape[1]]
                random_crop_right_bottom= cropped_img[cropped_img.shape[0]-227:cropped_img.shape[0],cropped_img.shape[1]-227:cropped_img.shape[1]]
                self.random_crops= [random_crop_left_top,random_crops_center, random_crop_left_bottom, random_crop_right_top, random_crop_right_bottom ]
                gender_model = ModelCache().getModel(self.factor)
                classifier_gender = gender_model.getModelClassifier()
                self.set_modelforinference(classifier_gender)
                self.face_id=k
                container.append(self.infer())
                k+=1
            return container
                    
    
    def set_modelforinference(self, classifier_for_inference):
        self.model_for_inference = classifier_for_inference

    def infer(self):
        male_prob=[]
        female_prob=[]
        points={}
        predo={}
        predo2={}
        predo3={}
        for rand in self.random_crops:
            resized_image = rand.transpose(2, 0, 1)
            resized_image = np.expand_dims(resized_image, axis=0)
            labels=['Female','Male']
            male_prob.append( list(self.model_for_inference.predict(resized_image)[0])[1])
            female_prob.append(list(self.model_for_inference.predict(resized_image)[0])[0])
        
        male_avg_prob= sum(male_prob) / 5.0
        female_avg_prob= sum(female_prob) / 5.0
        points['top']=  self.facial_points[0]
        points['Bottom']= self.facial_points[1]
        points['left']= self. facial_points[2]
        points['right']= self. facial_points[3]
        predo['facial points']= points
        predo['gender detected']= {'Male': male_avg_prob, 'female' : female_avg_prob}
        predo['face_id']= self.face_id
        predo2['Dimensions']= {'Width': self.image_input.shape[1], 'Height': self.image_input.shape[0]}
        predo2['faces']=predo 
        predo3['image']=predo2
        return predo3
        
