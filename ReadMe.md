# Image Analyser REST Services
A set of new services developed by Synchronoss Innovation Center to perform neural networking based image analytics

## About This project
Pulls together different published neural networking models to perform:
1. Object tagging (using Inception v3, currently on caffe)
2. Place/scene categorisation (CNN based, on caffe)
3. Image aesthetics & quality (using custom resnet50 based model running on keras/tensorflow)
4. Face detection (using MTCNN (caffe) or DLIB)
5. Age / Gender detection of faces (CNN based on caffe)
6. Face recognition (using Openface, theano, torch) 

## Dependencies
- Python 2.7 (64 bit version)
- Amazon Deep Learning AMI
- Minimum 16 GB of RAM
- GPU for decent speed + all the CUDA dependencies

## How to run?
1. Clone project repository from : [Stash URL](https://stash.synchronoss.net/projects/IN/repos/text-analyser-poc/browse)
2. Install: caffe, keras, dlib, opencv, theano, openface, torch, tensorflow
3. quickie - python ./apifactory
or
3. less quick but better as service - copy supervisord.conf to /etc, add /etc/rc3.d/S101supervisord, start supervisord

4. Now the REST services will be accesible at port 7000.

## Sample Tests:
    
-  To get object tags in an image:
    ```$ curl -F file=@Google\ Drive/IMG_5759.JPG http://ec2-35-166-7-221.us-west-2.compute.amazonaws.com:7000/things```

-  To get place tags in an image:
    ```$ curl -F file=@Google\ Drive/IMG_5759.JPG http://ec2-35-166-7-221.us-west-2.compute.amazonaws.com:7000/places```

-  To get aesthetics tags in an image:
    ```$ curl -F file=@Google\ Drive/IMG_5759.JPG http://ec2-35-166-7-221.us-west-2.compute.amazonaws.com:7000/aesthetics```

-  To get face tags in an image:
    ```$ curl -F file=@Google\ Drive/IMG_5759.JPG http://ec2-35-166-7-221.us-west-2.compute.amazonaws.com:7000/faces```

-  To get face tags in an image and build a person DB for a particular repo/user:
    ```$ curl -F file=@Google\ Drive/IMG_5759.JPG http://ec2-35-166-7-221.us-west-2.compute.amazonaws.com:7000/faces/1234```


    
## License
The content of this repository is licensed under [Synchronoss Technologies](http://synchronoss.com/).
