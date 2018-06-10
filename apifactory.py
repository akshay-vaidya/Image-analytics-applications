import logging
from loggerfactory import *
from configlistener import ConfigListener
from flask import Flask
from flask import render_template, jsonify, request
from things import Things
from gender import Gender
from places import Places
from people import People
from aesthetics import Aesthetics
from food import Food
from modelloader import Modelloader
from utils import load_image, rotate_image, limit_size_image
import traceback
import cv2
from receipts import Receipts
from flasgger import Swagger

app = Flask(__name__)
#app.debug = True
app_config = ConfigListener().getConfigObject()
myloader = Modelloader(app_config)
myloader.loadmodels()
swagger = Swagger(app)

@app.errorhandler(Exception)
def handle_500(e=None):
    app.logger.error(traceback.format_exc())
    return 'Internal server error occured' + traceback.format_exc(), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/things', methods=['POST'])
def get_things():
    """
    This API analyses an image and returns the objects classification
    ---
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: formData
        name: upfile
        type: file
        required: true
        description: Image file to analyse
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    #logger=logging.getLogger(__name__)
    #logger.info("Calling /things")
    model = Things()
    query_class = request.args.get('class')
    iFile = request.files.getlist('upfile')[0]
    img = load_image(iFile)
    img = rotate_image(img, iFile)
    img = limit_size_image(img)
    response = model.get_tags(img, query_class)
    return jsonify(response)

@app.route('/places', methods=['POST'])
def get_places():
    """
    This API analyses an image an returns the scenic place classification
    ---
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: formData
        name: upfile
        type: file
        required: true
        description: Image file to analyse
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    #logger=logging.getLogger(__name__)
    #logger.debug("Calling /places")
    model = Places()
    query_class = request.args.get('class')
    iFile = request.files.getlist('upfile')[0]
    img = load_image(iFile)
    img = rotate_image(img, iFile)
    img = limit_size_image(img)
    response = model.get_tags(img, query_class)
    return jsonify(response)


@app.route('/people', methods=['POST'])
def get_faces():
    """
    This API searches an image for faces and returns the location of each face
    ---
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: formData
        name: upfile
        type: file
        required: true
        description: Image file to analyse
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    #logger=logging.getLogger(__name__)
    #logger.info("Calling prediction data for /people")
    response = None
    model = People()
    query_class = request.args.get('class')
    iFile = request.files.getlist('upfile')[0]
    img = load_image(iFile)
    img = rotate_image(img, iFile)
    img = limit_size_image(img)
    response = model.get_tags_coords(img, query_class)
    return jsonify(response)

@app.route('/people/id/<repo>', methods=['POST'])
def get_face_id(repo):
    """
    This API searches an image for faces and returns the person face grouop the name of a face group in the DB for a repo.
    ---
    paths:
      /people/{repo}
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: formData
        name: upfile
        type: file
        required: true
        description: Image file to analyse
      - in: path
        name: repo
        schema:
          type: integer
        required: true
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    #logger=logging.getLogger(__name__)
    #logger.info("Calling prediction data for /faces")
    response = None
    model = People(repo)
    query_class = request.args.get('class')
    iFile = request.files.getlist('upfile')[0]
    img = load_image(iFile)
    img = rotate_image(img, iFile)
    img = limit_size_image(img)
    response = model.get_tags_face(img, query_class)
    return jsonify(response)

@app.route('/people/id/<repo>/<personid>/<name>', methods=['POST'])
def set_person_name(repo, personid, name):
    """
    This API allows you to set the name of a person group in the DB for a repo
    ---
    paths:
      /people/{repo}/{personid}/{name}
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: path
        name: repo
        schema:
          type: integer
        required: true
      - in: path
        name: personid 
        schema:
          type: integer
        required: true
      - in: path
        name: name 
        schema:
          type: string
        required: true
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    logger=logging.getLogger(__name__)
    logger.info("Calling prediction data for /people/<repo>/<personid>/<name>")
    model = People(repo)
    query_class = request.args.get('class')
    response = model.set_name(repo, personid, name)
    return jsonify(response)

@app.route('/people/gender', methods=['POST'])
def get_gender():
    """
    This API returns the predicted gender of all faces detected in an image.
    Call this api passing a coloured image.
    ---
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: formData
        name: upfile
        type: file
        required: true
        description: Upload your file
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    response = None
    iFile = request.files.getlist('upfile')[0]
    img = load_image(iFile)
    img = rotate_image(img, iFile)
    img = limit_size_image(img)
    model = Gender(img)
    response = model.get_prediction()
    return jsonify(response)

@app.route('/aesthetics', methods=['POST'])
def get_aesthetics():
    """
    This API gives aesthetics score on images.
    Call this api passing a coloured image.
    ---
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: formData
        name: upfile
        type: file
        required: true
        description: Upload your file
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    #logger=logging.getLogger(__name__)
    #logger.info("Calling /aesthetics")
    response = None
    iFile = request.files.getlist('upfile')[0]
    img = load_image(iFile)
    img = rotate_image(img, iFile)
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1)
    model = Aesthetics(img)
    response = model.get_prediction()
    return jsonify(response)

@app.route('/receipts', methods=['POST'])
def receipts_classification():
    """
    This API returns whether the image uploaded is a receipt or not.
    Call this api passing a coloured image.
    ---
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: formData
        name: upfile
        type: file
        required: true
        description: Upload your file
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    #logger=logging.getLogger(__name__)
    #logger.info("Calling /aesthetics")
    response = None
    iFile = request.files.getlist('upfile')[0]
    print iFile
    img = load_image(iFile)
    img = rotate_image(img, iFile)
    img = cv2.resize(img, (512, 512))
    img = img.transpose(2, 0, 1)
    model = Receipts(img)
    response = model.get_prediction()
    return jsonify(response)

@app.route('/food', methods=['POST'])
def food_detection():
    """
    This API returns whether the image uploaded is a food or not.
    Call this api passing a coloured image.
    ---
    tags:
      - Image Analyzer
    consumes:
      - multipart/form-data
    produces: 
      -application/json
    parameters:
      - in: formData
        name: upfile
        type: file
        required: true
        description: Upload your file
    responses:
      500:
        description: Error, something went wrong!
      200:
        description: Detection success!
    """
    #logger=logging.getLogger(__name__)
    #logger.info("Calling /aesthetics")
    response = None
    iFile = request.files.getlist('upfile')[0]
    print iFile
    img = load_image(iFile)
    img = rotate_image(img, iFile)
    img = cv2.resize(img, (227, 227))
    img = img.transpose(2, 0, 1)
    model = Food(img)
    response = model.get_prediction()
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
