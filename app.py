from flask import Flask, request, send_file, render_template
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
import base64
from flask import jsonify


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Read the image
        print("Received file:", file.filename)
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        im = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Configuration for the model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = 'cpu'

        # Create predictor and make prediction
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)

        # Extract predictions
        predictions = []
        instances = outputs["instances"].to("cpu")
        for i in range(len(instances)):
            prediction_class = instances.pred_classes[i]
            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[prediction_class]
            if class_name not in predictions:
                predictions.append(class_name)

        # Print predictions to console
        #print("Predicted classes:", predictions)

        # Visualize the predictions on the image
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        overlayed_image = out.get_image()[:, :, ::-1]

        # Convert image to base64 string
        _, buffer = cv2.imencode('.jpg', overlayed_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Construct and send JSON response
        return jsonify({'image': base64_image, 'predictions': predictions})

        return response
if __name__ == '__main__':
    app.run(debug=True)
