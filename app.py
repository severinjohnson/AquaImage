from flask import Flask, request, send_file, render_template
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
from io import BytesIO

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

        # Visualize the predictions on the image
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        overlayed_image = out.get_image()[:, :, ::-1]

        # Convert image to bytes
        _, buffer = cv2.imencode('.jpg', overlayed_image)
        io_buf = BytesIO(buffer)

        return send_file(io_buf, mimetype='image/jpeg', as_attachment=True, attachment_filename='prediction.jpg')

if __name__ == '__main__':
    app.run(debug=True)
