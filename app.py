from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from compile import *
import cv2

app = Flask(__name__)

@app.route('/')
def upload_file1():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename('input.jpg'))

    path = 'coco.names'
    Classes = Load(path)

    # Read image as Image
    plt.figure(figsize=(24,14))
    plt.rcParams['figure.figsize'] = [24.0, 14.0]
    Image = cv2.imread('input.jpg')
    orig = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    resize = cv2.resize(orig, (Model.width, Model.height))
    IOU_Threshold = 0.4
    NMS_Threshold = 0.6
    BOXES = Detection(Model, resize, IOU_Threshold, NMS_Threshold)
    s = print_objects(BOXES, Classes)
    s = str(s)
    plot_boxes(orig, BOXES, Classes, plot_labels = True)
    text = 'File uploaded and processed successfully. Total humans identified = '
    s = text + s
    return s

if __name__ == '__main__':
   app.run(debug = True)
