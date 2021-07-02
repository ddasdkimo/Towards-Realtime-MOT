from flask import Response, Flask, render_template, request, make_response
import threading
import argparse
import datetime
import time
import cv2
import json
import numpy as np
import time
from track import conversion_frame_init, conversion_frame
from utils.parse_config import parse_model_cfg
outputFrame = None
showFrame = None
lock = threading.Lock()
plock = threading.Lock()

app = Flask(__name__)

# source = 0
# cap = cv2.VideoCapture(source)

cap = cv2.VideoCapture(0)
rame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 視訊縮放大小
resizew = 0
resizeh = 0
# 模型大小
modelsizew = 0
modelsizeh = 0

# cap = cv2.VideoCapture("tet.jpeg")
# 設定影像的尺寸大小
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
time.sleep(1.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def stream():
    global outputFrame, lock
    if cap.isOpened():
        while True:
            ret_val, frame = cap.read()
            if frame.shape:
                with lock:
                    outputFrame = frame.copy()
            else:
                continue
    else:
        print('camera open failed')

def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular 
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

def segmentT():
    global outputFrame, lock, showFrame, plock, qualityT, minisize
    while True:
        start = time.time()
        with lock:
            if outputFrame is None:
                time.sleep(0.0001)
                continue
            frame = outputFrame
            outputFrame = None
        with plock:
            # 耗時處理
            img, img0 = cov_frame(frame)
            online_im,data = conversion_frame(img, img0)
            showFrame = online_im
            
        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        # print( "Estimated frames per second : {0}".format(fps))
# 圖像縮放
def cov_frame(frame):
    img0 = cv2.resize(frame, (resizew, resizeh))
    # Padded resize
    img, _, _, _ = letterbox(img0, height=modelsizeh, width=modelsizew)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
    return  img, img0
# 換算需要縮放大小
def get_size(vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw *a), int(vh*a)

def generate():
    global showFrame, plock
    while True:
        # wait until the lock is acquired
        with plock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if showFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", showFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=False, default='0.0.0.0',
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=False, default=8000,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")

    ap.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    ap.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    ap.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    ap.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    ap.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    ap.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    ap.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    ap.add_argument('--input-video', type=str, help='path to the input video')
    ap.add_argument('--output-format', type=str, default='video', choices=['video', 'text'], help='Expected output format. Video or text.')
    ap.add_argument('--output-root', type=str, default='results', help='expected output root path')
    opt = ap.parse_args()
    args = vars(ap.parse_args())
    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]
    modelsizew = opt.img_size[0]
    modelsizeh = opt.img_size[1]
    resizew, resizeh = get_size(w, h, opt.img_size[0], opt.img_size[1])
    conversion_frame_init(opt=opt,frame_rate = rame_rate)
    t = threading.Thread(target=stream)
    t.daemon = True
    t.start()

    t1 = threading.Thread(target=segmentT)
    t1.daemon = True
    t1.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)


