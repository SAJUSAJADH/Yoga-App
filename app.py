from flask import Flask, render_template, Response
from hupyoga1 import predict
from flask_cors import CORS
from multiprocessing import Value


app = Flask(__name__)
CORS(app)

stop_flag = Value('b', True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(predict(stop_flag), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_predict', methods=['POST'])
def stop_predict():
    global stop_flag
    stop_flag.value = False
    return 'OK'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')