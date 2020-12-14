from flask import Flask, request
from flask import render_template
app = Flask(__name__)



@app.route('/index')
def index():
    data = {'nickname': 'Miguel'}  # fake user
    return render_template("index.html")

@app.route('/video')
def video():
    return render_template("video.html")

@app.route('/')
def hello_world():
    return 'hello world'



@app.route('/action')
def action():
    print(request)
    user_name = request.args.get('name')
    user_age = request.args.get('passwd')

    print("user_name = %s, user_age = %s" % (user_name, user_age))
    print("request.url" , request.url)
    print("request.args" , request.args)
    # data = request.data()
    # data = request.get_json()
    # print('request data',data)
    # data = request.args
    # print('request data', data.get('data'))
    return '123'

@app.route('/api/device/login',methods=['GET', 'POST'])
def login():
    print(request)
    print('request.data', request.data)
    return 'login'

@app.route('/api/device/heart',methods=['GET', 'POST'])
def heart():
    print(request)
    print('request.data',request.data)
    return 'heart'


@app.route('/api/device/event',methods=['GET', 'POST'])
def event():
    print(request)
    print('request.data',request.data)
    return 'event'


@app.route('/api/device/ statistics',methods=['GET', 'POST'])
def statistics():
    print(request)
    print('request.data', request.data)
    return 'statistics'


@app.route('/api/device/time',methods=['GET', 'POST'])
def time_sync():
    print(request)
    print('request.data', request.data)
    return 'statistics'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8555,debug=True)

    # rtsp地址 "rtsp://admin:123456@192.168.1.100:554/ch1/0"
