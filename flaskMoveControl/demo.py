from flask import Flask, request
from flask import render_template
app = Flask(__name__)



@app.route('/index')
def index():
    data = {'nickname': 'Miguel'}  # fake user
    return render_template("index.html")

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


if __name__ == '__main__':
    app.run(debug=True)