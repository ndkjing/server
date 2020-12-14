import json

with open('xinan_baidu.json','r') as f:
    data = json.load(f)

for point in data:
    print(point,type(point))