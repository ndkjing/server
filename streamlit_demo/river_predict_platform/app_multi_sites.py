# refer   https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py
"""
多个站点预测
"""
from keras.models import load_model

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from datetime import datetime,timedelta
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import psycopg2
from PIL import ImageFont, ImageDraw, Image
import cv2
import json
from tqdm import tqdm

DATE_TIME = "date/time"

st.title("河流预测可视化")
# st.markdown(
# """
# 河流 降雨量 与 环保相关指数关系预测模型可视化显示
# """)

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv("uber-raw-data-sep14.csv", nrows=nrows)
    print(data )
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data

@st.cache(suppress_st_warning=True)
def load_xinan_data(path="xinan_data.csv"):
    data = pd.read_csv("xinan_data.csv")
    return data


# data = load_data(100000)
# xinan_data = load_xinan_data()

# rainfall = st.slider("降雨", 0.0, 50.0,step=0.1)
rainfall = st.text_input(r'降雨',value ='0.0',type='default')

try:
    float_rainfall = float(rainfall)
except:
    st.write('请输入整数')
    float_rainfall=0.0

datatime_pre= st.slider("时间", 0, 20,step=4)
# 选择显示指标
index_option = st.selectbox(
    '选择显示指标',
     ('氨氮','总磷','溶解氧','高锰酸盐','高锰酸钾'))

# 选择显示站点
# site_options = st.multiselect(
#     '站点名称',
#      ['广三高速断面','凤岗断面','科勒大道断面','芦苞涌入西南涌','鲁岗涌','芦苞涌古云桥'],
#      ['鲁岗涌'])

print(index_option)
print(rainfall)


##############模型预测部分
# 0-2      0-0.3     2-10       0-15
index_name_dict={'氨氮':0,'总磷':1,'溶解氧':2,'高锰酸盐':3,'高锰酸钾':4}
site_name_dict = {'鲁岗涌':0,'芦苞涌古云桥断面':1,  '科勒大道断面':2, '凤岗断面':3,'芦苞涌入西南涌断面':4, '广三高速断面':5}
zb_list=['nh3n_avg','tp_avg','do_avg','codmn_avg']
begin_time = '2019-08-01 00:00:00'
end_time= '2020-05-25 20:00:00'

#data = pd.read_excel('4.26气象数据合并rainfall_pre.xlsx',sheet_name='鲁岗涌')
# data = data[['data_time','mn','总磷','氨氮','水温','溶解氧','高锰酸盐指数']]
# data[['总磷','氨氮','水温','溶解氧','高锰酸盐指数']] = data[['总磷','氨氮','水温','溶解氧','高锰酸盐指数']].astype(float)
# data['mn'] = data['mn'].astype(str)
# data['mn']=data['mn'].map(lambda x:'0'+str(x) if len(str(x))==13 else x)
def query(conn, sql):
    connect = conn
    cur = connect.cursor()
    cur.execute(sql)
    index = cur.description
    result = []
    for res in cur.fetchall():
        row = {}
        for i in range(len(index)):
            row[index[i][0]] = res[i]
        result.append(row)
        connect.close()
    return result

# @st.cache(persist=True)
def get_data():
    conn = psycopg2.connect(database="create_dw",
                        user="ds_usr",
                        password="QNWZWIPN6**F",
                        host="47.106.73.123",
                        port='5432')
    SQL = "SELECT data_time,site_name,nh3n_avg,tp_avg,do_avg,codcr_avg,codmn_avg,rainfall FROM ds_wdp.wdp_rpt_site_mon_data_list where site_name in ('鲁岗涌','芦苞涌古云桥断面','芦苞涌入西南涌断面','科勒大道断面','凤岗断面','广三高速断面') and data_time between '"+begin_time+"' and '"+end_time + "'"
    rows = query(conn = conn,sql = SQL)
    data= pd.DataFrame(rows)
    data[['nh3n_avg','tp_avg','do_avg','codcr_avg','codmn_avg','rainfall']]=data[['nh3n_avg','tp_avg','do_avg','codcr_avg','codmn_avg','rainfall']].astype(float)
    data['data_time'] = pd.to_datetime(data['data_time'])
    data = data.sort_values(by='data_time')

    data['next_rain'] = data['rainfall'][1:].tolist()+[0.0]
    data.head()
    # data.to_csv('xinan_data.csv')
    return data

data = get_data()
def create_dataset(dataset,look_back,bc):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-bc+1):
        x = dataset[i:i+look_back,]#全部特征参与建模
        dataX.append(x)
        y = dataset[i+look_back:i+look_back+bc,0]#只抽取第一个氨氮
        dataY.append(y)
    return np.array(dataX),np.array(dataY)

# 加载模型
@st.cache(suppress_st_warning=True)
def load_predict_model(model_list=None):
    model_dict = {}
    temp_dict = {}
    for site in ['鲁岗涌','芦苞涌古云桥断面','科勒大道断面', '凤岗断面','芦苞涌入西南涌断面', '广三高速断面']:
        if site in ['芦苞涌古云桥断面', '芦苞涌入西南涌断面', '科勒大道断面', '凤岗断面', '广三高速断面']:
            zb_list = ['nh3n_avg', 'tp_avg', 'do_avg', 'codcr_avg']
        else:
            zb_list = ['nh3n_avg', 'tp_avg', 'do_avg', 'codmn_avg']
        for zb in zb_list:
            model_path = 'lstm_model/lstm_%s_%s.h5'%(site,zb)
            model = load_model(model_path)
            scale_model = joblib.load('lstm_model/scaler_%s_%s'%(site,zb))
            print('model_path,scale_model',model_path,scale_model)
            temp_dict[zb]=[model,scale_model]
        model_dict[site] = temp_dict
    return model_dict


def pre_site(data=None,rain_now=None):
    if data is None:
        data=get_data()
    df1 = pd.DataFrame(columns=['date_time', 'site_name', 'nh3n_avg', 'tp_avg', 'do_avg', 'codmn_avg', 'codcr_avg'])
    for site in tqdm(data['site_name'].unique()):
        print(site)
        site_data = data[data['site_name'] == site]
        fill_date = pd.date_range(begin_time, end_time, freq='4H')
        D_fill_date = pd.DataFrame(fill_date, columns=['data_time'])
        f_site_data = pd.merge(D_fill_date, site_data, on='data_time', how='left')
        f_site_data['site_name'] = site
        f_site_data = f_site_data.fillna(method='ffill')
        test_time = f_site_data[f_site_data['data_time'] >= '2020-05-22 00:00:00'][['data_time']]
        # site_data = site_data.set_index('data_time').resample('4H').asfreq().fillna(method='ffill').fillna(method='bfill').reset_index()
        if site in ['芦苞涌古云桥断面', '芦苞涌入西南涌断面', '科勒大道断面', '凤岗断面', '广三高速断面']:
            zb_list = ['nh3n_avg', 'tp_avg', 'do_avg', 'codcr_avg']
        else:
            zb_list = ['nh3n_avg', 'tp_avg', 'do_avg', 'codmn_avg']

        #     site_data = data[data['site_name']=='鲁岗涌']
        #     site_data = site_data.set_index('data_time').resample('4H').asfreq().fillna(method='ffill').fillna(method='bfill').reset_index()

        y2 = test_time.iloc[-1].values[0]
        y3 = pd.to_datetime(y2)
        date = []
        for i in range(6):
            y3 += timedelta(hours=4)
            date.append(y3)
        pre_data = {}
        for zb in zb_list:
            model2 = load_model('lstm_model/lstm_%s_%s.h5' % (site, zb))
            scale_model = joblib.load('lstm_model/scaler_%s_%s' % (site, zb))
            test_data = f_site_data[f_site_data['data_time'] >= '2020-05-22 00:00:00'][[zb, 'next_rain']]
            test_data = np.array(test_data)
            an_test_data = test_data[:, 0].reshape(-1, 1)
            rain_test_data = test_data[:, 1:].reshape(-1, 1)
            rain_test_data[-1] = rain_now
            an_test = scale_model.transform(an_test_data).reshape(-1, 1)
            test_data = np.hstack([an_test, rain_test_data])
            x_test, y_test = create_dataset(test_data, 18, 6)

            y_test_predict = model2.predict(x_test)
            # y_test = scale_model.inverse_transform(y_test)
            y_test_predict = scale_model.inverse_transform(y_test_predict)

            pre_data[zb] = y_test_predict[-1]
        data_p = pd.DataFrame(pre_data)
        data_p['site_name'] = site
        data_p['date_time'] = date
        df1 = pd.concat([df1, data_p], axis=0)

    return df1


def pre_site_without_load_model(data=None,model_dict=None,rain_now=None):
    if data is None:
        data=get_data()
        # data = load_xinan_data(path="xinan_data.csv")
    df1 = pd.DataFrame(columns=['date_time', 'site_name', 'nh3n_avg', 'tp_avg', 'do_avg', 'codmn_avg', 'codcr_avg'])
    for site in tqdm(data['site_name'].unique()):
        print(site)
        site_data = data[data['site_name'] == site]
        fill_date = pd.date_range(begin_time, end_time, freq='4H')
        D_fill_date = pd.DataFrame(fill_date, columns=['data_time'])
        f_site_data = pd.merge(D_fill_date, site_data, on='data_time', how='left')
        f_site_data['site_name'] = site
        f_site_data = f_site_data.fillna(method='ffill')
        test_time = f_site_data[f_site_data['data_time'] >= '2020-05-22 00:00:00'][['data_time']]
        # site_data = site_data.set_index('data_time').resample('4H').asfreq().fillna(method='ffill').fillna(method='bfill').reset_index()
        if site in ['芦苞涌古云桥断面', '芦苞涌入西南涌断面', '科勒大道断面', '凤岗断面', '广三高速断面']:
            zb_list = ['nh3n_avg', 'tp_avg', 'do_avg', 'codcr_avg']
        else:
            zb_list = ['nh3n_avg', 'tp_avg', 'do_avg', 'codmn_avg']
        y2 = test_time.iloc[-1].values[0]
        y3 = pd.to_datetime(y2)
        date = []
        for i in range(6):
            y3 += timedelta(hours=4)
            date.append(y3)
        pre_data = {}
        for zb in zb_list:
            model2 = model_dict[site][zb][0]
            scale_model = model_dict[site][zb][1]
            test_data = f_site_data[f_site_data['data_time'] >= '2020-05-22 00:00:00'][[zb, 'next_rain']]
            test_data = np.array(test_data)
            an_test_data = test_data[:, 0].reshape(-1, 1)
            rain_test_data = test_data[:, 1:].reshape(-1, 1)
            rain_test_data[-1] = rain_now
            an_test = scale_model.transform(an_test_data).reshape(-1, 1)
            test_data = np.hstack([an_test, rain_test_data])
            x_test, y_test = create_dataset(test_data, 18, 6)

            y_test_predict = model2.predict(x_test)
            # y_test = scale_model.inverse_transform(y_test)
            y_test_predict = scale_model.inverse_transform(y_test_predict)

            pre_data[zb] = y_test_predict[-1]
        data_p = pd.DataFrame(pre_data)
        data_p['site_name'] = site
        data_p['date_time'] = date
        df1 = pd.concat([df1, data_p], axis=0)

    return df1


# 通过缓存读取模型
model_dict = load_predict_model(model_list=1)
print('model_dict',model_dict)
pre_data = pre_site_without_load_model(model_dict=model_dict,rain_now=float_rainfall)
# 每次预测读取模型
# pre_data = pre_site(rain_now=rainfall)
print('pre_data',pre_data)
# print(type(pre_data))
list_pre_data = pre_data.values.tolist()
print('list_pre_data',list_pre_data,type(list_pre_data))

st.subheader("站点可视化")

# with open('./other/xinan_baidu.json','r') as f:
#     river_points= json.load(f)

# 站点坐标与显示文字坐标
site_points =[[177, 618],  [313, 511], [490, 437],  [612, 423], [821, 325], [441, 143]]
sitetext_points =[[180, 630],  [313, 530], [490, 460],  [612, 435], [840, 300], [460, 115]]
middle_point = [[245,561],[420,470],[550,447],[724,383],[566,290]]
point_name = ['广三高速断面','凤岗断面','科勒大道断面','芦苞涌入西南涌','鲁岗涌','芦苞涌古云桥']

# # 显示站点字体
fontpath = "./simsun.ttc" # <== 这里是宋体字体路径  存在放jing_vision/utils下
font = ImageFont.truetype(fontpath, 25)
img= cv2.imread('./other/xinan_baidu.png')
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
for i ,point in enumerate(sitetext_points):
    draw.text(point,  point_name[i], font = font, fill = (80, 1, 1, 1))
img = np.array(img_pil)

# 显示河道 红色215,48,39  165,0,38     绿色  102,189,99   0,104,55
red = (165,0,38)
green = (102,189,99)
# 设置根据预测数据显示颜色
assert index_name_dict[index_option] in [0,1,2,3,4],print('预测指标不在指定范围')
datatime_pre_index = int(datatime_pre/4)
#
color_percentage_lists=[]
color_lists = []

if index_name_dict[index_option]==0:  # 选择氨氮显示
    # 循环添加6个站点颜色比i
    for i in range(6):
        color_percentage = min(list_pre_data[6*i+datatime_pre_index][2],2)/2
        color_percentage_lists.append(color_percentage)
    for i in range(6):
        color = (green[2] - int(abs(red[2] - green[2]) * color_percentage_lists[i]),
                 green[1]  - int(abs(red[1] - green[1]) *color_percentage_lists[i]),
               green[0] + int(abs(red[0] - green[0]) * color_percentage_lists[i]),)  # 颜色设置
        color_lists.append(color)

elif index_name_dict[index_option]==1:# 选择总磷显示
    for i in range(6):
        color_percentage = min(list_pre_data[6*i+datatime_pre_index][3],0.3)/0.3
        color_percentage_lists.append(color_percentage)
    for i in range(6):
        color = (green[2] - int(abs(red[2] - green[2]) * color_percentage_lists[i]),
                 green[1]  - int(abs(red[1] - green[1]) *color_percentage_lists[i]),
               green[0] + int(abs(red[0] - green[0]) * color_percentage_lists[i]),)  # 颜色设置
        color_lists.append(color)

elif index_name_dict[index_option]==2:# 选择溶解氧显示
    for i in range(6):
        color_percentage = max(10-list_pre_data[6*i+datatime_pre_index][4],0)/10
        color_percentage_lists.append(color_percentage)
    for i in range(6):
        color = (green[2] - int(abs(red[2] - green[2]) * color_percentage_lists[i]),
                 green[1]  - int(abs(red[1] - green[1]) *color_percentage_lists[i]),
               green[0] + int(abs(red[0] - green[0]) * color_percentage_lists[i]),)  # 颜色设置
        color_lists.append(color)

elif index_name_dict[index_option]==3 :# 选择高锰酸盐显示
    for i in range(1):
        color_percentage = min(list_pre_data[6*i+datatime_pre_index][5],15)/15
        color_percentage_lists.append(color_percentage)
    for i in range(1):
        color = (green[2] - int(abs(red[2] - green[2]) * color_percentage_lists[i]),
                 green[1]  - int(abs(red[1] - green[1]) *color_percentage_lists[i]),
               green[0] + int(abs(red[0] - green[0]) * color_percentage_lists[i]),)  # 颜色设置
        color_lists.append(color)

elif index_name_dict[index_option]==4 :# 选择高锰酸钾显示
    for i in range(1,6):
        color_percentage = min(list_pre_data[6*i+datatime_pre_index][6],30)/30
        color_percentage_lists.append(color_percentage)
    for i in range(5):
        color = (green[2] - int(abs(red[2] - green[2]) * color_percentage_lists[i]),
                 green[1]  - int(abs(red[1] - green[1]) *color_percentage_lists[i]),
               green[0] + int(abs(red[0] - green[0]) * color_percentage_lists[i]),)  # 颜色设置
        color_lists.append(color)
print('color_percentage_lists:',color_percentage_lists)
# green_percentage = min(list_pre_data[0][0],2)
# red_percentage = min(list_pre_data[0][0],2)
print('color_lists',color_lists,len(color_lists))
# 显示河道坐标点
print(index_name_dict[index_option] == 0,type(index_option),index_option)
# color_lists 顺序表 ['鲁岗涌', '芦苞涌古云桥', '科勒大道断面', '凤岗断面', '芦苞涌入西南涌','广三高速断面']
# 判断绘制区域
# if index_name_dict[index_option] in [0,1,2]:
#     i=0
#     for site in ['鲁岗涌', '芦苞涌古云桥', '科勒大道断面', '凤岗断面', '芦苞涌入西南涌','广三高速断面']:
#         json_file_path = './other/%s.json' % (site)
#         print('json_file_path', json_file_path)
#         with open(json_file_path, 'r') as f:
#             points = json.load(f)
#
#         for point in points:
#             cv2.circle(img, (point[0], point[1]), 5, color_lists[i], -1)
#         i=i+1
#
# elif index_name_dict[index_option] == 3:
#     i = 0
#     for site in ['鲁岗涌']:
#         json_file_path = './other/%s.json'%(site)
#         print('json_file_path',json_file_path)
#         with open(json_file_path,'r') as f:
#             points = json.load(f)
#         for point in points :
#             cv2.circle(img, (point[0], point[1]), 5, color_lists[i], -1)
#
# elif index_name_dict[index_option] == 4:
#     i = 0
#     for site in ['芦苞涌古云桥', '科勒大道断面', '凤岗断面', '芦苞涌入西南涌','广三高速断面']:
#         json_file_path = './other/%s.json' % (site)
#         print('json_file_path', json_file_path)
#         with open(json_file_path, 'r') as f:
#             points = json.load(f)
#         for point in points:
#             cv2.circle(img, (point[0], point[1]), 5, color_lists[i], -1)
#         i=i+1

"""
渐变色绘制
"""
# 绘制河面断面名称
# 氨氮，总磷，溶解氧
if index_name_dict[index_option] in [0,1,2]:
    river_points_json_file_name = ['_广三高速断面','广三高速_凤岗','凤岗_科勒大道','科勒大道_芦苞涌入西南涌','芦苞涌入西南涌_鲁岗涌','鲁岗涌_','_芦苞涌古云桥','芦苞涌古云桥_芦苞涌入西南涌']
    for i,name  in enumerate(river_points_json_file_name):
        json_file_path = './other/%s.json' % (name)
        print('json_file_path', json_file_path)
        with open(json_file_path, 'r') as f:
            points = json.load(f)
            # 只有一个检测点绘制单一颜色
        if i==0:
            color = color_lists[5]
            for point in points:
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i==5:
            color = color_lists[0]
            for point in points:
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i==6:
            color = color_lists[1]
            for point in points:
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        # 有两个监测点间断取色 （注意取色顺序与上面存颜色顺序对应)
        elif i==1:
            length = len(points)
            color_start_b,color_start_g,color_start_r =color_lists[5]
            color_end_b,color_end_g,color_end_r =color_lists[3]
            for j,point in enumerate(points):
                color = (int(color_start_b+j*(color_end_b-color_start_b)/length),
                         int(color_start_g+j*(color_end_g-color_start_g)/length),
                         int(color_start_r+j*(color_end_r-color_start_r)/length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i==2:
            length = len(points)
            color_start_b,color_start_g,color_start_r =color_lists[3]
            color_end_b,color_end_g,color_end_r =color_lists[2]
            for j,point in enumerate(points):
                color = (int(color_start_b+j*(color_end_b-color_start_b)/length),
                         int(color_start_g+j*(color_end_g-color_start_g)/length),
                         int(color_start_r+j*(color_end_r-color_start_r)/length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i==3:
            length = len(points)
            color_start_b,color_start_g,color_start_r =color_lists[2]
            color_end_b,color_end_g,color_end_r =color_lists[4]
            for j,point in enumerate(points):
                color = (int(color_start_b+j*(color_end_b-color_start_b)/length),
                         int(color_start_g+j*(color_end_g-color_start_g)/length),
                         int(color_start_r+j*(color_end_r-color_start_r)/length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i==4:
            length = len(points)
            color_start_b,color_start_g,color_start_r =color_lists[4]
            color_end_b,color_end_g,color_end_r =color_lists[0]
            for j,point in enumerate(points):
                color = (int(color_start_b+j*(color_end_b-color_start_b)/length),
                         int(color_start_g+j*(color_end_g-color_start_g)/length),
                         int(color_start_r+j*(color_end_r-color_start_r)/length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i==7:
            length = len(points)
            color_start_b,color_start_g,color_start_r =color_lists[1]
            color_end_b,color_end_g,color_end_r =color_lists[4]
            for j,point in enumerate(points):
                color = (int(color_start_b+j*(color_end_b-color_start_b)/length),
                         int(color_start_g+j*(color_end_g-color_start_g)/length),
                         int(color_start_r+j*(color_end_r-color_start_r)/length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
# 高锰酸钾情况
elif index_name_dict[index_option] == 3:
    i = 0
    for site in ['鲁岗涌']:
        json_file_path = './other/%s.json'%(site)
        print('json_file_path',json_file_path)
        with open(json_file_path,'r') as f:
            points = json.load(f)
        for point in points :
            cv2.circle(img, (point[0], point[1]), 5, color_lists[i], -1)
# 为高锰酸盐情况
elif index_name_dict[index_option] == 4:
    river_points_json_file_name = ['_广三高速断面', '广三高速_凤岗', '凤岗_科勒大道', '科勒大道_芦苞涌入西南涌', '_芦苞涌古云桥',
                                   '芦苞涌古云桥_芦苞涌入西南涌']
    for i, name in enumerate(river_points_json_file_name):
        json_file_path = './other/%s.json' % (name)
        print('json_file_path', json_file_path)
        with open(json_file_path, 'r') as f:
            points = json.load(f)
        # 只有一个边有检测站点
        if i == 0:
            color = color_lists[4]
            for point in points:
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i == 4:
            color = color_lists[0]
            for point in points:
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i == 1:
            length = len(points)
            color_start_b, color_start_g, color_start_r = color_lists[4]
            color_end_b, color_end_g, color_end_r = color_lists[2]
            for j, point in enumerate(points):
                color = (int(color_start_b + j * (color_end_b - color_start_b) / length),
                         int(color_start_g + j * (color_end_g - color_start_g) / length),
                         int(color_start_r + j * (color_end_r - color_start_r) / length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i == 2:
            length = len(points)
            color_start_b, color_start_g, color_start_r = color_lists[2]
            color_end_b, color_end_g, color_end_r = color_lists[1]
            for j, point in enumerate(points):
                color = (int(color_start_b + j * (color_end_b - color_start_b) / length),
                         int(color_start_g + j * (color_end_g - color_start_g) / length),
                         int(color_start_r + j * (color_end_r - color_start_r) / length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i == 3:
            length = len(points)
            color_start_b, color_start_g, color_start_r = color_lists[1]
            color_end_b, color_end_g, color_end_r = color_lists[3]
            for j, point in enumerate(points):
                color = (int(color_start_b + j * (color_end_b - color_start_b) / length),
                         int(color_start_g + j * (color_end_g - color_start_g) / length),
                         int(color_start_r + j * (color_end_r - color_start_r) / length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)
        elif i == 5:
            length = len(points)
            color_start_b, color_start_g, color_start_r = color_lists[0]
            color_end_b, color_end_g, color_end_r = color_lists[3]
            for j, point in enumerate(points):
                color = (int(color_start_b + j * (color_end_b - color_start_b) / length),
                         int(color_start_g + j * (color_end_g - color_start_g) / length),
                         int(color_start_r + j * (color_end_r - color_start_r) / length))
                cv2.circle(img, (point[0], point[1]), 5, color, -1)

# 绘制站点坐标点
for point in site_points :
    cv2.circle(img, (point[0], point[1]), 10, (255, 105, 65), -1)

cv2.imwrite('draw.png',img)
img = img[:,:,[2,1,0]]
st.image(img.astype(np.uint8))

st.subheader("其他参数折线图" )
# data1 = pd.DataFrame({'label': ['Feb', 'Jan', 'Mar', 'Apr', 'May', 'Aug', 'Seb'],
#                       'V1': [5, 8, 4.3, 7, 8.2, 4.6, 8],
#                       'V2': [6, 7, 8.2, 6, 4.5, 7, 8.6],
#                       'V3': [7, 8, 7.2, 4, 3.5, 8, 8.9],
#                       'V4': [5, 5, 8.2, 6, 4.5, 6, 9.6],
                      # 'V5': [6, 7, 4.2, 7, 7.5, 7, 8.6],
                      # 'V6': [4, 8, 8.6, 6, 4.5, 5, 6.6],
                      # 'V7': [6, 7, 3.2, 9, 5.5, 8, 4.6],
                      # 'V8': [6, 4, 8.2, 6, 6.5, 7, 7.6],
                      # 'V9': [6, 7, 3.2, 9, 5.5, 8, 4.6]
                      # })
# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['氨', '氮', '含氧量'])

st.line_chart(pre_data)


if st.checkbox("Show raw data", False):
    # st.subheader("Raw data by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
    st.write(pre_data)
