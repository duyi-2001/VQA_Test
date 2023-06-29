import json
import os

import requests
from flask import Flask, jsonify, request
import torch
from torchvision import transforms
import skvideo.io
from PIL import Image
import numpy as np
from VSFA import VSFA
from CNNfeatures import get_features
from argparse import ArgumentParser
import time

app = Flask(__name__)


def get_file(file_url):
    # 临时文件夹
    if (not os.path.exists('temp')):
        os.mkdir('temp')
    filename = 'temp/' + file_url.split('/')[-1]

    # 从通过http获取图片
    proxies = {"http": None, "https": None}
    req = requests.get(file_url, verify=False, proxies=proxies)
    # 保存文件
    with open(filename, 'wb') as f:
        f.write(req.content)
    return filename


@app.route('/inference', methods=['POST'])
def inference():
    params = request.get_json()
    file_url = params['fileUrl']
    # 把视频文件下载到本地临时文件夹
    # 目录存在/创建
    # 下载文件
    # ToDo 返回的是一个文件在本地的路径

    # 视频路径
    video_path = get_file(file_url)

    # 推理过程

    model_path = 'models/VSFA.pt'
    video_format = 'RGB'  # video format: RGB or YUV420
    video_width = None
    video_height = None
    frame_batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    # data preparation
    assert video_format == 'YUV420' or video_format == 'RGB'
    if video_format == 'YUV420':
        video_data = skvideo.io.vread(video_path, video_height, video_width, inputdict={'-pix_fmt': 'yuvj420p'})
    else:
        video_data = skvideo.io.vread(video_path)

    video_length = video_data.shape[0]
    video_channel = video_data.shape[3]
    video_height = video_data.shape[1]
    video_width = video_data.shape[2]
    transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for frame_idx in range(video_length):
        frame = video_data[frame_idx]
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_video[frame_idx] = frame

    print('Video length: {}'.format(transformed_video.shape[0]))

    # feature extraction
    features = get_features(transformed_video, frame_batch_size=frame_batch_size, device=device)
    features = torch.unsqueeze(features, 0)  # batch size 1

    # quality prediction using VSFA
    model = VSFA()
    model.load_state_dict(torch.load(model_path))  #
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_length = features.shape[1] * torch.ones(1, 1)
        outputs = model(features, input_length)
        y_pred = outputs[0][0].to('cpu').numpy()
        print("Predicted quality: {}".format(y_pred))

    end = time.time()

    # print('Time: {} s'.format())

    # 结果转换成json格式

    print('Time: {} s'.format(end - start))
    print('Predicted quality: {}'.format(y_pred))

    # 将y_pred转换成float类型并保留两位小数
    y_pred = float(y_pred)
    y_pred = "{:.2f}".format(y_pred * 100)

    # 计算视频长度并保留两位小数
    length = end - start
    length = "{:.2f}".format(length)

    result = {'length': length, 'score': y_pred}
    return json.dumps(result)


if __name__ == '__main__':
    app.run()
