from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import os
import re
from feature import features
 
def proceed():
    f = features()
    f.filePath = '../public/fragments/fragment.png'
    f.getImg()
    f.strengthen()
    # f.getSIFT()

def auto_save_file(path):
    directory, file_name = os.path.split(path)
    while os.path.isfile(path):
        pattern = '\d+\.'
        if re.search(pattern, file_name) is None:
            file_name = file_name.replace('.', '0.')
            # print(file_name)
        else:
            # print(re.findall(pattern, file_name)[-1][0])
            current_number = int(re.findall(pattern, file_name)[-1][0])
            new_number = current_number + 1
            file_name = file_name.replace(f'{current_number}.', f'{new_number}.')
        path = os.path.join(directory + os.sep + file_name)
        # print(path)
    print(path)
    return path

def get_imgstream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        # print(img_stream)
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/')
def sayHello():
    return 'Hello Python'

@app.route('/getProceed', methods=['POST'])
def getProceed():
    try:
        data = request.get_json()
        print(data)
        file_name = data.get('fileName')
        sname = '0.jpg'
        f = features()
        f.filePath = '../public/fragments/fragment.png'
        f.getImg()

        if file_name == 'SIFT':
            print('SIFT')
            f.getSIFT()
        elif file_name == 'HOG':
            print('HOG')
            f.getHOG()
        elif file_name == 'Sobel':
            print('Sobel')
            f.strengthen(10)
            print('done')
        elif file_name == 'Scarr':
            print('scharr')
            f.getScharr()
        elif file_name =='Gradient':
            print('Gradient')
            f.getGradient()
        elif file_name == 'K-means':
            print('K-Means')
            f.color_kmeans()
            print('K-Means Done')
        elif file_name == 'Color-Hist':
            f.colorHist()
        
        img_stream = get_imgstream(sname)
        # print(img_stream)
        return jsonify({'img_stream': img_stream})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/getFragment', methods=['POST'])
def get_fragment():
    try:
        data = request.get_json()
        base64_image = data.get('image')
        # print(base64_image)
        if base64_image:
            # 解码 Base64 图像数据
            image_data = base64.b64decode(base64_image.split(',')[1])
            # 在这里处理接收到的图像数据，可以保存到服务器或进行其他操作
            # 例如，保存图像并返回成功响应
            with open('../public/fragments/fragment.png', 'wb') as f:
                # f.write(image_data)
                print("fragment written:",f.write(image_data))
            with open(auto_save_file('../public/fragments/fragment0.png'), 'wb') as f:
                # f.write(image_data)
                print("fragment0 written:",f.write(image_data))
                # proceed()
            return jsonify({'message': '上传成功'})
        else:
            return jsonify({'error': '没有上传图像'})
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)