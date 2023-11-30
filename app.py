# index 화면에서 파일 선택 후 예측 버튼을 클릭하면
# 어떤 이미지 인지 분류하여 화면에 출력
import pandas as pd
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
# 모델 불러오기
model = tf.keras.models.load_model('./music/archive/EfficientNetB3-instruments-99.33.h5')

import pandas as pd
df = pd.read_csv('./music/archive/class_dict.csv') # 두번째 열 뽑기
second_column = df.iloc[:, 1].values  # 두 번째 열 선택
labels = second_column

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])

def upload_file():
    # 전송  받은 파일이 이미지 파일 형태가 아니면 400 리턴
    if 'image' not in request.files:
        return jsonify({'error': 'no file upload'}), 400
    f = request.files['image']
    if f.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    image = Image.open(f)
    image = image.resize((224, 224))
    image = image.convert('RGB')
    image_array = np.array(image)

    # resahpe 함수와 같이
    image_array = image_array.reshape(1, *image_array.shape)

    pred = model.predict(image_array)
    pred_class = np.argmax(pred, axis=-1)[0]

    # 1순위와 2순위를 가져오기
    sorted_indices = np.argsort(pred)[0][::-1][:2]
    top_labels = [labels[i] for i in sorted_indices]
    top_probabilities = [float(pred[0][i]) for i in sorted_indices]

    # 1순위와 2순위의 확률을 얻기
    top1_percent = round(top_probabilities[0] * 100, 2)
    top2_percent = round(top_probabilities[1] * 100, 2)

    return jsonify({'prediction': labels[pred_class],
                    'top1': top1_percent,
                    'top2': top2_percent,
                    'top_1_prediction': top_labels[0],
                    'top_2_prediction': top_labels[1]
})
if __name__ == '__main__':
    app.run(debug=True, port=5050,host='192.168.0.25')
