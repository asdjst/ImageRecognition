import numpy as np
from PIL import Image
import os
from tensorflow import keras

# 加载预训练的模型
try:
    model = keras.models.load_model('model.h5')
except Exception as e:
    print(f"加载模型失败: {e}")
    exit(1)

# 定义图像处理和预测函数
def predict_digit(image_path):
    try:
        # 加载图像并转换为灰度
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))  # 确保图像尺寸为 28x28
        img = np.array(img, dtype=np.float32)  # 转换为 numpy 数组，指定数据类型为 float32

        # 正则化图像数据
        img = img / 255.0

        # 调整图像形状
        img = img.reshape((1, 28, 28, 1))

        # 使用模型进行预测
        pred = model.predict(img)
        digit = np.argmax(pred)
        return digit
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None

# 处理 img 文件夹中的所有 PNG 文件
if not os.path.exists('img'):
    print("未找到 'img' 文件夹，请确保包含 PNG 图像的文件夹名为 'img'")
    exit(1)

for file_name in os.listdir('img'):
    if file_name.endswith(".png"):
        file_path = os.path.join('img', file_name)
        digit = predict_digit(file_path)
        if digit is not None:
            print(f"文件 {file_name} 识别的数字是: {digit}")
        else:
            print(f"文件 {file_name} 无法识别")