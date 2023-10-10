import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
def compress_image(image_path, output_path, max_size= 100*1024*1024):
    # 读取图像
    image = Image.open(image_path)
    
    # 设置初始图像质量
    quality = 100
    # 循环直到图像大小小于200KB
    while True:
        # 将图像保存为JPEG格式
        image.save(output_path, format="JPEG", quality=quality)

        # 计算输出图像的大小
        output_size = os.path.getsize(output_path)
        # 如果图像大小小于200KB，则退出循环
        if output_size <= max_size:
            break

        # 否则降低图像质量并继续循环
        quality -= 5

# 将图像从原始路径读取并压缩到输出路径
compress_image(r"./test/eso1719a.tif", "output.jpg")

