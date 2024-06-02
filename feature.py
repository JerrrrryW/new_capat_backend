import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

from HOG import *
from sklearn.cluster import KMeans
from PIL import Image
from pywt import dwt2, idwt2
from matplotlib import font_manager
import os
import re
import argparse
 

class features:

    def __init__(self, inputPath, outputPath):
        self.filePath = inputPath
        self.savePath = outputPath
        # self.savePath = 'public/fragments/proceed0.png'

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

    # 自动获取存储路径
    def getSavePath(self,an):
        savePath = self.savePath
        directory, file_name = os.path.split(savePath)
        while os.path.isfile(savePath):
            pattern = '\d+\.'
            if re.search(pattern, file_name) is None:
                file_name = file_name.replace('.', '0.')
                # print(file_name)
            else:
                # print(re.findall(pattern, file_name)[-1][0])
                current_number = int(re.findall(pattern, file_name)[-1][0])
                new_number = current_number + 1
                file_name = file_name.replace(f'{current_number}.', f'{new_number}.')
            savePath = os.path.join(directory + os.sep + file_name)
        current_number = int(re.findall('\d+\.', file_name))[-1][0]
        file_name = file_name.replace(f'{current_number}.', f'{current_number}' + an + '.')
        savePath = os.path.join(directory + os.sep + file_name)
            # print(path)
        return savePath

    def getImg(self):
        self.img = cv.imread(self.filePath)
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        print("The shape of original image: ", self.img.shape)
    
    # sobel and gradient
    # for 
    def strengthen(self, alpha = 10):
        alpha = alpha / 10

        # Read the input image
        moon_f = self.img

        # Convert the image to grayscale
        # moon_f = cv.cvtColor(moon_f, cv2.COLOR_BGR2GRAY)

        # Compute the gradient of the image
        gx = cv.convertScaleAbs(cv.Sobel(moon_f, cv.CV_64F, 1, 0, ksize=3))
        gy = cv.convertScaleAbs(cv.Sobel(moon_f, cv.CV_64F, 0, 1, ksize=3))
        gs = cv.addWeighted(gx, 0.5, gy, 0.5, 0)
        gradient = cv.convertScaleAbs(gx) + cv.convertScaleAbs(gy)

        # plt.subplot(2,2,1),plt.imshow(self.img)
        # plt.title('Original'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,2,2),plt.imshow(gx)
        # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,2,3),plt.imshow(gy)
        # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,2,4),plt.imshow(gs)
        # plt.title('Sobel s'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # cv.imwrite('0.jpg', gs)
        return gs
        # cv.imshow('gradient', gradient)
        # cv.imshow('sharp', sharp)
        # cv.waitKey(0)

        # return qpixmap

    def getGradient(self, alpha = 10):
        alpha = alpha / 10

        # Read the input image
        moon_f = self.img

        # Convert the image to grayscale
        # moon_f = cv.cvtColor(moon_f, cv2.COLOR_BGR2GRAY)

        # Compute the gradient of the image
        gx = cv.convertScaleAbs(cv.Sobel(moon_f, cv.CV_64F, 1, 0, ksize=3))
        gy = cv.convertScaleAbs(cv.Sobel(moon_f, cv.CV_64F, 0, 1, ksize=3))
        gs = cv.addWeighted(gx, 0.5, gy, 0.5, 0)
        gradient = cv.convertScaleAbs(gx) + cv.convertScaleAbs(gy)

        # Apply the gradient to the image
        # sharp = cv.addWeighted(moon_f, 1, gradient, alpha, 0)
        # sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))

        # # Convert the image to uint8
        # sharp = sharp.astype("uint8")

        # cv.imwrite('0.jpg', gradient)
        return gradient
    # 
    def erode(self, kernel_size_num=3, start_color=(255, 255, 255), end_color=(255, 0, 0),
            num_iterations=12, has_background=True):
        kernel_size = (kernel_size_num, kernel_size_num)
        # 转换图像
        img = self.img

        # 转换为灰度图像
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 应用阈值函数来去除背景
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        if not has_background: cv.bitwise_not(thresh, thresh)
        # cv.imshow('thresh', thresh)

        # 定义腐蚀核
        kernel = np.ones(kernel_size, np.uint8)

        # 腐蚀图像并显示结果
        i = 0
        while np.any(thresh > 0) and i < num_iterations:
            # 计算当前层次对应的颜色
            color = np.array(start_color) + (np.array(end_color) - np.array(start_color)) * i / float(num_iterations)
            color = color.astype(np.uint8)

            thresh = cv.erode(thresh, kernel)
            img[thresh > 0] = color
            i += 1

        # cv.imshow('Eroded Image', img)
        return img        
    
    # scharr
    def getScharr(self):
        scharrx = cv.convertScaleAbs(cv.Scharr(self.img,cv.CV_64F,1,0))
        scharry = cv.convertScaleAbs(cv.Scharr(self.img,cv.CV_64F,0,1))
        scharrs = cv.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
        # plt.subplot(2,2,1),plt.imshow(self.img)
        # plt.title('Original'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,2,2),plt.imshow(self.scharrx)
        # plt.title('Scharr X'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,2,3),plt.imshow(self.scharry)
        # plt.title('Scharr Y'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,2,4),plt.imshow(self.scharrs)
        # plt.title('scharr s'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # cv.imwrite('0.jpg', self.scharrs)
        return scharrs

    #laplas
    def getLaplas(self):
        return cv.Laplacian(self.img, cv.CV_64F)
    
    # SIFT
    # 
    def getSIFT(self):
        sift = cv.xfeatures2d.SIFT_create(nfeatures=1000)
        img1 = cv.imread(self.filePath)
        img2 = cv.imread(self.filePath)
        img3=img2
        #img3=cv.flip(img2,-1)
        #求中心点，对图像进行旋转
        #(h,w)=img2.shape[:2]
        #center=(w//2,h//2)
        #M=cv.getRotationMatrix2D(center,30,1.0)
        #img3=cv.warpAffine(img2,M,(w,h))
        #灰度化
        img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        img3_gray = cv.cvtColor(img3, cv.COLOR_RGB2GRAY)
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img3_gray, None)
        #绘制特征点图
        img1t=cv.drawKeypoints(img1_gray,kp1,img1)
        img3t=cv.drawKeypoints(img3_gray,kp2,img3)
        #进行KNN特征匹配，K设置为2
        start=time.time()
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good=[]
        print(len(matches))
        matchesMask = [[0, 0] for i in range(len(matches))]
        for i, (m1, m2) in enumerate(matches):
            if m1.distance < 0.7* m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
                good.append(m1)
                matchesMask[i]=[1,0]
                pt1 = kp1[m1.queryIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
                pt2 = kp2[m1.trainIdx].pt  # trainIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
                #print(kpts1)
                print(i, pt1, pt2)   #打印匹配点个数，并标出两图中的坐标位置
                #画特征点及其周围的圆圈
                cv.circle(img1, (int(pt1[0]), int(pt1[1])), 5, (0, 255, 0), -1)
                num = "{}".format(i)
                cv.putText(img1, num, (int(pt1[0]), int(pt1[1])),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.circle(img3, (int(pt2[0]), int(pt2[1])), 5, (0, 255, 0), -1)
                cv.putText(img3, num, (int(pt2[0]), int(pt2[1])),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        end=time.time()
        print("good match num:{} good match points:".format(len(good)))
        print("number of feature points:", len(kp1), len(kp2))
        #匹配连线
        draw_params = dict(matchColor=(255, 0, 0),
                        singlePointColor=(0, 0, 255),
                        matchesMask=matchesMask,
                        flags=0)

        res = cv.drawMatchesKnn(img1, kp1, img3, kp2, matches, None,**draw_params)


        print("运行时间:%.2f秒"%(end-start))
        # cv.imwrite("0.jpg",img1)
        return img1



    def DWT(self):
        # 0是表示直接读取灰度图
        img = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)

        # 对img进行haar小波变换：,haar小波
        cA, (cH, cV, cD) = dwt2(img, 'haar')
        #cA, (cH, cV, cD) = dwt2(cA, 'haar')
        #cA, (cH, cV, cD) = dwt2(cA, 'haar')

        # 小波变换之后，低频分量对应的图像：
        a = np.uint8(cA / np.max(cA) * 255)
        # 小波变换之后，水平方向高频分量对应的图像：
        b = np.uint8(cH / np.max(cH) * 255)
        # 小波变换之后，垂直平方向高频分量对应的图像：
        c = np.uint8(cV / np.max(cV) * 255)
        # 小波变换之后，对角线方向高频分量对应的图像：
        d = np.uint8(cD / np.max(cD) * 255)

        # 根据小波系数重构回去的图像
        rimg = idwt2((cA, (cH, cV, cD)), 'haar')

        cv2.namedWindow("result", 0)
        cv2.imshow("result", a)
        cv.waitKey(0)
        # cv2.imwrite('dipin.jpg', a)
        # cv2.imwrite('shuipinggaopin.jpg', b)
        # cv2.imwrite('chuizhigaopin.jpg', c)
        # cv2.imwrite('duijiaogaopin.jpg', d)

        # fontnamelist = font_manager.get_font_names()
        # print(fontnamelist)
        plt.rcParams['font.sans-serif'] = ['Heiti Tc']
        #plt.subplot(231), plt.imshow(img, 'gray'), plt.title('原始图像'), plt.axis('off')
        plt.subplot(221), plt.imshow(a, 'gray'), plt.title('低频分量'), plt.axis('off')
        plt.subplot(222), plt.imshow(b, 'gray'), plt.title('水平方向高频分量'), plt.axis('off')
        plt.subplot(223), plt.imshow(c, 'gray'), plt.title('垂直平方向高频分量'), plt.axis('off')
        plt.subplot(224), plt.imshow(d, 'gray'), plt.title('对角线方向高频分量'), plt.axis('off')
        #plt.subplot(236), plt.imshow(rimg, 'gray'), plt.title('重构图像'), plt.axis('off')
        # plt.savefig('3.new-img.jpg')
        plt.show()


    def calcHist_CV(self, img, color):
            hist= cv.calcHist([img], [0], None, [256], [0.0,255.0])  
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)  
            histImg = np.zeros([256,256,3], np.uint8)  
            hpt = int(0.9* 256);  
            
            for h in range(256):  
                intensity = int(hist[h]*hpt/maxVal)  
                cv2.line(histImg,(h,256), (h,256-intensity), color)  
            return histImg
    
    def colorHist(self):
        colorr = [255, 0, 0]
        colorg = [0, 255, 0]
        colorb = [0, 0, 255]
        img = cv2.resize(self.img,None,fx=0.6,fy=0.6,interpolation = cv2.INTER_CUBIC)
        r, g, b = cv.split(img)
    
        histImgB = self.calcHist_CV(b, [0, 0, 255])  
        histImgG = self.calcHist_CV(g, [0, 255, 0])  
        histImgR = self.calcHist_CV(r, [255, 0, 0])  

        # plt.subplot(2, 4, 1), plt.imshow(histImgB)
        # plt.title('histImgB'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2, 4, 2), plt.imshow(histImgG)
        # plt.title('histImgG'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2, 4, 3), plt.imshow(histImgR)
        # plt.title('histImgR'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2, 4, 4), plt.imshow(img)
        # plt.title('Img'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2, 4, 5), plt.imshow(b)
        # plt.title('ImgB'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2, 4, 6), plt.imshow(g)
        # plt.title('ImgG'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2, 4, 7), plt.imshow(r)
        # plt.title('ImgR'), plt.xticks([]), plt.yticks([])
        
        # plt.show()


    # HOG
    def getHOG(self):
        # Convert the image to grayscale
        gray_image = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        print("The shape of gray image: ", gray_image.shape)
        hog = Hog_descriptor(gray_image, cell_size=8, bin_size=9)
        vector, image = hog.extract()

        # 输出图像的特征向量shape
        # print(np.array(vector).shape)
        # plt.imshow(image, cmap=plt.cm.gray)
        # plt.show()
        # cv.imwrite('0.jpg', image)
        return image

    # color k-means
    def color_kmeans(self):
        km = KMeans(n_clusters = 5)
        f = open(self.filePath, 'rb')
        data = []
        img = Image.open(f).convert("RGB")
        m, n = img.size
        for i in range(m):
            for j in range(n):
                x, y, z = img.getpixel((i, j))
                data.append([x/256.0, y/256.0, z/256.0])
        imgData = np.array(data)
        row, col = m, n
        #聚类获取每个像素所属的类别
        label = km.fit_predict(imgData)
        label = label.reshape([row, col])
        #创建一张新的灰度图保存聚类后的结果
        pic_new = Image.new('L', (row, col))

        #根据所属类别向图片中添加灰度值
        # 最终利用聚类中心点的RGB值替换原图中每一个像素点的值，便得到了最终的分割后的图片
        for i in range(row):
            for j in range(col):
                pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
        #pic_new = cv2.applyColorMap(cv2.convertScaleAbs(pic_new, alpha=-1), cv2.COLORMAP_JET)
        #以JPEG格式保存图片
        return pic_new


def proceed():
    f = features()
    f.getImg()
    f.strengthen()
    f.getSIFT()

def main():
    parser = argparse.ArgumentParser(description="Process an image using various algorithms.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("-o", "--output", default="./processed_images", help="Path to the output directory")
    parser.add_argument("-a", "--algorithm", nargs='+', default=["all"], help="List of algorithms to apply. Options: all, strengthen, erode, scharr, laplas, sift, dwt, hog, kmeans")
    
    args = parser.parse_args()

    f = features(args.input, args.output)

    if "all" in args.algorithm or "strengthen" in args.algorithm:
        f.getImg()
        f.strengthen()
    
    if "all" in args.algorithm or "erode" in args.algorithm:
        f.erode()

    if "all" in args.algorithm or "scharr" in args.algorithm:
        f.getScharr()

    if "all" in args.algorithm or "laplas" in args.algorithm:
        f.getLaplas()

    if "all" in args.algorithm or "sift" in args.algorithm:
        f.getSIFT()

    if "all" in args.algorithm or "dwt" in args.algorithm:
        f.DWT()

    if "all" in args.algorithm or "hog" in args.algorithm:
        f.getHOG()

    if "all" in args.algorithm or "kmeans" in args.algorithm:
        f.color_kmeans()

if __name__ == "__main__":
    main()