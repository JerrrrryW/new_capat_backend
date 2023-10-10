import cv2 as cv
import time

def getSIFT():
    sift = cv.xfeatures2d.SIFT_create(nfeatures=1000)
    img1 = cv.imread("./SealData/train/阿长/阿长1.png")
    img2 = cv.imread("./SealData/train/阿长/阿长1.jpg")
    img3=img2
    # img3=cv.flip(img2,-1)
    # # 求中心点，对图像进行旋转
    # (h,w)=img2.shape[:2]
    # center=(w//2,h//2)
    # M=cv.getRotationMatrix2D(center,30,1.0)
    # img3=cv.warpAffine(img2,M,(w,h))
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
    cv.imshow("img1_gray",img1_gray)
    cv.imshow("img3_gray",img3_gray)
    cv.imshow("Result", res)
    cv.imshow("img1", img1)
    cv.imshow("img3", img3)
    # cv.imwrite("SIFTimg1_gray.jpg",img1_gray)
    # cv.imwrite("SIFTimg3_gray.jpg",img3_gray)
    # cv.imwrite("SIFTimg1.jpg",img1)
    # cv.imwrite("SIFTimg3.jpg",img3)
    # cv.imwrite("SIFTimg1t.jpg",img1t)
    # cv.imwrite("SIFTimg3t.jpg",img3t)
    # cv.imwrite("SIFTResult.jpg",res)
    cv.waitKey(0)
    #cv.destroyAllWindows()

    if len(good) / min(len(kp1), len(kp2)) >= 0.7:
        return True
    return False
print(getSIFT())
# region
# from utils import qtpixmap_to_cvimg, drawOutRectgle, cvImg_to_qtImg

# def drawOutRectgle(cont, img=None, isdrawing=False):
#     # 最小外接正矩形————用于计算轮廓内每个像素灰度值(去除 矩形-外轮廓)
#     cnt = cont
#     st_x, st_y, width, height = cv2.boundingRect(cnt)  # 获取外接正矩形的xy
#     # 对应的四个顶点(0,1,2,3) 0：左上，1：右上，2：右下，3：左下
#     bound_rect = np.array([[[st_x, st_y]], [[st_x + width, st_y]],
#                            [[st_x + width, st_y + height]], [[st_x, st_y + height]]])
#     if isdrawing:
#         cv2.drawContours(img, [bound_rect], -1, (0, 0, 255), 2)  # 绘制最小外接正矩形
#     x_min, x_max, y_min, y_max = st_x, st_x + width, st_y, st_y + height  # 矩形四顶点
#     # 通过每一个最小外接正矩形(四个顶点坐标)，判断矩形内累加坐标像素的灰度值，除去小于阈值的像素(在轮廓外)
#     return x_min, x_max, y_min, y_max



# # if __name__ == "__main__":
# def findStamp(imgInput):
#     # img = cv2.imread(sys.argv[1])
#     img = cv2.imread(imgInput)
#     # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
#     grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 从RGB色彩空间转换到HSV色彩空间
#     grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

#     # H、S、V范围一：
#     lower1 = np.array([0, 43, 46])
#     upper1 = np.array([10, 255, 255])
#     mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
#     res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

#     # H、S、V范围二：(HSV中红色有两个范围)
#     lower2 = np.array([156, 43, 46])
#     upper2 = np.array([180, 255, 255])
#     mask2 = cv2.inRange(grid_HSV, lower2, upper2)
#     res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

#     # 将两个二值图像结果 相加
#     mask3 = mask1 + mask2

#     # 结果显示
#     # cv2.imshow("mask3", mask3)
#     # cv2.imshow("img", img)

#     # cv2.imshow("Mask1", mask1)
#     # cv2.imshow("res1", res1)
#     # cv2.imshow("Mask2", mask2)
#     # cv2.imshow("res2", res2)
#     # cv2.imshow("grid_RGB", grid_RGB[:, :, ::-1])  # imshow()函数传入的变量也要为b g r通道顺序

#     # 开闭运算[先膨胀-后腐蚀]，尝试去除噪声(去除尖端)
#     img2 = mask3.copy()
#     k = np.ones((4, 4), np.uint8)  # 卷积核 如(10, 10)= 10X10的矩阵(或称数组)
#     thresh_open = cv2.morphologyEx(img2, cv2.MORPH_OPEN, k)  # 开运算[先膨胀-后腐蚀]
#     k = np.ones((30, 30), np.uint8)  # 卷积核2
#     thresh_open2 = cv2.morphologyEx(thresh_open, cv2.MORPH_CLOSE, k)  # 闭运算消除空洞
#     # thresh_open2 = thresh_open

#     # cv2.imshow("open operation", thresh_open)  # 暂时屏蔽
#     # cv2.imshow("open operation2", thresh_open2)  # 暂时屏蔽

#     cnts = cv2.findContours(thresh_open2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
#     cnts = imutils.grab_contours(cnts)  # 返回轮廓 contours —— cnts
#     # print (cnts)

#     # (6).cnts 返回的是所有轮廓，所以需要for循环来遍历每一个轮廓
#     savePath = "./SealData"
#     img = img.copy()
#     for i, c in enumerate(cnts):
#         # 计算轮廓区域的图像矩。 在计算机视觉和图像处理中，图像矩通常用于表征图像中对象的形状。
#         # 计算最小外接正矩形的四个顶点，是否绘制外矩形框
#         x_min, x_max, y_min, y_max = drawOutRectgle(c, img, True)
#         img_mini = img[y_min:y_max, x_min:x_max]
#         # cv2.imshow("stamp" + str(i), img_mini)
#         cv2.imwrite(savePath + str(i)+".jpg", img_mini)
#         print(savePath + str(i)+".jpg")

#     # cv2.imshow("img with rectangle", img)
#     # imgOutput = QPixmap(cvImg_to_qtImg(img))


#     # cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # imgInput = './SealData/train/阿长/阿长1.png'
# # findStamp(imgInput)
# endregion
