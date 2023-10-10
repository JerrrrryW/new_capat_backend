import cv2 as cv
import time
import xlsxwriter
import re
import opencc

class textual:
    
    def __init__(self):
        self.workbook_file = '/Users/gritty/Desktop/workspace/CAPAT_DB/capat/test/20230906_文科实验室用石涛数据.xlsx'
        
    def sealMatching(self):
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
    
    def wordMatching(self, pattern):
        workbook = xlsxwriter.Workbook('匹配结果.xlsx')
        worksheet = workbook.add_worksheet()

        with open(self.workbook_file, 'rb') as file:
            xlsx_reader = xlsxwriter.Workbook(file, {'in_memory': True})
            xlsx_sheet = xlsx_reader.get_sheet_by_index('0')

            for row_idx in range(xlsx_sheet.nrows):
                row = xlsx_sheet.row_values(row_idx)
                new_row = []
                
                for cell_value in row:
                    if cell_value:
                        match = re.search(pattern, str(cell_value))
                        if match:
                            cell_value = re.sub(pattern, lambda x: f'[[red]]{x.group()}[[/red]]', str(cell_value))

                        worksheet.write_rich_string(row_idx, len(new_row), cell_value,)
                        new_row.append(cell_value)

        workbook.close()