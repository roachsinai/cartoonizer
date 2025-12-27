import cv2
import numpy as np

img = cv2.imread("Boat.JPG")
print(img.shape)

# 1) Edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5) # 每个位置的像素值为周围5x5区域的像素的均值
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9) # 哪些位置的像素为黑哪些为白，后面的数越大灰度值中255越少

# 2) Color
# color = cv2.medianBlur(img, 21) # 事实上，直接对原图进行很大size的模糊，得到的结果也不差，😃。主要是边缘扭曲太严重。
# color = cv2.bilateralFilter(img, 9, 300, 300)
color = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
cv2.imshow("color", color)
# color = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

# 3) Cartoon
cartoon = cv2.bitwise_and(color, color, mask=edges) # mask中为255的使用color对应位置的像素值，mask中为0的对应位置的像素值也是0。这就显示出了漫画中的线条。

# one_step = cv2.stylization(img, sigma_s=60, sigma_r=0.07)

# cv2.imshow("Image", img)
cv2.imshow("Cartoon", cartoon)
# cv2.imshow("color", color)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
