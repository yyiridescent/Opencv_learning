# <center>OPENCV学习笔记
## 图像
### 坐标系
坐标原点在左上角

每一个点的坐标是（height,width,channel） 

一张彩色的图片的channel是3：每张图每个像素有3个颜色组合而成（黑白图片的channel为2）
### opencv基本操作
读取图片:`cv2.imread`

获取图片的形状：`img.shape`，返回一个(roew,heights,channels)

获取图片的大小：`img.size`, 返回一个rows*heights*channels

显示图片：`cv2.imshow()`

等待：`cv2.waitKey(0)`

关闭：`cv2.destroyAllWindows()`

#### 彩色图片：
```python
import cv2#导入库
img = cv2.imread('./风景.jpg')
img.shape#读取图片形状
img_dtype=img.dtype#返回数据类型
#（B，G，R）
(b,g,r)=img[2,6]
#取色
b=img[2,6,0]
g=ing[2,6,1]
r=img[2,6,2]
#重新给像素赋值，更换颜色
img[2,6]=(0,,0,255)
#显示图片
cv2.imshow("image",img)
#等待一定时间
cv2.waitKey(0)
#关闭窗口
cv2.destroyAllWindows()
```
 灰度图片的通道默认是0

#### BGR顺序
```python
import cv2
import matplotlib as plt
img_logo=cv2.imread('./logo.jpg')
b,g,r=cv2.split(img_logo)
img_new=cv2.merge([r,g,b])#改变r,g,b顺序
plt.subplot(121)
plt.imshow(img_logo)#绘制图像
plt.subplot(122)
plt.imshow(img_new)
plt.show()#展示图像
```
### 读取、显示图片
```python
import cv2
import argparse

#获取参数
parser=argparse.ArgumentParser()

#添加参数
parser.add_argument("-path_image",help="path to input the image") #通过终端传参数

#解析参数
args=parser.parse_args()

#加载图片 方式一
img=cv2.imread(args.path_image)
cv2.imshow("logo",img)

#加载图片 方式二
args_dict=vars(parser.parse_args()) #以字典形式
img2=cv2.imread(args_dict["path_image"])
cv2.imshow("logo_two",img2)

#等待
cv2.waitKey(0)

#关闭
cv2.destroyWindow()
```
### 读取、处理、保存图片
```python
import cv2
import argparse

#获取参数
parser=argparse.ArgumentParser()

#添加参数
parser.add_argument("img_input",help="read one image")
parser.add_argument("img_output",help="save the processed image")

#解析参数，用字典形式保存参数和值
args=vars(parser.parse_args())

#加载图片
img=cv2.imread(args["img_input"])

#灰度处理
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 保存图片
cv2.imwrite(args["img_output"],img_gray)

#显示图片
cv2.imshow("Original picture",img)
cv2.imshow("Gray picture",img_gray)

#等待
cv2.waitKey(0)

#关闭窗口
cv2.destroyWindow()
```