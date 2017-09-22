#!/usr/bin/python
#coding:utf8
'for my image'
__author__ = 'Hippo'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt





plt.figure('test',figsize=(80,50), dpi=40)

# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
plt.subplot(5,2,1)

#X轴 从 0 到 5 一共10 个采样点
X = np.linspace(0, 5, 10,endpoint=True)

C,S = np.cos(X), np.sin(X)
#画cos 设置 X Y 轴的值,使用蓝色的、连续的、宽度为 1 （像素）的线条
plt.plot(X,C,color="blue", linewidth=1.0, linestyle="-",label="test1")

#画sin 设置 将S的值作为  X   X的值作为 Y 轴的值 不连续的线条
plt.plot(S,X,'.',color="green",label="test2")

#设置注释位置
plt.legend(loc='upper left')


# 设置横轴的上下限
plt.xlim(-4.0,6.0)

# 设置横轴记号 可以设置一个数组，第二个数组代表显示的内容
plt.xticks([-4,0,1,2,3,4,5,6],['-4','0','1','a2','a3','a4','a5','a6'])

# 设置竖轴的上下限
plt.ylim(min(C.min(),X.min())*1.1,max(C.max(),X.max())*1.1)

# 设置竖轴记号 从-4 到 6 一共 3 个标签  3-1 格
plt.yticks(np.linspace(-2,6,3,endpoint=True))

#每个图都有4条轴
ax = plt.gca()
#右边框设为透明
ax.spines['right'].set_color('none')
#上边框设置为透明
ax.spines['top'].set_color('red')

#将x轴设置为底边框
ax.xaxis.set_ticks_position('bottom')
#将数据0设置为边框位置
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
#将数据1设置为边y轴位置
ax.spines['left'].set_position(('data',1))

#设置每个坐标标签格式
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    #设置底色红色，透明度0.65 边缘颜色 透明
    label.set_bbox(dict(facecolor='red', edgecolor='None', alpha=0.65 ))

#设置标记点  x 坐标 0 2 y 坐标 2 2 两个点，大小10
plt.scatter([0,2,],[2,2,], 10, color ='blue')

#画虚线垂直线 从 0,0 到 0,2
plt.plot([0,0],[0,2], color ='red', linewidth=2.5, linestyle="--")

plt.plot([2,2],[0,2], color ='red', linewidth=2.5, linestyle="--")


#给1 2 这个点增加注释
plt.annotate(r'this is test',
         xy=(0, 2), xycoords='data',
         xytext=(+10, +30), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

################################################
#散点
plt.subplot(5,2,2)
n = 64
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
plt.scatter(X,Y)

################################################
#柱状图
plt.subplot(5,2,3)

n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

plt.ylim(-1.25,+1.25)

####################
#等高
plt.subplot(5,2,4)

def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap='jet')
C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)

##############
#灰度图
def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

plt.subplot(5,2,5)
n = 10
x = np.linspace(-3,3,4*n)
y = np.linspace(-3,3,3*n)
X,Y = np.meshgrid(x,y)
plt.imshow(f(X,Y))


###########
#饼状图
plt.subplot(5,2,6)
n = 20
Z = np.random.uniform(0,1,n)
plt.pie(Z)






#量场图
plt.subplot(5,2,7)

n = 8
X,Y = np.mgrid[0:n,0:n]
plt.quiver(X,Y)



#网格
plt.subplot(5,2,8)
axes = plt.gca()
axes.set_xlim(0,4)
axes.set_ylim(0,3)
axes.set_xticklabels([])
axes.set_yticklabels([])




#极轴图
plt.subplot(5,2,9)
#plt.axes([0,0,1,1])

N = 20
theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
radii = 10*np.random.rand(N)
width = np.pi/4*np.random.rand(N)
bars = plt.bar(theta, radii, width=width, bottom=0.0)

for r,bar in zip(radii, bars):
    bar.set_facecolor( plt.cm.jet(r/10.))
    bar.set_alpha(0.5)




#3D图
#plt.subplot(5,2,10)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')


#显示
plt.show()






