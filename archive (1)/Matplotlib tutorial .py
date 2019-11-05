# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:08:31 2019

@author: jg

Matplotlib tutorial code

"""

import numpy as np
import matplotlib.pyplot as plt
import imageio


data = np.arange(100, 201)
plt.plot(data)
plt.show()

# multiple graphs


data = np.arange(100, 201)
plt.plot(data)

data2 = np.arange(200, 301)
# generate a new figure window
plt.figure()
plt.plot(data2)

plt.show()

# using subplots
# subplot.py

data = np.arange(100, 201)
plt.subplot(2, 2, 1)
plt.plot(data)

data2 = np.arange(200, 301)
plt.subplot(2, 1, 2)
plt.plot(data2)

plt.show()

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.figure()
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

# linear plots

plt.plot([1, 2, 3], [3, 6, 9], '-r')
plt.plot([1, 2, 3], [2, 4, 9], ':g')

plt.show()


# Scatter plot
plt.close('all')

N = 20

plt.scatter(np.random.rand(N) * 100,
            np.random.rand(N) * 100,
            c='red', s=10, alpha=0.5)

plt.scatter(np.random.rand(N) * 100,
            np.random.rand(N) * 100,
            c='g', s=200, alpha=0.5)

plt.scatter(np.random.rand(N) * 100,
            np.random.rand(N) * 100,
            c='pink', s=300, alpha=0.9)

plt.show()

# pie charts


labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

data = np.random.rand(7) * 100

plt.pie(data, labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.legend()

plt.show()

# bar charts


N = 7

x = np.arange(N)
data = np.random.randint(low=0, high=100, size=N)
colors = np.random.rand(N * 3).reshape(N, -1)
labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.title("Weekday Data")
plt.xlabel("Week Days")
plt.ylabel("Number of Customers")
plt.bar(x, data, alpha=0.8, color=colors, tick_label=labels)
plt.show()

# you can also plot a horizontal bar charts for categorical data
plt.figure()
plt.title("Weekday Data")
plt.barh(x,data, alpha=0.8, color=colors, tick_label=labels)
plt.show()
# Histogram plot


data = [np.random.randint(0, n, n) for n in [3000, 4000, 5000]]
labels = ['3K', '4K', '5K']
bins = [0, 100, 500, 1000, 2000, 3000, 4000, 5000]


plt.hist(data, bins=bins, label=labels)
plt.legend()

plt.show()


# 3D mesh plot using Matplot lib

'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# images

img = imageio.imread('imageio:chelsea.png')
img_tinted = img * [1, 0.75, 0.7]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()