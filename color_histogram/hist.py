import cv2
import numpy
from matplotlib import pyplot as plt
import time

img = cv2.imread("grant_jp.jpg")
cv2.imshow("image", img)

# Split the image into R, B, G channels

chans = cv2.split(img)
colors = ("b", "g", "r")

plt.figure()
plt.title("Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []

for (color, ch) in zip(colors, chans):
  hist = cv2.calcHist(images = [ch],
                      channels = [0], #only grayscale, [0,1,2] for RGB
                      mask = None,
                      histSize = [256], #The number of bins we want to compute
                      ranges = [0, 256] # range of values to be considered for computing histogram
                      )

  features.extend(hist)
  plt.plot(hist, color = color)
  plt.xlim([0, 256])


# Convert to gray scale
#gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gs_image", gs_img)
plt.show()

if cv2.waitKey(10) == ord(' '):
  cv2.destroyAllWindows()
