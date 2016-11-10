import numpy as np
import cv2

class RGBHistogram:
  def __init__(self, bins):
    self.bins = bins

  def descriptor(self, image):
    """
      Computes the 3D histogram in the RGB colorspace
      then normalizes the histogram so that images with the 
      same content but different scales will more or less
      will have the same histogram
    """
    hist = cv2.calcHist([image], 
                        [0, 1, 2], 
                        None, 
                        self.bins, 
                        [0, 256, 0, 256, 0, 256]
                        )
    hist = cv2.normalize(hist)
    
    # Returns the histogram as a flat array instead
    # as a matrix, because vector distance is easier to
    # calculate.
    return hist.flatten()
