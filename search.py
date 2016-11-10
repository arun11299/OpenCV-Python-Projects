from image_descriptor import RGBHistogram
import cv2
import numpy as np
import glob
import sys


class ImSearch:
  def __init__(self):
    self.index = dict() # key = image name ; Value = descriptor
    self.image_repo = "./imageDB/"
    self. histo = RGBHistogram([8, 8, 8])

  def create_index(self):
    for path in glob.glob(self.image_repo + "*.png"):
      name = path[path.rfind("/") + 1:]
      image = cv2.imread(path)
      self.index[name] = self.histo.descriptor(image)

  def search(self, query_desc):
    result_map = dict()
    for (name, feature_desc) in self.index.items():
      d = self.chi2_dist(feature_desc, query_desc)
      result_map[name] = d

    result_map = sorted([(v, k) for (k, v) in result_map.items()])

    return result_map


  def chi2_dist(self, f1, f2, eps = 1e-10):
    d = 0.5 * np.sum([
            ((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(f1, f2)])

    return d


if __name__ == "__main__":
  print sys.argv[1]
  img = cv2.imread(sys.argv[1])
  h = RGBHistogram([8, 8, 8])
  feature = h.descriptor(img)

  srch = ImSearch()
  srch.create_index()
  res = srch.search(feature)
  print res
  
