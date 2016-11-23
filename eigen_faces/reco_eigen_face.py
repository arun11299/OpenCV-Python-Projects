import numpy as np
import cv2
import glob

"""
https://onionesquereality.wordpress.com/2009/02/11/face-recognition-using-eigenfaces-and-distance-classifiers-a-tutorial/
"""

def get_target_images(dir_name, imgs_per_person):
  image_map = dict()
  person_dir = "./db/{}".format(dir_name)

  for image in range(imgs_per_person):
    image = image + 1
    image_name = person_dir + "/test-{}.jpg".format(image)
    img = cv2.imread(image_name)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (240, 220)) 
    cv2.imshow(person_dir, resized_image)
    print resized_image.shape
    image_map[image_name] = resized_image

  return image_map


def get_mean_face(image_map):
  accum = None
  first = True
  for (iname, img) in image_map.items():
    vect = np.array(img.flatten(), dtype=np.uint16).reshape(img.shape[0]*img.shape[1], 1)
    print vect.T
    if first:
      first = False
      accum = vect
      continue

    accum = cv2.add(accum, vect)
    print "Res: "
    print accum.T

  return np.uint8(accum / len(image_map.keys()))


def get_normalized_face_vectors(image_map, mean):
  norm_img_vector_map = dict()
  for (name, img) in image_map.items():
    f = np.array(img.flatten()).reshape(img.shape[0]*img.shape[1], 1)
    norm_img_vector_map[name] = np.uint8(f - mean)
    cv2.imshow("subtracted-{}".format(name), norm_img_vector_map[name].reshape(img.shape[0], img.shape[1]))

  return norm_img_vector_map



if __name__ == "__main__":
  images_per_person = 3
  total_images = images_per_person

  image_map = get_target_images("arun", images_per_person)
  print "Got all images"

  avg_face_vector = get_mean_face(image_map)
  print "Mean:"
  print avg_face_vector.shape
  cv2.imshow("mean-face", avg_face_vector.reshape(240, 220))

  norm_img_vector_map = get_normalized_face_vectors(image_map, avg_face_vector)
  print "normalization done"

  mat_A = np.hstack([v for (n, v) in norm_img_vector_map.items()])
  print mat_A.shape

  covariance_mat = np.dot(mat_A.T, mat_A)
  print covariance_mat
  print covariance_mat.shape

  eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
  print eigenvectors

  eigen_face = np.inner(mat_A, eigenvectors * -1)
  print eigen_face.shape * -1

  print eigen_face

  for i in range(eigen_face.shape[1]):
    first = np.array(eigen_face[:, i]).reshape(240, 220)
    cv2.imshow("image-{}".format(i), first)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

