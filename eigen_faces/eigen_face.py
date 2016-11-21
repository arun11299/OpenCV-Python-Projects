import numpy as np
import cv2
import glob

"""
https://onionesquereality.wordpress.com/2009/02/11/face-recognition-using-eigenfaces-and-distance-classifiers-a-tutorial/
"""

def get_all_images(num_persons, imgs_per_person):
  image_map = dict()
  for person_dir in range(num_persons):
    person_dir = person_dir + 1
    person_dir = "./db/s{}".format(person_dir)

    for image in range(imgs_per_person):
      image = image + 1
      image_name = person_dir + "/{}.pgm".format(image)
      img = cv2.imread(image_name)
      gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      cv2.imshow("im", gray_image)
      print gray_image.shape
      image_map[image_name] = gray_image

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

  return np.uint8(accum) / len(image_map.keys())


def get_normalized_face_vectors(image_map, mean):
  norm_img_vector_map = dict()
  for (name, img) in image_map.items():
    print name
    f = np.array(img.flatten()).reshape(img.shape[0]*img.shape[1], 1)
    norm_img_vector_map[name] = f - mean

  return norm_img_vector_map



if __name__ == "__main__":
  num_diff_persons = 4
  images_per_person = 1
  total_images = num_diff_persons * images_per_person

  image_map = get_all_images(num_diff_persons, images_per_person)
  print "Got all images"

  avg_face_vector = get_mean_face(image_map)
  print "Mean:"
  print avg_face_vector.shape
  print avg_face_vector

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

  first = np.array(eigen_face[:, 0]).reshape(112, 92)
  print first.shape

  cv2.imshow("image", first)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

