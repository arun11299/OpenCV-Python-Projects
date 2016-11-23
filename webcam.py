import cv2

camera = cv2.VideoCapture(0)

img_count = 0

while True:
    return_value,image = camera.read()

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',gray)

    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite('test-{}.jpg'.format(img_count),image)
        img_count += 1
        if img_count == 5: break


    #if img_count == 5: break

camera.release()
cv2.destroyAllWindows()
