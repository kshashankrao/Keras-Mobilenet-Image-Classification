from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.applications import mobilenet
import numpy as np
import cv2


from keras import models
image_size = 224

model = mobilenet.MobileNet(weights='imagenet',input_shape=(image_size, image_size, 3))


image = cv2.imread('image.jpg')
orig = image.copy()

image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)
label = decode_predictions(prediction)
print(label)
label = str(label[0][0])
print(label)

cv2.putText(orig, label, (50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 4)
cv2.imshow("Image",orig)
cv2.waitKey(0)
