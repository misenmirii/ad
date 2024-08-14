import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# get model
model = EfficientNetV2L()

# pretreatment
img_path = './picture/img.png'
img = image.load_img(img_path, target_size=(480, 480))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# predict
predtop3 = decode_predictions(model.predict(x), top=3)[0]
print('Predicted:', predtop3)

# Load and display the original image
img = image.load_img(img_path, target_size=(480, 480))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

# Display the original image on the first subplot
ax1.imshow(img)
ax1.axis('off')
ax1.set_title('Original Image')

# Display predicted labels and probabilities on the second subplot
labels = [pred[1] for pred in predtop3]
probabilities = [pred[2] for pred in predtop3]
ax2.barh(range(3), probabilities, tick_label=labels)
ax2.set_xlabel('Probability')
ax2.set_title('Top 3 Predictions')

plt.show()
