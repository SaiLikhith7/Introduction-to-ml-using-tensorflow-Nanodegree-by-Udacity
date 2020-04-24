import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import PIL
import json

parse = argparse.ArgumentParser()

parse.add_argument('--img_path', required=True)
parse.add_argument('--model', required=True)
parse.add_argument('--topK', default=3, required=False)
parse.add_argument('--category_names', default='label_map.json', required=False)

args = parse.parse_args()
model = tf.keras.models.load_model(str(args.model), custom_objects={'KerasLayer':hub.KerasLayer})
image = np.asarray(PIL.Image.open(args.img_path))
top_k = args.topK

with open(str(args.category_names), 'r') as f:
    class_names = json.load(f)

# Defining necessary functions
def process_image(image):
    image_tf = tf.convert_to_tensor(image)
    image_size = tf.image.resize(image_tf, [224, 224])
    image_norm = image_size/225
    return image_norm.numpy()

def predict(image, model, top_k=3):
    processed_image = process_image(image)
    img_batch = np.expand_dims(processed_image, axis=0) # image (224,224,3) is expended to img_batch (1,224,224,3)
    output = model.predict(img_batch) # output is a list of 102 probability values, whose indixes represent the key to flower names dictionary.
    top_index = (-output[0,:]).argsort() # Returns the index of elements sorted in descending order
    top_probs = output[0, (-output[0,:]).argsort()] # Returns the elements in descending order
    return  top_index[0:top_k], top_probs[0:top_k]

#with tf.device('/GPU:0'):
indexes, probs = predict(image, model, top_k)

names=[]
for i in indexes:
    names.append(class_names[str(i+1)])

for j in range(len(names)):
    print(f"\nThe probability that the flower is {names[j]} is {probs[j]}")