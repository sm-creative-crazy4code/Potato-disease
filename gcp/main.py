from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME="sneha_personal_projects"  #name of the bucket created on google cloud
class_names=["Early Blight","Late Blight","Healthy"]

model =None
interpreter = None
input_index = None
output_index = None
# https://storage.cloud.google.com/sneha_personal_projects/CnnModels/modelH5.h5

def download_blob(bucket_name, source_blob_name,destination_blob_name):
    storage_client = storage.Client()
    bucket =storage_client.get_bucket(bucket_name)
    blob=bucket.blob(source_blob_name)
    blob.download_to_file(file_obj)


def predict(requests):
    global model
    if model is None:
        download_blob(BUCKET_NAME,"CnnModels/modelH5.h5","/tmp/potatoes.h5")
        model = tf.keras.models.load_model("/tmp/potatoes.h5")

    image= requests.files["file"]
    image=np.array(Image.open(image).convert("RGB").resize({256,256})) 
    image=image/255
    img_array=tf.expand_dims(image,0)
    predictions=model.predict(img_array)
    print(predictions)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return{"class":predicted_class,"confidence":confidence}



