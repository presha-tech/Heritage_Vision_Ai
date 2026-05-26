import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model(
    "model/best_resnet_model.h5"
)

print("Converting...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

with open(
    "model/heritage_model.tflite",
    "wb"
) as f:

    f.write(tflite_model)

print("Done.")