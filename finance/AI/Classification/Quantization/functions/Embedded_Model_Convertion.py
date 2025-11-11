import tensorflow as tf
import time

try:
    from src.learning.embedded.Quantization.functions.Embedded_Model import Embedded_Model
except:
    from Embedded_Model import Embedded_Model


class Embedded_Model_Convertion(Embedded_Model):

    def __init__(self, model_path, destination_path="", quantize=False):
        """
        Convert a Keras/TensorFlow model to TFLite, with optional quantization.

        Args:
            model_path (str): Path to SavedModel or Keras model.
            destination_path (str): Where to save .tflite model (default = model_path).
            quantize (bool): If True, apply dynamic quantization. If False, keep float32.
        """
        if destination_path == "":
            self.destination_path = model_path
        else:
            self.destination_path = destination_path

        super().__init__(self.destination_path)

        self.model_path = model_path
        self.quantize = quantize

        self.compatibility_detection()
        self.convert_model_tflite()
        self.interpreter = self.load_tflite_model()

    def compatibility_detection(self):
        """Simple check for TF/TFLite compatibility"""
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def f(x):
            return tf.cosh(x)

        result = f(tf.constant([0.0]))
        print(f"compatibility test result = {result}")

    def convert_model_tflite(self):
        """Convert the Keras model to TFLite, with or without quantization"""
        t0 = time.time()
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)

        if self.quantize:
            print("⚡ Applying dynamic quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert
        tflite_model = converter.convert()

        # Save
        with open(f'{self.destination_path}.tflite', 'wb') as f:
            f.write(tflite_model)

        t1 = time.time()
        print(f'The model conversion took {t1-t0:.2f} seconds.')
        print(f'Model saved at: {self.destination_path}.tflite')
