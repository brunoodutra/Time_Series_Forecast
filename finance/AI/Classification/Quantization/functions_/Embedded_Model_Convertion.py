import tensorflow as tf
import time
from Embedded_Model import Embedded_Model

class Embedded_Model_Convertion(Embedded_Model):

    def __init__(self, model_path):
        '''Basic Init Function, record model path, load the model from model_path and then convert the model to TFLite

        Args:
            model_path (string): Model Path
        '''
        self.compatibility_detection()
        self.model_path = model_path
        self.convert_model_tflite()
        self.interpreter = self.load_tflite_model()
        
    def compatibility_detection(self):
        ''' Verify the compatibility of the tf and TFLite versions
        '''  
        
        @tf.lite.experimental.authoring.compatible
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.float32)
        ])
        def f(x):
            return tf.cosh(x)

        # Evaluate the tf.function
        result = f(tf.constant([0.0]))
        print (f"result = {result}")
        
    def convert_model_tflite(self):
        ''' Convert and Optimize the Keras model to TFLite model
        '''     
        t0 = time.time()
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path) # Load model from model_path
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]                  # Optimize the model
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        #converter.allow_custom_ops = True # does the trick
        try:
            tflite_model = converter.convert()                                    # Convert the model to TFLite
        except Exception as e:
            print(f"Got an exception: {e}")
        

        with open(f'{self.model_path}.tflite', 'wb') as f:                    # Save the model
            f.write(tflite_model)
        t1 = time.time()
        print(f'A conversão do modelo durou {t1-t0} segundos.')