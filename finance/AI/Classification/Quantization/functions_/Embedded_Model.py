import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

class Embedded_Model():
    
    def __init__(self, model_path):
        '''Basic Init Function, record model path, load the model from model_path and then convert the model to TFLite

        Args:
            model_path (string): Model Path
        '''
        self.model_path = model_path
        self.interpreter = self.load_tflite_model()
        
    def print_model_details(self):
        '''Print Model details (Layers, resulting data types after each layer operation)
        '''
        for details in self.interpreter.get_tensor_details():                        # Iterates for each model detail
            print(details['name'], details['dtype'])
        print(details.keys())
        
    def set_input_tensor(self, data):
        ''' Load data to be predicted in the model input layer

        Args:
            interpreter (Interpreter): TFLite model
            data ([float]): data to be loaded
        '''
        tensor_index = self.interpreter.get_input_details()[0]['index']   # Get Input Layer expected data format 
        input_tensor = self.interpreter.tensor(tensor_index)()[0]         
        input_tensor[:, :] = data                                         # Load data into Input Layer

    def classify_data(self, data):
        ''' Model classification method

        Args:
            interpreter (Interpreter): TFLite model
            data ([float]): data to be predicted

        Returns:
            output ([int]): Model output list for each data sample

        '''
        self.set_input_tensor(data)                                                  # Load data into Model Input Layer

        self.interpreter.invoke()                                                    # Runs Classification to data loaded on Input Layer
        output_details = self.interpreter.get_output_details()                    # Get model Output Layer expected data format 

        if len(output_details)>1:
            # Get predictions for all output heads
            predictions = []
            for output_detail in output_details:

                output = np.squeeze(self.interpreter.get_tensor(output_detail['index']))    # Format the model output to expected format
                predictions.append(output)
        else:
            output_details = self.interpreter.get_output_details()[0]                    # Get model Output Layer expected data format 
            predictions = np.squeeze(self.interpreter.get_tensor(output_details['index']))    # Format the model output to expected format
        return predictions

    def load_tflite_model(self):
        ''' Load TFLite model

        Returns:
            interpreter (Interpreter): TFLite model
        '''

        interpreter = Interpreter(self.model_path+'.tflite')  # Load TFLite model from model path
        interpreter.allocate_tensors()                        # Allocate model tensors
        
        return interpreter
    
    def classify_batch_data(self, X_test):
        '''Model classification to data batch

        Args: 
            X_test ([float]): Signal Data Batch

        Returns:
            y_pred ([int]): Prediction array
        '''
        n_classifications = X_test.shape[0]                                 # Get number of classifications

        y_pred = []                                                         # Initialize a empty array of predictions
        t0 = time.time()
        for i in range(n_classifications):                                  # Interates over number of classifications
            print(f'{i+1} out of {n_classifications}', end='\r')            # Debugging
            y_pred.append(self.classify_data(X_test[i]))                    # Appending the prediction for the i-th data sample in Data Batch
        
        y_pred = np.array(y_pred) - 1                                       # Post-processing predictions array ( keep between the interval of 0 to n_classes-1 )
        y_pred = np.argmax(y_pred, axis=1)                                  # Casting the predictions array from a (n_classes, n_samples) shape array of 0 and 1 to a (n_samples) 
                                                                            # shape array ranging from 0 to n_classes-1
        t1=time.time()

        print(f'A classificação de todos as amostras foi de {t1-t0} segundos ou {np.round((t1-t0)/n_classifications, 2)} segundos por amostras')

        return y_pred