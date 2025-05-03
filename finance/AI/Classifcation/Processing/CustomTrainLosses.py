from keras import backend as K
import tensorflow as tf

class CustomTrainLosses():
    def __init__(self):
        pass

    def focal_loss(self,gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            cross_entropy = -y_true * K.log(y_pred)
            focal_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
            return K.sum(focal_loss, axis=-1)
        return loss
    
    def custom_weighted_categorical_crossentropy(self, weights, penalty_matrix):
        """
        Custom weighted categorical crossentropy with penalty for false positives.
        
        Args:
            weights: numpy array of shape (C,) where C is the number of classes.
            penalty_matrix: numpy array of shape (C, C) where penalty_matrix[i][j] is the penalty
                        for classifying class i as class j.
        
        Returns:
            A loss function.
        """

        weights = tf.Variable(weights, dtype=tf.float32)
        penalty_matrix = tf.Variable(penalty_matrix, dtype=tf.float32)
        
        def loss(y_true, y_pred):
            # Clip predictions to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            
            # Calculate the base loss
            base_loss = y_true * K.log(y_pred) * weights
            base_loss = -K.sum(base_loss, -1)
            
            # Calculate the penalty for false positives
            y_true_class = K.argmax(y_true, axis=-1)  # True classes
            y_pred_class = K.argmax(y_pred, axis=-1)  # Predicted classes
            
            # Create a mask for false positives
            false_positive_mask = tf.not_equal(y_true_class, y_pred_class)
            
            # Get the penalty for each sample
            sample_penalties = tf.gather_nd(
                penalty_matrix,
                tf.stack([y_true_class, y_pred_class], axis=-1)
            )
            
            # Apply the penalties only to false positives
            penalty_loss = tf.where(false_positive_mask, sample_penalties, 0.0)
            
            # Combine the base loss and penalty loss
            total_loss = base_loss + penalty_loss
            return total_loss
        
        return loss
    
    def weighted_categorical_crossentropy(self,weights):
        """
        from https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        
        #weights = K.variable(weights)
        weights = tf.Variable(weights, dtype=tf.float32)    
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            #y_true_printed = tf.print("y_true =", y_true)
            #y_pred_printed = tf.print("y_pred =", y_pred)
            
            #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
        
        return loss 