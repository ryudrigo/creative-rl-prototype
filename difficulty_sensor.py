from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics as kmetrics
from sklearn.metrics import confusion_matrix

class Sensor:
    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        self.model = models.Sequential()
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(16, input_shape= (6*3,), activation = 'relu'))
        self.model.add(layers.Dense(4, activation = 'relu'))
        self.model.add(layers.Dense(1, activation = 'sigmoid'))
        metrics=[kmetrics.BinaryAccuracy(name='accuracy'), kmetrics.FalseNegatives(name='fn'), kmetrics.Recall(name='recall'), kmetrics.Precision(name='precision')]
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics = metrics)

        K.set_value(self.model.optimizer.learning_rate, 1e-4)
    
    def train(self, data, labels):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
        
        counts = np.bincount(y_train)
        weight_for_0 = (1.0 /counts[0])
        weight_for_1 = (1.0 /counts[1])
        class_weight = {0:weight_for_0, 1:weight_for_1}
        
        self.model.fit (X_train, y_train, epochs =1000, batch_size = 2048, verbose=2, class_weight=class_weight)
        test_loss, test_accm, test_fn, test_recall, test_precision = self.model.evaluate(X_test, y_test)
        
        predictions = self.model.predict(X_test)
        print (predictions)
        print (confusion_matrix(y_test, predictions>predictions.mean()))
        print (confusion_matrix(y_test, predictions>0.5))
        print (test_loss, test_accm, test_fn, test_recall, test_precision)
        print (counts)        
        
        self.model.save('sensor')
    def load (self):
        self.model = load_model('sensor')
    def predict(self, data):
        return self.model.predict(data)

