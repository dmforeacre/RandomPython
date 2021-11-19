"""
Author: Daniel Foreacre
Date:   11/18/21
Lab:    Keras/Neural Networks
Desc:   Lab using Keras to create a neural network to classify type of tissue
        from test data.
"""

# Libraries
import keras
from numpy.core.fromnumeric import mean
import tensorflow
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import pandas

# Main function
# @return   A Keras model object
def get_model():
    # Load data from file
    dataset = numpy.loadtxt("breasttissue_train.csv", delimiter=',')
    outputs = dataset[:,0]
    inputs = dataset[:,1:10]

    # Scale data to a 0,1 range
    scaler = MinMaxScaler((0,1))
    scaler.fit(inputs)
    scaler.transform(inputs)

    # Create Keras model    
    model = keras.Sequential()
    model.add(keras.layers.Dense(32, input_dim=9, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(32, activation='sigmoid'))
    model.add(keras.layers.Dense(4, activation='softmax'))

    # K-fold cross validation
    kfold = KFold(n_splits=4, shuffle=True)

    acc_per_fold = []
    fold_no = 1
    for train, test in kfold.split(inputs, outputs):
        print(f"Training fold {fold_no}.")

        model.compile(loss='mean_squared_error', optimizer='adam',metrics='accuracy')

        history = model.fit(inputs[train], outputs[train], epochs=150, batch_size=50, verbose=0)

        scores = model.evaluate(inputs[test], outputs[test])

        acc_per_fold.append(scores[1] * 100)

        fold_no = fold_no + 1

    # Display results
    for i in range(0, len(acc_per_fold)):
        print(f'Fold {i+1} accuracy: {acc_per_fold[i]:.2f}%')
    print(f'Average accuracy: {mean(acc_per_fold):.2f}')

    return model

thisModel = get_model()
