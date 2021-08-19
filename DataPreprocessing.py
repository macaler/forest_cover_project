''' This should do just the data preprocessing '''

# Import analysis packages:
import pandas as pd
import numpy as np

import random
import tensorflow as tf

# Import data pre-processing packages:
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(11)

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(11)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(11)

forest_cover_data = pd.read_csv('cover_data.csv')

forest_cover_data['Aspect'] = forest_cover_data['Aspect'].apply(lambda x: (x-360) if x > 180 else x)

ratio_train = 0.7
ratio_valid = 0.15
ratio_test = 0.15
ratio_valid_adjusted = ratio_valid / (1 - ratio_test)

def output_train_test_data():
    
    variables = forest_cover_data.iloc[:,0:53]
    labels = forest_cover_data.iloc[:,-1]

    labels_list = labels.to_list()
    
    vars_nottest, vars_test, labels_nottest, labels_test = \
    train_test_split(variables, labels, test_size = ratio_test, stratify = labels_list, random_state = 11)
    
    labels_nottest_list = labels_nottest.to_list()
    
    vars_train, vars_valid, labels_train, labels_valid = \
    train_test_split(vars_nottest, labels_nottest, test_size = ratio_valid_adjusted, stratify = labels_nottest_list, random_state = 11)
    
    print('Proportion of Data Points per Class, Full Data Set:')
    print(round(labels.value_counts()/len(labels),4))
    print('')
    print('Proportion of Data Points per Class, Training Data Subset:')
    print(round(labels_train.value_counts()/len(labels_train),4))
    print('')
    print('Proportion of Data Points per Class, Validation Data Subset:')
    print(round(labels_valid.value_counts()/len(labels_valid),4))
    print('')
    print('Proportion of Data Points per Class, Test Data Subset:')
    print(round(labels_test.value_counts()/len(labels_test),4))
    
    coltransform = ColumnTransformer([('minmax', MinMaxScaler(), \
                                     ['Elevation', 'Aspect', 'Slope', \
                                      'Horizontal_Distance_To_Hydrology', \
                                      'Vertical_Distance_To_Hydrology', \
                                      'Horizontal_Distance_To_Roadways', \
                                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', \
                                      'Horizontal_Distance_To_Fire_Points'])], \
                                     remainder='passthrough')
    vars_scaled_nottest = coltransform.fit_transform(vars_nottest)
    vars_scaled_train = coltransform.transform(vars_train)
    vars_scaled_valid = coltransform.transform(vars_valid)
    vars_scaled_test = coltransform.transform(vars_test)
    
    le=LabelEncoder()
    labelz_nottest=le.fit_transform(labels_nottest.astype(str))
    labelz_train=le.transform(labels_train.astype(str))
    labelz_valid=le.transform(labels_valid.astype(str))
    labelz_test=le.transform(labels_test.astype(str))

    labelz_train = to_categorical(labelz_train, dtype='int64')
    labelz_valid = to_categorical(labelz_valid, dtype='int64')
    labelz_test = to_categorical(labelz_test, dtype='int64')
    
    print('Training Data and Labels Shapes:')
    print(vars_scaled_train.shape, labelz_train.shape)
    print('')    
    print('Validation Data and Labels Shapes:')
    print(vars_scaled_valid.shape, labelz_valid.shape)
    print('')
    print('Test Data and Labels Shapes:')
    print(vars_scaled_test.shape, labelz_test.shape)
    
    return vars_scaled_train, labelz_train, vars_scaled_valid, labelz_valid, vars_scaled_test, labelz_test