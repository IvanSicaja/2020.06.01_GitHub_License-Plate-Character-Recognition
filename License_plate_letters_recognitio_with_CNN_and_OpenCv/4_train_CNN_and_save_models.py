# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings

# This Python 3 environment comes with many helpful anal,ytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras import backend as K


from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

 


image_index=77777   

import os


# Any results you write to the current directory are saved as output.
# read train 
train = pd.read_csv("5.0_Data_for_traning_CNN/My_Data_A-Z_and_0-9__HandwrittenData_train.csv")

print("#############################################################################")
print("Initial¸train shape is :", train.shape)
print("#############################################################################")
train.head()

# read test 
test= pd.read_csv("5.0_Data_for_traning_CNN/A_Z_HandwrittenData_test.csv")
print("#############################################################################")
print("Initial¸test shape is :", test.shape)
print("#############################################################################")
test.head()

# put labels into y_train variable
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

print("#############################################################################")
print("LABELED SHAPES")
print("Train label shape is :", Y_train.shape)
print("Train image shape is :", X_train.shape)
print("#############################################################################")



# visualize number of digits classes
#plt.figure(figsize=(15,7))

g = sns.countplot(Y_train, palette="icefire")
#plt.title("Number of digit classes")
#plt.show()
Y_train.value_counts()

# plot some samples
img = X_train.iloc[0].to_numpy()
print("#############################################################################")
print("BEFORE RESHAPE SHAPES")
print("Train img shape is :", img.shape)
print("#############################################################################")
img = img.reshape((28,28))
#plt.imshow(img,cmap='gray')
#plt.title(train.iloc[0,0])
#plt.axis("off")
#plt.show()

print("#############################################################################")
print("PREVIEWED SHAPES")
print("Train img shape is :", img.shape)
print("#############################################################################")

# plot some samples
img = X_train.iloc[3].to_numpy()
img = img.reshape((28,28))
#plt.imshow(img,cmap='gray')
#plt.title(train.iloc[3,0])
#plt.axis("off")
#plt.show()

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print("#############################################################################")
print("AFTER NORMALISE SHAPES")
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)
print("#############################################################################")

# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

print("#############################################################################")
print("AFTER RESHAPEING BY (-1,28,28,1)")
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)
print("#############################################################################")

# Label Encoding 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 36)

# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

print("#############################################################################")
print("AFTER SPLITTING FOR VALIDATION")
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)
print("#############################################################################")

# Some examples
#plt.imshow(X_train[2][:,:,0],cmap='gray')
#plt.show()

# 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(36, activation = "softmax"))

# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


#model.add(Dense(26, activation = "softmax"))


##################################################################################################################################
epochs = 60  # for better result increase the epochs
batch_size =512
##################################################################################################################################


# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset     
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=False,  # randomly rotate images in the range 5 degrees
        zoom_range = 1, # Randomly zoom image 5%
        width_shift_range=1,  # randomly shift images horizontally 5%
        height_shift_range=1,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=True)  # randomly flip images


datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)


model.summary()

# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 


# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
#plt.xlabel("Predicted Label")
#plt.ylabel("True Label")
#plt.title("Confusion Matrix")
#plt.show()


# Saving model  
model.save("5.1_Trained_saved_CNN_models/Model_Batchsize="+str(batch_size)+"__Epoch="+str(epochs)+"__Accuracy=.h5")
print("Model saved.")


print("################################################################")
print("Test image shape is :" ,test[image_index][:,:,0].shape)
print("################################################################")
print("Plotting image for prediction")
plt.imshow(test[image_index][:,:,0],cmap='gray')
plt.show()
print("Image for predivtion plotted")
print("################################################################")
pred = model.predict(test[image_index].reshape(1, 28, 28, 1))
print("Predicted value is : ")
print(pred.argmax())
#plt.show()

#print("                                                                ")
print("################################################################")

print("End of script ;)")
 