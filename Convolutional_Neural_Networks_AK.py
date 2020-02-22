#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[3]:


#Print the data type of x_train
print(type(x_train))
#Print the data type of y_train
print(type(y_train))
#Print the data type of x_test
print(type(x_test))
#Print the data type of y_test
print(type(y_test))


# In[4]:


#Get the shape of x_train
print('x_train shape:', x_train.shape)
#Get the shape of y_train
print('y_train shape:', y_train.shape)
#Get the shape of x_train
print('x_test shape:', x_test.shape)
#Get the shape of y_train
print('y_test shape:', y_test.shape)


# In[5]:


x_train[0]


# In[7]:


import matplotlib.pyplot as plt
img = plt.imshow(x_train[0])


# In[8]:


print('The label is:', y_train[0])


# In[11]:


from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


# In[12]:


print(y_train_one_hot)


# In[13]:


print('The one hot label is:', y_train_one_hot[0])


# In[14]:


x_train = x_train / 255
x_test = x_test / 255


# In[15]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
model = Sequential() 

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax')) 


model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


# In[16]:


hist = model.fit(x_train, y_train_one_hot, 
           batch_size=256, epochs=10, validation_split=0.3 )


# In[ ]:


model.evaluate(x_test, y_test_one_hot)[1]


# In[17]:


#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

