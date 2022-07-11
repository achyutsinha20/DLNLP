#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:\\Users\\025005\\Desktop\\CNN Dataset\\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[3]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:\\Users\\025005\\Desktop\\CNN Dataset\\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[4]:


cnn = tf.keras.models.Sequential()


# In[5]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# In[6]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[7]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[8]:


cnn.add(tf.keras.layers.Flatten())


# In[9]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[10]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[11]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[12]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# In[15]:


import numpy as np
from keras.preprocessing import image
test_image = tf.keras.preprocessing.image.load_img('C:\\Users\\025005\\Desktop\\CNN Dataset\\single_prediction\\cat_or_dog_1.jpg', target_size = (64, 64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'


# In[16]:


print(prediction)


# In[ ]:




