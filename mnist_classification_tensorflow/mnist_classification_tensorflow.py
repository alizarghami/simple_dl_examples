#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[ ]:


saver = tf.train.Saver()
model_path = 'models/model.ckpt'


# In[ ]:


BATCH_SIZE = 100
EPOCH_NUMBER = 10

t = tf.placeholder(tf.bool, name='IfTrain_placeholder')                                                     # if we are in training phase
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='Data_placeholder')
y = tf.placeholder(dtype=tf.int32, shape=[None], name='Label_placeholder')

X_data = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
y_data = tf.data.Dataset.from_tensor_slices(y).batch(BATCH_SIZE)


# In[ ]:


X_iter = X_data.make_initializable_iterator()
X_batch = X_iter.get_next()

y_iter = y_data.make_initializable_iterator()
y_batch = y_iter.get_next()


# In[ ]:


oh_y = tf.one_hot(indices=y_batch, depth=10)

c1 = tf.layers.conv2d(inputs=X_batch, 
                      filters=32, 
                      kernel_size=[5,5], 
                      padding='same', 
                      activation=tf.nn.relu, 
                      name='CNN1')

m1 = tf.layers.max_pooling2d(inputs=c1, 
                             pool_size=[2,2], 
                             strides=2, 
                             padding='same',
                             name='MaxPool1')

c2 = tf.layers.conv2d(inputs=m1, 
                      filters=64, 
                      kernel_size=[5,5], 
                      padding='same', 
                      activation=tf.nn.relu, 
                      name='CNN2')

m2 = tf.layers.max_pooling2d(inputs=c2, 
                             pool_size=[2,2], 
                             strides=2, 
                             padding='same', 
                             name='MaxPool2')

f1 = tf.reshape(tensor=m2, shape=[-1, 7*7*64], name='Flat1')

d1 = tf.layers.dense(inputs=f1, 
                     units=1024,
                     activation=tf.nn.relu,
                     name='Dense1')

dr1 = tf.layers.dropout(inputs=d1, rate=0.4, training=t, name='Dropout1')

d2 = tf.layers.dense(inputs=dr1, 
                     units=10, 
                     name='Dense2')

loss = tf.losses.softmax_cross_entropy(onehot_labels=oh_y, logits=d2)
classes = tf.argmax(input=d2, axis=1, name='ArgMax1')

init = tf.global_variables_initializer()

#saver = tf.train.Saver()


# In[ ]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003, name='GD1')
train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name='Optimizer1')

guess = tf.nn.softmax(d2)
is_correct = tf.equal(tf.argmax(guess, 1), tf.argmax(oh_y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# In[ ]:


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
X_train = np.reshape(mnist.train.images, (-1, 28, 28, 1))
y_train = np.asarray(mnist.train.labels, dtype=np.int32)
X_test = np.reshape(mnist.test.images, (-1, 28, 28, 1))
y_test = np.asarray(mnist.test.labels, dtype=np.int32)


# In[ ]:


losses = []
train_accuracies = []
test_accuracies = []

with tf.Session() as sess:
    sess.run(init)

    # Training
    for e in tqdm(range(EPOCH_NUMBER)):
        sess.run(X_iter.initializer, feed_dict={X:X_train})
        sess.run(y_iter.initializer, feed_dict={y:y_train})
  
        while True:
            try:
                out = sess.run({'accuracy': accuracy, 'loss': loss, 'train optimizer': train_op}, feed_dict={t:True})

                losses.append(out['loss'])
                train_accuracies.append(out['accuracy'])
            except:
                break
    saver.save(sess, model_path)
                
    # Evaluation            
    sess.run(X_iter.initializer, feed_dict={X:X_test})
    sess.run(y_iter.initializer, feed_dict={y:y_test})
  
    while True:
        try:
            out = sess.run({'accuracy': accuracy, 'loss': loss}, feed_dict={t:False})
            test_accuracies.append(out['accuracy'])
        except:
            break


# In[ ]:


ave_loss = []
ave_acc = []
for i in range(100,len(losses)):
    ave_loss.append(np.mean(losses[i-100:i]))
    ave_acc.append(np.mean(train_accuracies[i-100:i]))


# In[ ]:


fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(ave_loss)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.subplot(1,2,2)
plt.plot(ave_acc)
plt.xlabel('Batch')
plt.ylabel('Accuracy')


print('Average test accuracy is %{:.2}'.format(100*np.mean(test_accuracies)))


# In[ ]:




