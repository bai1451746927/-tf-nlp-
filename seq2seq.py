import tensorflow as tf
import numpy as np
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.contrib.rnn import static_rnn
from tensorflow.nn import dynamic_rnn
x=np.random.randint(1,4,size=(5,5))
print(x)
y=np.random.randint(1,4,size=(5,4))
print('y is \n',y)
y_target=np.random.randint(1,4,size=(5,4))
init=tf.global_variables_initializer()
# xy=tf.get_variable('aa',shape=[4,100])

    # x = tf.unstack(x)
    # y=tf.unstack(y)
    # tf.reset_default_graph()
encode_input=tf.placeholder(shape=[None,None],dtype=tf.int32,name='encode_input')
decode_target=tf.placeholder(shape=[None,None],dtype=tf.int32,name='encode_input')
decode_input=tf.placeholder(shape=[None,None],dtype=tf.int32,name='encode_input')
embedding=tf.Variable(tf.random_uniform([4,10],-1.0,1.0),dtype=tf.float32)#生成词汇表，前面是字符数量，后面是词嵌入大小
encode_embedding=tf.nn.embedding_lookup(embedding,encode_input)
decode_embedding=tf.nn.embedding_lookup(embedding,decode_input)
lstm_cell = LSTMCell(4)
outputs, states = dynamic_rnn(lstm_cell, encode_embedding, dtype=tf.float32)
print('states is ',states)
    # y=tf.unstack(y,4,1)/
lstm_cell2=LSTMCell(num_units=4)
logit,states2=dynamic_rnn(lstm_cell2,decode_embedding,dtype=tf.float32,initial_state=states,scope='decode_output')

print('2')
la=tf.one_hot(y_target, depth=4, dtype=tf.float32)
print(la)
pre=tf.nn.softmax(logit)
print('logit is ',logit)
print('pre is ',pre)
decoder_prediction = tf.argmax(pre, 2)
loss1=tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_target,depth=4,dtype=tf.float32),
                                                               logits=logit)
loss=tf.reduce_mean(loss1)
opmizer=tf.train.GradientDescentOptimizer(learning_rate=1.0)
op=opmizer.minimize(loss)
# pred=tf.equal(tf.arg_max(pre),tf.arg_max())
# accuacy=tf.reduce_mean(tf.cast(pred,tf.float32))
init=tf.global_variables_initializer()

sess=tf.InteractiveSession()
tf.reset_default_graph()
sess.run(init)
k=sess.run(loss,feed_dict={encode_input:x,decode_input:y,decode_target:y_target})
print('logit is',k)







