import tensorflow as tf    
import numpy as np 
input_data = np.array([[0,0],[0,1],[1,0],[1,1]])  # XOR input
output_data = np.array([[0],[1],[1],[0]])  # XOR output

n_input = tf.placeholder(tf.float32, shape=[None, 2], name="n_input")
n_output = tf.placeholder(tf.float32, shape=[None, 1], name="n_output")

hidden_nodes = 8

b_hidden = tf.Variable(tf.random_normal([hidden_nodes]), name="hidden_bias")
W_hidden = tf.Variable(tf.random_normal([2, hidden_nodes]), name="hidden_weights")
hidden = tf.nn.relu(tf.matmul(n_input, W_hidden) + b_hidden)

W_output = tf.Variable(tf.random_normal([hidden_nodes, 1]), name="output_weights")  # output layer's weight matrix
output = tf.sigmoid(tf.matmul(hidden, W_output))  # calc output layer's activation

cross_entropy = tf.square(n_output - output)  # simpler, but also works

loss = tf.reduce_mean(cross_entropy)  # mean the cross_entropy
optimizer = tf.train.AdamOptimizer(0.001)  # take a gradient descent for optimizing with a "stepsize" of 0.001
train = optimizer.minimize(loss)  # let the optimizer train

init = tf.global_variables_initializer()

sess = tf.Session()  # create the session and therefore the graph
sess.run(init)  # initialize all variables  

for epoch in range(0, 21000):
    # run the training operation
    cvalues = sess.run([train, loss, W_hidden, b_hidden, W_output],
                       feed_dict={n_input: input_data, n_output: output_data})

    if epoch % 100 == 0:
        print("")
        print("step: {:>3}".format(epoch))
        print("loss: {}".format(cvalues[1]))

print("")
print("input: {} | output: {}".format(input_data[0], sess.run(output, feed_dict={n_input: [input_data[0]]})))
print("input: {} | output: {}".format(input_data[1], sess.run(output, feed_dict={n_input: [input_data[1]]})))
print("input: {} | output: {}".format(input_data[2], sess.run(output, feed_dict={n_input: [input_data[2]]})))
print("input: {} | output: {}".format(input_data[3], sess.run(output, feed_dict={n_input: [input_data[3]]})))
