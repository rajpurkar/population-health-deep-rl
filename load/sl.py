from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from predict import *

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
total_batch = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 295
n_classes = 2


def get_dataset(file = "../../mis-data/kepr7hdt/KEPR7HFL.DTA.CSV-processed.csv-postprocessed.csv"):
    df = pd.read_csv(file, low_memory=False)
    y_column_name = 'Final result of malaria from blood smear test'
    columns = list(df.columns)
    num_features = len(columns)
    input_X = []
    input_y = []
    cols = list(list(findsubsets(columns, num_features))[0])
    ignore_phrase_columns = [y_column_name.lower(), 'presence of species:', 'rapid test', 'number']
    cols = filter(lambda col: not any(phrase.lower() in col.lower() for phrase in ignore_phrase_columns), cols)
    input_X, feature_names = get_X_cols(df, cols)
    input_y = get_Y_col(df, y_column_name)
    xdims = input_X.shape
    return input_X, input_y

def get_next_batch(input_X, input_y, i, batch_size):
    batch_X = input_X[i*batch_size: i*batch_size+ batch_size]
    batch_y = input_y[i*batch_size: i*batch_size+ batch_size]
    return batch_X, batch_y

# Create model
def network(x, weights, biases):
    # Hidden layer with RELU activation
    x = layers.flatten(x)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("int64", [None, ])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

input_X, input_y = get_dataset()

# Construct model
pred = network(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch - 1):
            batch_x, batch_y = get_next_batch(input_X, input_y, i, batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    predictions = tf.argmax(pred, 1)
    print(y)
    #y_argmax = tf.argmax(y, 1)
    correct_prediction = tf.equal(predictions, y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    evalx, evaly = get_next_batch(input_X, input_y, total_batch, batch_size)
    print("Accuracy:", accuracy.eval({x: evalx, y: evaly}))
    array_pred = predictions.eval({x: evalx, y: evaly})
