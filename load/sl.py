from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from predict import *

# Parameters
learning_rate = 0.001
training_epochs = 1
batch_size = 100
display_step = 1
total_batch = 10
eval_step = 1

# Network Parameters
n_classes = 2

def get_fake_dataset(batch_size, width, height, depth):
 	batch_x = np.random.rand(batch_size, width, height, depth)
 	batch_y = np.random.randint(n_classes, size=(batch_size))
 	batch_y = np.array(batch_y)
 	return batch_x, batch_y

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

def get_3d_data(file = "../../mis-data/kepr7hdt/KEPR7HFL.DTA.CSV-processed.csv-postprocessed.csv"):
    return get_X_Y_from_data(file)

def get_next_batch(input_X, input_y, i, batch_size):
    batch_X = input_X[i*batch_size: i*batch_size+ batch_size]
    batch_y = input_y[i*batch_size: i*batch_size+ batch_size]
    return batch_X, batch_y

# Create model
def cnn_network(x):
    out = x
    out = layers.convolution2d(out, num_outputs=10, kernel_size=[1,1], activation_fn=tf.nn.relu, stride=4)
    out = layers.convolution2d(out, num_outputs=64, kernel_size=[2,2], activation_fn=tf.nn.relu, stride=2)
    out = layers.convolution2d(out, num_outputs=64, kernel_size=[3,3], activation_fn=tf.nn.relu, stride=1)
    out = layers.flatten(out)
    out = layers.fully_connected(out, 512, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, n_classes, activation_fn=None)
    return out

# tf Graph input
def add_placeholder(width, height, depth):
    x = tf.placeholder("float", [None, width, height, depth])
    y = tf.placeholder("int64", [None, ])
    return x, y

def build(input_dim):
    width = input_dim[1]
    height = input_dim[2]
    depth = input_dim[3]

    #get placeholders:
    x, y = add_placeholder(width, height, depth)

    # Construct model
    pred = cnn_network(x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    return x, y, pred, cost, optimizer, init

def run(x, y, pred, cost, optimizer, init):
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for i in range(total_batch - 1):
                batch_x, batch_y = get_next_batch(input_X, input_y, i, batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            if epoch % eval_step == 0:
                # Test model
                predictions = tf.argmax(pred, 1)
                evalx, evaly = get_next_batch(input_X, input_y, total_batch - 1, batch_size)
                output = predictions.eval({x: evalx, y: evaly})
                score = sklearn.metrics.precision_recall_fscore_support(evaly, output, average='binary')
                print(score)
        print("Optimization Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    predict_from_data(args.file)
    if file is not None:
        input_X, input_y = get_3d_data(file)
    else:
        input_X, input_y = get_3d_data()
    print ("Inputx shape: " , input_X.shape)
    print ("Inputy shape: " , input_y.shape)
    x, y, pred, cost, optimizer, init = build(input_X.shape)
    run(x, y, pred, cost, optimizer, init)
