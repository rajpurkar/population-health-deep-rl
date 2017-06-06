from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from predict import *

# Parameters
learning_rate = 0.0001
training_epochs = 50
batch_size = 100
display_step = 1
eval_step = 1

# Network Parameters
n_classes = 1

def get_fake_dataset(batch_size, width, height, depth):
 	batch_x = np.random.rand(batch_size, width, height, depth)
 	batch_y = np.random.randint(n_classes, size=(batch_size))
 	batch_y = np.array(batch_y)
 	return batch_x, batch_y


def get_dataset(file):
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


def get_3d_data(file = "data/KE_2015_MIS_05232017_1847_107786/kepr7hdt/KEPR7HFL.DTA.CSV-processed.csv-postprocessed.csv"):
    return get_X_Y_from_data(file)


def get_next_batch(input_X, input_y, input_weights, i, batch_size):
    batch_X = input_X[i*batch_size: i*batch_size+ batch_size]
    batch_y = input_y[i*batch_size: i*batch_size+ batch_size]
    batch_weights = input_weights[i*batch_size: i*batch_size + batch_size]
    return batch_X, batch_y, batch_weights

def cnn_network(x):
    out = x
    out = layers.convolution2d(out, num_outputs=10, kernel_size=[1, 1], activation_fn=tf.nn.relu, stride=1)
    out = layers.flatten(out)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, n_classes, activation_fn=None)
    out = tf.reshape(out, shape=(-1,))
    return out

# tf Graph input
def add_placeholder(width, height, depth):
    x = tf.placeholder("float", [None, width, height, depth])
    y = tf.placeholder("float32", [None, ])
    loss_weights = tf.placeholder("float32", [None, ])
    return x, y, loss_weights

def build(input_dim):
    width = input_dim[1]
    height = input_dim[2]
    depth = input_dim[3]

    #get placeholders:
    x, y, loss_weights = add_placeholder(width, height, depth)

    # Construct model
    pred = cnn_network(x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=pred, multi_class_labels=y, weights=loss_weights))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    return x, y, loss_weights, pred, cost, optimizer, init

def eval(gt_y, output):
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    score = sklearn.metrics.precision_recall_fscore_support(gt_y, output, average='binary')
    return score

def run(x, y, weights, pred, cost, optimizer, init):

    total_batch = input_X.shape[0] / batch_size
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for i in range(total_batch - 1):
                batch_x, batch_y, batch_weights = get_next_batch(input_X, input_y, input_weights, i, batch_size)
                train_pred, _, c = sess.run([pred, optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              weights: batch_weights})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:
                predictions = tf.nn.sigmoid(train_pred)
                score = eval(batch_y, predictions.eval())
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
                print("Train f1 score: ", score)
            if epoch % eval_step == 0:
                # Test model
                evalx, evaly, evalweights = get_next_batch(input_X, input_y, input_weights, total_batch - 1, batch_size)
                predictions = tf.nn.sigmoid(pred)
                output = predictions.eval({x: evalx, y: evaly})
                score = eval(evaly, output)
                print("Eval f1 score: ", score)
        print("Optimization Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    input_X, input_y = get_3d_data(args.file)
    print ("Inputx shape: " , input_X.shape)
    print ("Inputy shape: " , input_y.shape)
    input_weights = [k+1 for k in input_y]
    x, y, weights, pred, cost, optimizer, init = build(input_X.shape)
    run(x, y, weights, pred, cost, optimizer, init)
