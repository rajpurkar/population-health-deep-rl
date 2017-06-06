from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from predict import *

# Parameters
training_epochs = 75
batch_size = 100
display_step = 1
eval_step = 1

# Network Parameters
n_classes = 2
learning_rate = 0.0001
reg = 0


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
    cols = list(list(findsubsets(columns, num_features))[0])
    ignore_phrase_columns = [y_column_name.lower(), 'presence of species:', 'rapid test', 'number']
    cols = filter(lambda col: not any(phrase.lower() in col.lower() for phrase in ignore_phrase_columns), cols)
    input_X, feature_names = get_X_cols(df, cols)
    input_y = get_Y_col(df, y_column_name)
    return input_X, input_y


def split_data(train_frac=0.8):
    total_data = len(input_y)
    print(total_data)
    num_train = int(train_frac * total_data)
    assert num_train > 0

    train_X = input_X[:num_train]
    train_y = input_y[:num_train]
    train_weights = input_weights[:num_train]

    test_X = input_X[num_train:]
    test_y = input_y[num_train:]
    test_weights = input_weights[num_train:]

    return train_X, train_y, train_weights, test_X, test_y, test_weights


def get_3d_data(file = "data/KE_2015_MIS_05232017_1847_107786/kepr7hdt/KEPR7HFL.DTA.CSV-processed.csv-postprocessed.csv"):
    return get_X_Y_from_data(file)


def get_next_batch(X, y, weights, i, batch_size):
    batch_X = X[i*batch_size: i*batch_size+ batch_size]
    batch_y = y[i*batch_size: i*batch_size+ batch_size]
    batch_weights = weights[i*batch_size: i*batch_size + batch_size]
    return batch_X, batch_y, batch_weights


def cnn_network(x):
    out = x
    out = layers.convolution2d(out, num_outputs=10, kernel_size=[1, 1], activation_fn=tf.nn.relu, stride=1)
    out = layers.flatten(out)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, n_classes, activation_fn=None)
    return out


# tf Graph input
def add_placeholder(width, height, depth):
    x = tf.placeholder("float", [None, width, height, depth])
    y = tf.placeholder("int64", [None, ])
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
    cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(y, pred, weights=loss_weights))
    l2_loss = 0.
    for var in tf.trainable_variables():
        l2_loss += tf.nn.l2_loss(var)
    cost += reg * l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    return x, y, loss_weights, pred, cost, optimizer, init


def eval(gt_y, output):
    score = sklearn.metrics.precision_recall_fscore_support(gt_y, output, average='binary')
    return score


def run(x, y, weights, pred, cost, optimizer, init):
    global learning_rate
    total_batch = train_X.shape[0] / batch_size
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for i in range(total_batch ):
                batch_x, batch_y, batch_weights = get_next_batch(train_X, train_y, train_weights, i, batch_size)
                train_pred, _, c = sess.run([pred, optimizer, cost], feed_dict={x: batch_x,
                                                                                y: batch_y,
                                                                                weights: batch_weights})
                # Compute average loss
                avg_cost += c / total_batch


            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                predictions = tf.arg_max(tf.nn.softmax(train_pred), dimension=1)
                output = predictions.eval()
                score = eval(batch_y, output)
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                print("Train f1 score: ", score)

            if epoch % eval_step == 0:
                # Test model
                predictions = tf.arg_max(tf.nn.softmax(pred), dimension=1)
                output = predictions.eval({x: test_X, y: test_y, weights: test_weights})
                score = eval(test_y, output)
                print("Eval f1 score: ", score)
        print("Optimization Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    input_X, input_y, _ = get_3d_data(args.file)
    print ("Inputx shape: ", input_X.shape)
    print ("Inputy shape: ", input_y.shape)

    # input_weights = [k+1 for k in input_y]
    neg_count = float(len([k for k in input_y if k == 0]))
    pos_count = float(len([k for k in input_y if k == 1]))
    input_weights = [neg_count/pos_count if k == 1. else 1. for k in input_y]

    train_X, train_y, train_weights, test_X, test_y, test_weights = split_data()
    print(train_X.shape)
    print(train_y.shape)

    x, y, weights, pred, cost, optimizer, init = build(input_X.shape)
    run(x, y, weights, pred, cost, optimizer, init)
