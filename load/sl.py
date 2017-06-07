from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.layers as layers
from predict import *
from utils.survey_env import SurveyEnv
from configs.survey_env import config as SurveyEnvConfig


class Config():
    def __init__(self, epochs=50, batch_size=100, n_classes=2,
                 learning_rate=5e-4, reg=1e-1, display_step=1, eval_step=1,
                 weighted_loss=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.reg = reg
        self.display_step = display_step
        self.eval_step = eval_step
        self.weighted_loss = weighted_loss


def get_fake_dataset(batch_size, width, height, depth, n_classes=2):
    batch_x = np.random.rand(batch_size, width, height, depth)
    batch_y = np.random.randint(n_classes, size=(batch_size))
    batch_y = np.array(batch_y)
    return batch_x, batch_y


def get_next_batch(env, batch_size):
    batch_X = []
    batch_y = []
    for k in xrange(0, batch_size):
        ex_x, ex_y = env.reset(get_y=True)
        batch_X.append(ex_x)
        batch_y.append(ex_y)

    return np.asarray(batch_X), np.asarray(batch_y)


def cnn_network(config, x):
    out = x
    out = layers.convolution2d(out, num_outputs=10, kernel_size=[1, 1], activation_fn=tf.nn.relu, stride=1)
    out = layers.flatten(out)
    out = layers.fully_connected(out, 10, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, config.n_classes, activation_fn=None)
    return out


# tf Graph input
def add_placeholder(width, height, depth):
    x = tf.placeholder("float", [None, width, height, depth])
    y = tf.placeholder("int64", [None, ])
    return x, y


def build(config, input_dim):
    width = input_dim[1]
    height = input_dim[2]
    depth = input_dim[3]

    #get placeholders:
    x, y = add_placeholder(width, height, depth)

    # Construct model
    pred = cnn_network(config, x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(y, pred))
    l2_loss = 0.
    for var in tf.trainable_variables():
        l2_loss += tf.nn.l2_loss(var)
    cost += config.reg * l2_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    return x, y, pred, cost, optimizer, init


def get_rewards(env, gt_y, output):
    total_reward = 0.
    for name in feature_names:
        total_reward += env.reward_config.get_reward(name)

    rewards = np.ones_like(output) * total_reward
    rewards[output == gt_y] = env.reward_config.correctAnswerReward
    rewards[output != gt_y] = env.reward_config.wrongAnswerReward

    return np.mean(rewards)


def run(train_env, test_env, x, y, pred, cost, optimizer, init, config=None, overfit=True):
    if config is None:
        config = Config()

    if overfit:
        print("Trying to overfit..")
        total_batch = 20
    else:
        total_batch = train_X.shape[0] / config.batch_size

    print("Number of training batches: {:d}".format(total_batch))

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(config.epochs):
            avg_cost = 0.
            # Loop over all batches
            all_train_preds = None
            all_train_y = None

            for i in range(total_batch):
                batch_x, batch_y = get_next_batch(train_env, config.batch_size)
                train_pred, _, c = sess.run([pred, optimizer, cost], feed_dict={x: batch_x,
                                                                                y: batch_y})
                predictions = tf.arg_max(tf.nn.softmax(train_pred), dimension=1)
                output = predictions.eval()
                if all_train_preds is None:
                    all_train_preds = output
                else:
                    all_train_preds = np.concatenate([all_train_preds, output], axis=0)

                if all_train_y is None:
                    all_train_y = batch_y
                else:
                    all_train_y = np.concatenate([all_train_y, batch_y], axis=0)
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if (epoch+1) % config.display_step == 0:
                print(len(all_train_y))
                print(len(all_train_preds))
                avg_reward = get_rewards(train_env, all_train_y, all_train_preds)
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                print("Average train reward: {:.2f} ".format(avg_reward))

            if (epoch+1) % config.eval_step == 0:
                # Test model
                test_batch_X, test_batch_y = get_next_batch(test_env, survey_config.num_episodes_test)
                predictions = tf.arg_max(tf.nn.softmax(pred), dimension=1)
                output = predictions.eval({x: test_batch_X, y: test_batch_y})
                avg_reward = get_rewards(test_env, test_batch_y, output)
                print("Average test reward: {:.2f}".format(avg_reward))
        print("Optimization Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    parser.add_argument('--overfit', action='store_true', help='If specified, tries to overfit to a single batch of '
                                                               'training data')
    args = parser.parse_args()
    input_X, input_y, feature_names = get_X_Y_from_data(args.file)
    print ("Inputx shape: ", input_X.shape)
    print ("Inputy shape: ", input_y.shape)

    config = Config()
    if args.overfit:
        config.reg = 0
        config.epochs = 250
        config.eval_step = 10000  # don't display stats on test set if overfitting

    # input_weights = [k+1 for k in input_y]
    if config.weighted_loss:
        neg_count = float(len([k for k in input_y if k == 0]))
        pos_count = float(len([k for k in input_y if k == 1]))
        input_weights = [neg_count/pos_count if k == 1. else 1. for k in input_y]
    else:
        input_weights = [1.0] * len(input_y)

    train_X, train_y, test_X, test_y = split_data(0.8, input_X, input_y)
    print(train_X.shape)
    print(train_y.shape)

    survey_config = SurveyEnvConfig()
    train_env = SurveyEnv(train_X, train_y, feature_names, survey_config)
    test_env = SurveyEnv(test_X, test_y, feature_names, survey_config)
    x, y, pred, cost, optimizer, init = build(config, input_X.shape)
    run(train_env, test_env, x, y, pred, cost, optimizer, init, config=config, overfit=args.overfit)
