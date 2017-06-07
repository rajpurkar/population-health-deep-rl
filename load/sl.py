from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.layers as layers
from predict import *
from utils.survey_env import SurveyEnv
from configs.survey_env import config as SurveyEnvConfig
from utils.dataset import Dataset


class Config():
    def __init__(self, epochs=50, batch_size=100, n_classes=2,
                 learning_rate=5e-4, reg=1e-1, display_step=1, eval_step=1,
                 weighted_loss=False, num_train_examples=6000, num_test_examples=100):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.reg = reg
        self.display_step = display_step
        self.eval_step = eval_step
        self.weighted_loss = weighted_loss
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples


def get_fake_dataset(batch_size, width, height, depth, n_classes=2):
    batch_x = np.random.rand(batch_size, width, height, depth)
    batch_y = np.random.randint(n_classes, size=(batch_size))
    batch_y = np.array(batch_y)
    return batch_x, batch_y


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
    width = input_dim[0]
    height = input_dim[1]
    depth = input_dim[2]

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


def get_state(env, split='train'):
    total_reward = 0.
    state = env.reset(split=split)

    for f in xrange(0, env.feature_length):
        state, r, done = env.step(f)
        total_reward += r
        assert not done

    return state, total_reward


def get_prediction_reward(env, prediction):
    _, r, done = env.step(env.feature_length + prediction[0])
    assert done
    return r


def run_epoch(env, x, y, pred, cost, optimizer, sess, num_examples, split='train'):
    avg_cost = 0.
    avg_reward = 0.
    # Loop over all batches

    for i in range(num_examples):
        example_X, reward = get_state(env, split=split)
        example_X = np.expand_dims(example_X, axis=0)
        q_reward = reward
        train_pred = sess.run([pred], feed_dict={x: example_X})
        train_pred = train_pred[0]
        predictions = tf.arg_max(tf.nn.softmax(train_pred), dimension=1)
        output = predictions.eval()
        pred_reward = get_prediction_reward(env, output)
        if pred_reward == env.config.correctAnswerReward:
            example_y = output
        else:
            example_y = 1 - output

        if split == 'train':
            _, c = sess.run([optimizer, cost], feed_dict={x: example_X, y: example_y})
            avg_cost += c

        # Compute average loss
        avg_reward += q_reward + pred_reward

    if split == 'train':
        avg_cost /= num_examples

    avg_reward /= num_examples

    return avg_cost, avg_reward


def run(env, x, y, pred, cost, optimizer, init, config=None, overfit=True):
    if config is None:
        config = Config()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(config.epochs):

            avg_train_cost, avg_train_reward = run_epoch(env, x, y, pred, cost, optimizer,
                                                         sess, config.num_train_examples)
            # Display logs per epoch step
            if (epoch+1) % config.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_train_cost))
                print("Average train reward: {:.2f} ".format(avg_train_reward))

            if (epoch+1) % config.eval_step == 0:
                # Test model
                _, avg_test_reward = run_epoch(env, x, y, pred, cost, optimizer, sess,
                                                           config.num_test_examples, split='test')
                print("Average test reward: {:.2f}".format(avg_test_reward))

        print("Optimization Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    parser.add_argument('--overfit', action='store_true', help='If specified, tries to overfit to a single batch of '
                                                               'training data')
    args = parser.parse_args()

    config = Config()
    if args.overfit:
        config.reg = 0
        config.epochs = 250
        config.eval_step = 10000  # don't display stats on test set if overfitting

    # input_weights = [k+1 for k in input_y]
    sampler = Dataset(args.file)
    survey_config = SurveyEnvConfig()
    env = SurveyEnv(survey_config, sampler)
    x, y, pred, cost, optimizer, init = build(config, sampler.state_shape)
    print(sampler.state_shape)
    run(env, x, y, pred, cost, optimizer, init, config=config, overfit=args.overfit)
