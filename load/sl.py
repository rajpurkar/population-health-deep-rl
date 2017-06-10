from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.layers as layers
from predict import *
from utils.survey_env import SurveyEnv
from configs.survey_env import config as SurveyEnvConfig
from utils.dataset import Dataset
from utils.exploration import LinearSchedule


class Config():
    def __init__(self, epochs=100, batch_size=32, n_classes=2, reg=0, display_step=1, eval_step=1,
                 weighted_loss=False, num_train_examples=1000, num_test_examples=2000,
                 keep_prob=0.9, lr_begin=0.0025, lr_end=0.0005):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.reg = reg
        self.display_step = display_step
        self.eval_step = eval_step
        self.weighted_loss = weighted_loss
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples
        self.keep_prob = keep_prob
        self.lr_begin = lr_begin
        self.lr_end = lr_end


def get_fake_dataset(batch_size, width, height, depth, n_classes=2):
    batch_x = np.random.rand(batch_size, width, height, depth)
    batch_y = np.random.randint(n_classes, size=(batch_size))
    batch_y = np.array(batch_y)
    return batch_x, batch_y


def cnn_network(config, x, train_placeholder):
    out = x
    # out = layers.convolution2d(out, num_outputs=survey_config.max_steps, kernel_size=[1, 1], stride=1,
    #                            activation_fn=None)
    # out = layers.batch_norm(out, is_training=train_placeholder, updates_collections=None)
    # out = tf.nn.relu(out)
    # out = layers.dropout(out, keep_prob=config.dropout, is_training=train_placeholder)
    out = layers.flatten(out)
    for i in xrange(0, 4):
        out = layers.fully_connected(out, survey_config.max_steps, activation_fn=None)
        out = tf.nn.relu(out)
        out = layers.dropout(out, keep_prob=config.keep_prob, is_training=train_placeholder)
    out = layers.fully_connected(out, config.n_classes, activation_fn=None)
    return out


# tf Graph input
def add_placeholder(width, height, depth):
    x = tf.placeholder("float", [None, width, height, depth], name="x")
    y = tf.placeholder("int64", [None, ], name="y",)
    train_placeholder = tf.placeholder(dtype=tf.bool, name="istraining")
    lr_placeholder = tf.placeholder(tf.float32)
    return x, y, train_placeholder, lr_placeholder


def build(config, input_dim):
    width = input_dim[0]
    height = input_dim[1]
    depth = input_dim[2]

    #get placeholders:
    x, y, train_placeholder, lr_placeholder = add_placeholder(width, height, depth)

    # Construct model
    pred = cnn_network(config, x, train_placeholder)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(y, pred))
    l2_loss = 0.
    for var in tf.trainable_variables():
        l2_loss += tf.nn.l2_loss(var)
    cost += config.reg * l2_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=lr_placeholder).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    return x, y, train_placeholder, lr_placeholder, pred, cost, optimizer, init


def get_state(env, split='train'):
    total_reward = 0.

    state = env.reset(split=split)

    for f in xrange(0, survey_config.max_steps):
        state, r, done = env.step(f)
        total_reward += r
        assert not done

    return state, total_reward


def get_final_reward(env):
    _, r, done = env.step(env.feature_length)
    assert done
    return r


def get_batch(env, split='train', batch_size=32):
    batch_X = []
    batch_reward = []
    batch_y = []
    for i in xrange(0, batch_size):
        example_X, reward = get_state(env, split=split)

        batch_X.append(example_X)
        batch_reward.append(reward)
        final_reward = get_final_reward(env)
        if final_reward == survey_config.correctAnswerReward:
            batch_y.append(0)
        else:
            batch_y.append(1)

    return np.asarray(batch_X), np.asarray(batch_reward), np.asarray(batch_y)


def run_epoch(epoch_num, env, x, y, is_training, pred, lr_placeholder, lr_schedule, cost, optimizer, sess, num_examples, split='train'):
    avg_cost = 0.
    avg_reward = 0.
    # Loop over all batches
    all_preds = []
    num_batches = num_examples/config.batch_size
    for i in range(num_batches):
        batch_X, batch_reward, batch_y = get_batch(env, split, config.batch_size)

        q_reward = batch_reward
        if split == 'train':
            training = True
        else:
            training = False

        train_pred = sess.run([pred], feed_dict={x: batch_X, is_training: training})
        train_pred = train_pred[0]
        predictions = tf.arg_max(tf.nn.softmax(train_pred), dimension=1).eval()

        # print(q_reward)
        pred_rewards = np.zeros_like(batch_y)
        right_idxes = np.where(batch_y == predictions)[0]
        if right_idxes.shape[0] > 0:
            pred_rewards[right_idxes] = survey_config.correctAnswerReward
        wrong_idxes = np.where(batch_y != predictions)[0]
        if wrong_idxes.shape[0] > 0:
            pred_rewards[wrong_idxes] = survey_config.wrongAnswerReward

        all_preds.extend([1 if batch_y[i] == predictions[i] else 0 for i in xrange(predictions.shape[0])])
        if split == 'train':
            t = epoch_num * num_batches + i
            lr_schedule.update(t)
            # print("Epoch {:d} batch {:d} t={:d} Learning rate: {:.7f}".format(epoch_num, i, t, lr_schedule.epsilon))
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_X,
                                                          y: batch_y,
                                                          is_training: training,
                                                          lr_placeholder: lr_schedule.epsilon})
            avg_cost += np.sum(c)

        # Compute average loss
        # print(pred_rewards)
        avg_reward += np.sum(q_reward + pred_rewards)

    if split == 'train':
        avg_cost /= num_examples

    avg_reward /= num_examples

    acc = np.mean(all_preds)
    std = np.std(all_preds)

    return avg_cost, avg_reward, acc, std


def run(env, x, y, train_placeholder, lr_placeholder, pred, cost, optimizer, init, output_file, config=None):
    if config is None:
        config = Config()

    total_steps = config.epochs * config.num_train_examples / config.batch_size
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, total_steps)
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(config.epochs):

            avg_train_cost, avg_train_reward, train_acc, train_std = run_epoch(epoch, env, x, y, train_placeholder, pred,
                                                                               lr_placeholder, lr_schedule,
                                                                               cost, optimizer,
                                                                               sess, config.num_train_examples)
            # Display logs per epoch step
            if (epoch+1) % config.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_train_cost))
                print("Average train reward: {:4f} ".format(avg_train_reward))
                print("Train accuracy: {:.4f}\tStd: {:.4f}".format(train_acc, train_std))

                output_file.write("Epoch: {:04d}\ttraining cost={:.9f}\n".format(epoch+1, avg_train_cost))
                output_file.write("Average train reward: {:.4f}\n".format(avg_train_reward))
                output_file.write("Train accuracy: {:.4f}\tStd: {:.4f}\n".format(train_acc, train_std))

            if (epoch+1) % config.eval_step == 0:
                # Test model
                _, avg_test_reward, test_acc, test_std = run_epoch(epoch, env, x, y, train_placeholder, pred,
                                                                   lr_placeholder, lr_schedule,
                                                                   cost, optimizer, sess,
                                                                   config.num_test_examples, split='test')
                print("Average test reward: {:.4f}".format(avg_test_reward))
                print("Test accuracy: {:.4f}\tStd: {:.4f}".format(test_acc, test_std))
                output_file.write("Average test reward: {:.4f}\n".format(avg_test_reward))
                output_file.write("Test accuracy: {:.4f}\tStd: {:.4f}\n".format(test_acc, test_std))

            print("-----------------------")
            output_file.write("-----------------------")
            output_file.flush()

        print("Optimization Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    parser.add_argument('max_steps', type=int)
    parser.add_argument('--stats-dir', type=str, default='stats/')
    parser.add_argument('--run-id', type=str, default='sl')
    args = parser.parse_args()

    if not os.path.exists(args.stats_dir):
        os.makedirs(args.stats_dir)

    results_file = os.path.join(args.stats_dir, args.run_id + "_results.txt")
    results_file = open(results_file, 'w')
    path_log_file = os.path.join(args.stats_dir, args.run_id + "_paths.txt")
    path_log_file = open(path_log_file, 'w')
    config = Config()

    # input_weights = [k+1 for k in input_y]
    sampler = Dataset(args.file)
    survey_config = SurveyEnvConfig()
    survey_config.max_steps = args.max_steps
    env = SurveyEnv(survey_config, sampler, log_file=path_log_file)
    x, y, train_placeholder, lr, pred, cost, optimizer, init = build(config, sampler.state_shape)
    run(env, x, y, train_placeholder, lr, pred, cost, optimizer, init, results_file, config=config)

    results_file.close()
    path_log_file.close()
