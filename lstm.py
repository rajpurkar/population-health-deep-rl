import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from predQ_network import SimpleQN


from configs.q3_nature import config


class LstmQN(SimpleQN):
    
    def add_loss_op(self, q, target_q, pred):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        action_taken = tf.argmax(q)
        num_actions = self.env.action_space.n
        # self.done_mask = tf.equal(action_taken, num_actions - 1)
        # pred_loss = tf.cast(self.done_mask, tf.float32)*tf.losses.sigmoid_cross_entropy(self.y, logits=pred)
        pred_loss = tf.losses.sigmoid_cross_entropy(self.y, logits=pred)
        qsamp = self.r + self.config.gamma * tf.reduce_max(target_q, axis=1)
        qs = tf.reduce_sum(tf.one_hot(self.a, num_actions)*q, axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(qsamp, qs)) + tf.reduce_mean(pred_loss)


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n
        common = state

        with tf.variable_scope(scope, reuse):
            common = layers.flatten(common)
            common = layers.fully_connected(common, 24, activation_fn=tf.nn.relu)
            common = layers.fully_connected(common, 48, activation_fn=tf.nn.relu)
            pred = layers.fully_connected(common, 1, activation_fn=tf.sigmoid)
            out = layers.fully_connected(common, num_actions, activation_fn=None)
        return out, pred


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((5, 1, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = SimpleQN(env, config)
    model.run(exp_schedule, lr_schedule)
