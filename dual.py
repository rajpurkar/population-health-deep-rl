import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN


class SimpleQN(DQN):
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        state_shape = list(self.env.observation_space.shape)
        batch_size = None
        image_width = state_shape[0]
        image_height = state_shape[1]
        channels = state_shape[2] * self.config.state_history
        self.s = tf.placeholder(tf.uint8,
            (batch_size, image_width, image_height, channels))
        self.a = tf.placeholder(tf.int32, (batch_size))
        self.r = tf.placeholder(tf.float32, (batch_size))
        self.sp = tf.placeholder(tf.uint8,
            (batch_size, image_width, image_height, channels))
        self.done_mask = tf.placeholder(tf.bool, (batch_size, ))
        self.lr = tf.placeholder(tf.float32, batch_size)
        self.y = tf.placeholder(tf.int32, (batch_size))


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network
        """
        q_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_q_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        q_variables = sorted(q_variables, key=lambda x: x.name)
        target_q_variables = sorted(target_q_variables, key=lambda x: x.name)
        ops = []
        for i, var in enumerate(q_variables):
            ops.append(tf.assign(target_q_variables[i], var))

        self.update_target_op = tf.group(*ops, name="update_target_op")


    def add_loss_op(self, q, target_q, pred):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        """
        if self.config.k_class:
            pred_loss = tf.losses.softmax_cross_entropy(self.y, logits=pred)
        else:
            pred_loss = tf.losses.sigmoid_cross_entropy(self.y, logits=pred)
        # action_taken = tf.argmax(q)
        num_actions = self.env.action_space.n
        qsamp = self.r + self.config.gamma * tf.reduce_max(target_q, axis=1) * tf.cast()
        qs = tf.reduce_sum(tf.one_hot(self.a, num_actions)*q, axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(qsamp, qs)) + tf.reduce_mean(pred_loss)
        """
        num_actions = self.env.action_space.n
        q_samp = self.r + (1.0 - tf.cast(self.done_mask, tf.float32)) * self.config.gamma * tf.reduce_max(target_q, axis=1)
        q_sa = tf.reduce_sum(tf.multiply(q, tf.one_hot(self.a, num_actions, axis=1)), axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, q_sa))


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads_and_vars = optimizer.compute_gradients(self.loss, variables)

        if self.config.grad_clip:
            capped_gv = []
            for g, v in grads_and_vars:
                if g is not None:
                    capped_gv.append((tf.clip_by_norm(g, self.config.clip_val), v))
            grads_and_vars = capped_gv

        self.train_op = optimizer.apply_gradients(grads_and_vars)

        grads = [g for g, v in grads_and_vars]

        self.grad_norm = tf.global_norm(grads)


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
            if self.config.k_class is True:
                state_shape = list(self.env.observation_space.shape)
                k = state_shape[0] + 1
                pred = layers.fully_connected(common, k, activation_fn=None)
            else:
                pred = layers.fully_connected(common, 1, activation_fn=tf.sigmoid)
            out = layers.fully_connected(common, num_actions, activation_fn=None)
        return out, pred


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    from configs.test_env import config
    env = EnvTest((10, 1, 1), config)
    model = SimpleQN(env, config)
    model.run()
