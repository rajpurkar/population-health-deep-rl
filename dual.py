import tensorflow as tf
import numpy as np
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
        if self.config.k_class is True:
            self.y = tf.placeholder(tf.int32, (batch_size, self.env.prediction_space.n))
        else:
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


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
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
        out = state
        with tf.variable_scope(scope, reuse):
            out = layers.flatten(out)
            out = layers.fully_connected(out, 64)
            out = layers.fully_connected(out, 64)
            out = layers.fully_connected(out, num_actions, activation_fn=None)
        return out

    def get_prediction(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if self.config.k_class is True:
            if np.random.random() < self.config.soft_epsilon:
                return self.env.prediction_space.sample()
            else:
                return self.get_best_prediction(state)[0]
        else:
            raise NotImplementedError

    def get_best_prediction(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        predictions = self.sess.run(self.pred, feed_dict={self.s: [state]})[0]
        if self.config.k_class is True:
            return np.argmax(predictions), predictions
        else:
            raise NotImplementedError

    def get_prediction_op(self, state, scope, reuse=False):
        """
        Returns prediction for state

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope
         """
        with tf.variable_scope(scope, reuse):
            out = layers.flatten(state)
            out = layers.fully_connected(out, 64)
            if self.config.k_class is True:
                pred = layers.fully_connected(
                    out,
                    self.env.prediction_space.n,
                    activation_fn=None)
            else:
                pred = layers.fully_connected(
                    out,
                    1,
                    activation_fn=tf.sigmoid)
        return pred

    def add_pred_loss_op(self, pred, y):
        self.pred_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=pred, labels=y, name='pred_loss'))


    def add_pred_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads_and_vars = optimizer.compute_gradients(self.pred_loss, variables)

        if self.config.grad_clip:
            capped_gv = []
            for g, v in grads_and_vars:
                if g is not None:
                    capped_gv.append((tf.clip_by_norm(g, self.config.clip_val), v))
            grads_and_vars = capped_gv

        self.pred_train_op = optimizer.apply_gradients(grads_and_vars)

        grads = [g for g, v in grads_and_vars]

        self.pred_grad_norm = tf.global_norm(grads)

"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    from configs.test_env import config
    num_features = 5
    max_choices = 4
    env = EnvTest((num_features, 1, 1), max_choices, config)
    model = SimpleQN(env, config)
    model.run()
