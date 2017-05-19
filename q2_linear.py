import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (80, 80, 4).
               - self.s: batch of states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        batch_size = None
        num_features = state_shape[0]
        input_dim = state_shape[1]
        self.s = tf.placeholder(tf.uint8, (batch_size, num_features, input_dim))
        self.a = tf.placeholder(tf.int32, (batch_size, ))
        self.r = tf.placeholder(tf.float32, (batch_size, ))
        self.sp = tf.placeholder(tf.uint8, (batch_size, num_features, input_dim))
        self.done_mask = tf.placeholder(tf.bool, (batch_size, ))
        self.lr = tf.placeholder(tf.float32, None)
        self.y = tf.placeholder(tf.int32, (batch_size, ))


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
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state

        ##############################################################
        """
        TODO: implement a fully connected with no hidden layer (linear
            approximation) using tensorflow. In other words, if your state s
            has a flattened shape of n, and you have m actions, the result of 
            your computation sould be equal to
                W s where W is a matrix of shape m x n

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 
        
        with tf.variable_scope(scope, reuse):
            out = layers.flatten(out)
            out = layers.fully_connected(out, num_actions, activation_fn=None)

        ##############################################################
        ######################## END YOUR CODE #######################

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network
        """
        q_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        q_variables = sorted(q_variables, key=lambda x: x.name)
        target_q_variables = sorted(target_q_variables, key=lambda x: x.name)
        ops = []
        for i, var in enumerate(q_variables):
            ops.append(tf.assign(target_q_variables[i], var))

        self.update_target_op = tf.group(*ops, name="update_target_op")

        ##############################################################
        ######################## END YOUR CODE #######################


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
        self.done_mask = tf.equal(action_taken, num_actions - 1)
        print "Q SHAPE", q.get_shape().ndims
        print "Pred SHAPE", pred.get_shape().ndims
        pred_loss = tf.cast(self.done_mask, tf.float32)*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=pred)
        qsamp = self.r + self.config.gamma * tf.reduce_max(target_q, axis=1)
        qs = tf.reduce_sum(tf.one_hot(self.a, num_actions)*q, axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(qsamp, qs)) + tf.reduce_mean(pred_loss)


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
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


if __name__ == '__main__':
    from configs.q3_nature import config
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
