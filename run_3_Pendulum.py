import gym
from RL_brain_3_Dueling_DQN import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 10000
ACTION_SPACE = 25

sess = tf.Session()
with tf.variable_scope('natural'):
    natural_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=False)

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    acc_r = [0]
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        next_state, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        acc_r.append(reward + acc_r[-1])  # accumulated reward

        RL.store_transition(observation, action, reward, next_state)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        observation = next_state
        total_steps += 1
    return RL.cost_his, acc_r

c_natural, r_natural = train(natural_DQN)
c_dueling, r_dueling = train(dueling_DQN)