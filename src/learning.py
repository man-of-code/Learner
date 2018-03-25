# epsilon : decide take what we have or go for new actions 0.1 10% random action
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback

import game
import numpy as num
import random
import csv
import os.path
import timeit
MAX_IN = 3 
TUNE = True
GAMMA = 0.9

def train_net(model, neural):

    filename = str(neural['network'][0]) + '-' + str(neural['network'][1]) + '-' + \
    		str(neural['batchSize']) + '-' + str(neural['buffer'])

    observe = 1000  
    epsilon = 1
    train_frames = 100000  
    batchSize = neural['batchSize']
    buffer = neural['buffer']

    max_distance = 0
    ob_distance = 0
    t = 0
    data_collect = []
    replay = []
    loss_log = []
    game_state = game.GameState()
    _, state = game_state.frame_step((2))
    start_time = timeit.default_timer()

    while t < train_frames:

        t += 1
        ob_distance += 1

        if random.random() < epsilon or t < observe:
            action = num.random.randint(0, 3) 
        else:
            qval = model.predict(state, batch_size=1)
            action = (num.argmax(qval)) 

        reward, new_state = game_state.frame_step(action)

        replay.append((state, action, reward, new_state))

        if t > observe:

            if len(replay) > buffer:
                replay.pop(0)

            minibatch = random.sample(replay, batchSize)

            # Get training values. catastrophic...................................
            X_train, y_train = process_minibatch(minibatch, model)

            history = LossHistory()
            model.fit(
                X_train, y_train, batch_size=batchSize,
                nb_epoch=1, verbose=0, callbacks=[history]
            )
            loss_log.append(history.losses)

        # Update the starting state with S'.
        state = new_state

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1.0/train_frames)

        if reward == -500:
            data_collect.append([t, ob_distance])

            if ob_distance > max_distance:
                max_distance = ob_distance

            tot_time = timeit.default_timer() - start_time
            fps = ob_distance / tot_time

            print("Max Score: %d at %d\tepsilon %f\t(%d)\t" %
                  (max_distance, t, epsilon, ob_distance))
            ob_distance = 0
            start_time = timeit.default_timer()

        if t % 25000 == 0:
            model.save_weights('models/' + filename + '-' +
                               str(t) + '.h5',
                               overwrite=True)
            print("Model Saved %s - %d" % (filename, t))

    log(filename, data_collect, loss_log)

def log(filename, data_collect, loss_log):

    with open('Data/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('Data/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

# epsilon greedy implementation.
def process_minibatch(minibatch, model):
    mb_len = len(minibatch)
    old_states = num.zeros(shape=(mb_len, 3))
    actions = num.zeros(shape=(mb_len,))
    rewards = num.zeros(shape=(mb_len,))
    new_states = num.zeros(shape=(mb_len, 3))
    for i, m in enumerate(minibatch):
        old_state_m, action_m, reward_m, new_state_m = m
        old_states[i, :] = old_state_m[...]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[...]

    old_qvals = model.predict(old_states, batch_size=mb_len)
    new_qvals = model.predict(new_states, batch_size=mb_len)

    maxQs = num.max(new_qvals, axis=1)
    y = old_qvals
    non_term_inds = num.where(rewards != -500)[0]
    term_inds = num.where(rewards == -500)[0]

    y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (GAMMA * maxQs[non_term_inds])
    y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

    X_train = old_states
    y_train = y
    return X_train, y_train

def launch_learn(neural):
    filename = str(neural['network'][0]) + '-' + str(neural['network'][1]) + '-' + \
    str(neural['batchSize']) + '-' + str(neural['buffer'])
    print("Trying %s" % filename)
    if not os.path.isfile('Data/loss_data-' + filename + '.csv'):
        open('Data/loss_data-' + filename + '.csv', 'a').close()
        print("Initializing Test...")
        model = neural_net(MAX_IN, neural['network'])
        train_net(model, neural)
    else:
        print("Done Testing....")

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def neural_net(num_sensors, neural, load=''):
    model = Sequential() 

    # First layer.
    model.add(Dense(
        neural[0], init='lecun_uniform', input_shape=(num_sensors,)
    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer with relu function.
    model.add(Dense(neural[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer with linear function.
    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms) # optimizer:algo to distribute weights

    if load:
        model.load_weights(load)

    return model
    
if __name__ == "__main__":
    if TUNE:
        neurals_list = []
        network_neural = [[164, 150], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]
        # test different values...............................................
        for network_neurals in network_neural:
            for batchSize in batchSizes:
                for buffer in buffers:
                    neural = {
                        "batchSize": batchSize,
                        "buffer": buffer,
                        "network": network_neurals
                    }
                    neurals_list.append(neural)

        for neurals_set in neurals_list:
            launch_learn(neurals_set)

    else:
        network_neurals = [128, 128]
        neural = {
            "batchSize": 64,
            "buffer": 50000,
            "network": network_neurals
        }
        model = neural_net(MAX_IN, network_neurals)
        train_net(model, neural)