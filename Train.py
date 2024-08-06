from fog_env import Offload
from RL_brain import DeepQNetwork
import numpy as np
import random
import tensorflow as tf

np.set_printoptions(threshold=np.inf)

def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

def reward_fun(delay, max_delay, unfinish_indi):
    penalty = - max_delay * 2
    if unfinish_indi:
        reward = penalty
    else:
        reward = - delay
    return reward

def train(iot_RL_list, NUM_EPISODE):
    RL_step = 0

    for episode in range(NUM_EPISODE):
        print(episode)
        print(iot_RL_list[0].epsilon)
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, size=[env.n_time, env.n_iot])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])

        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for iot_index in range(env.n_iot):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state),
                            'reward': 0}
                history[time_index].append(tmp_dict)

        observation_all = np.zeros([env.n_iot, env.n_features])
        lstm_state_all = np.zeros([env.n_iot, env.n_lstm_state])
        for iot_index in range(env.n_iot):
            observation_all[iot_index, :] = env.state_init[iot_index, :]
            lstm_state_all[iot_index, :] = env.lstm_state_init[iot_index, :]

        observation_all_ = np.zeros([env.n_iot, env.n_features])
        lstm_state_all_ = np.zeros([env.n_iot, env.n_lstm_state])
        reward_all = np.zeros(env.n_iot)
        action_all = np.zeros(env.n_iot)

        reward_indicator = np.zeros([env.n_time, env.n_iot])
        process_delay = env.max_delay * np.ones([env.n_time, env.n_iot])
        done = False
        env.state_reset()

        for time_index in range(env.n_time):
            for iot_index in range(env.n_iot):
                history[time_index][iot_index]['observation'] = observation_all[iot_index, :]
                history[time_index][iot_index]['lstm'] = lstm_state_all[iot_index, :]

            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all[iot_index, :])

            for iot_index in range(env.n_iot):
                action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation_all[iot_index, :])

            observation_all_, lstm_state_all_, reward_all, done = env.step(bitarrive, action_all, time_index)

            for iot_index in range(env.n_iot):
                history[time_index][iot_index]['observation_'] = observation_all_[iot_index, :]
                history[time_index][iot_index]['lstm_'] = lstm_state_all_[iot_index, :]
                history[time_index][iot_index]['reward'] = reward_fun(reward_all[iot_index], process_delay[time_index, iot_index], done)

            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].store_transition(
                    s=history[time_index][iot_index]['observation'],
                    lstm_s=history[time_index][iot_index]['lstm'],
                    a=action_all[iot_index],
                    r=history[time_index][iot_index]['reward'],
                    s_=history[time_index][iot_index]['observation_'],
                    lstm_s_=history[time_index][iot_index]['lstm_'])

            for iot_index in range(env.n_iot):
                if reward_indicator[time_index, iot_index] == 0:
                    iot_RL_list[iot_index].do_store_reward(episode, time_index, history[time_index][iot_index]['reward'])
                    iot_RL_list[iot_index].do_store_action(episode, time_index, action_all[iot_index])
                    if not done:
                        iot_RL_list[iot_index].do_store_delay(episode, time_index, process_delay[time_index, iot_index])
                    reward_indicator[time_index, iot_index] = 1

            RL_step += 1

            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            if done:
                break

if __name__ == "__main__":
    NUM_IOT = 50
    NUM_FOG = 5
    NUM_EPISODE = 1000
    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    iot_RL_list = list()
    for iot in range(NUM_IOT):
        iot_RL_list.append(DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                        learning_rate=0.01,
                                        reward_decay=0.9,
                                        e_greedy=0.99,
                                        replace_target_iter=200,  # each 200 steps, update target net
                                        memory_size=500,  # maximum of memory
                                        ))

    train(iot_RL_list, NUM_EPISODE)
    print('Training Finished')
