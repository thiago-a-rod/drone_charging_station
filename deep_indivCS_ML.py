import numpy as np
import pandas as pd
import charging_stations as obj
import ast
#import reward
import geopy.distance as geo
import charging_stations

import gym
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
from datetime import datetime
import reward as r

class CharStation_agent:
    # def __init__(self, S):
    def __init__(self, row, col, cs_location):
        self.location = (row, col)
        self.lat_lon_location = cs_location
        self.reward = 0

    # def convert_latlon(self, S):
    def convert_latlon(self, S, A, B):
        self.lat_lon_location = obj.converttolatlon(self.location[0], self.location[1], S.shape[0], S.shape[1], A, B)

    def encode(self, S):
        i = self.location[0]
        i *= S.shape[0]
        i += self.location[1]
        return i

    def move(self, direction, S):
        row = self.location[0]
        col = self.location[1]

        if direction == 0:  # up
            if (row != 0):
                self.location = (row - 1, col)
        elif direction == 1:  # down
            if (row != S.shape[0] - 1):
                self.location = (row + 1, col)
        elif direction == 2:  # left
            if (col != 0):
                self.location = (row, col - 1)
        elif direction == 3:  # right
            if (col != S.shape[1] - 1):
                self.location = (row, col + 1)
        elif direction == 4:  # stayput
            pass


def initialize_matrices(nrow, ncol):
    R = np.zeros((nrow, ncol))
    S = np.zeros((nrow, ncol)) #states
    return R, S


def create_nodes_dict(nrow, ncol):
    nodes = {}
    for i in range(nrow*ncol):
        gridNS = i // nrow
        gridEW = i % ncol
        nodes[i] = obj.converttolatlon(gridNS, gridEW, nrow, ncol)
    return nodes


def get_distances_cs(nodes, origin, cs_locations):
    df = pd.DataFrame()
    distances = []
    for i in range(len(nodes)):
        distances.append(geo.distance(origin, nodes[i]).km)
    df["d_origin"] = distances

    for cs in cs_locations:
        d_cs = []
        for i in range(len(nodes)):
            d_cs.append(geo.distance(cs, nodes[i]).km)
        df["d" + str(cs)] = d_cs
    return df


def select_random_cs_inrange(cs_locations, origin, drone_max_range, nodes):
    nodes_distance = get_distances_cs(nodes, origin, cs_locations)
    index = []
    columns = list(nodes_distance.columns)
    for i in range(len(nodes_distance)):
        for col in columns:
            if nodes_distance.loc[i, col] <= drone_max_range:
                index.append(i)
    index = list(set(index))
    drop = []
    for i in index:
        for cs in cs_locations:
            if nodes[i] == cs:
                drop.append(i)
    index = list(set(index) - set(drop))
    random_cs_inrange = np.random.choice(index, size=1)[0]
    return random_cs_inrange


def create_cs_locations(cs_locations, ncs, nrow, ncol, drone_max_range, nodes, origin):
    random_cs_inrange = select_random_cs_inrange(cs_locations, origin, drone_max_range, nodes)
    if len(cs_locations) < ncs:
        row = random_cs_inrange // nrow
        col = random_cs_inrange % ncol
        cs_locations.append(nodes[random_cs_inrange])
        return cs_locations, row, col

def agent(state_shape, action_shape, lr_nn, n, keras_loss, keras_opt, act):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    ## should we update this architecture or add more layers?
    learning_rate = lr_nn #0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    #model.add(keras.layers.Dense(n[0], input_shape=(state_shape,), activation='relu', kernel_initializer=init))
    #model.add(keras.layers.Dense(n[1], activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(n[0], input_shape=(state_shape,), activation= act, kernel_initializer=init))
    model.add(keras.layers.Dense(n[1], activation=act, kernel_initializer=init))
    #model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=keras_loss, optimizer=keras_opt(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def translate_action(output, Ncs, actions):
    a = len(actions)
    a_translated = []
    i = 1
    for cs in range(Ncs):
        q = output//(a**(Ncs-i))
        r = output%(a**(Ncs-i))
        a_translated.append(q)
        #print(q, r)
        output = r
        i += 1
    #print(a_translated)
    return a_translated

def get_qs(model, state):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(replay_memory, model, target_model, done, lr_train, discount, batchsize):
    learning_rate = lr_train #0.7 # Learning rate
    discount_factor = discount #0.618 #gama

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = batchsize #64 * 2
    mini_batch = random.sample(replay_memory, batch_size) ### ??
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    
    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index]) #Y_true
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q #Y_pred

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def rl_DQN_indiv(cs, ncs, demand, origin, A, B, nrow, ncol, drone_max_range, directions, max_iterations, lr_nn, n, lr_train,
                 discount, batchsize, maxlen, steps, epoch_len, keras_loss, keras_opt, act, reward_func,
                 prob_demand, bufferdist, origins): #new reward
    
    rewarddf = pd.DataFrame()
    reward_bestdf = pd.DataFrame()
    reward_alldf = pd.DataFrame()
    predictionsdf = pd.DataFrame()
    
    reward_dflist = []
    success_dflist = []
    utilization_dflist = []
    client_dflist = []
    store_dflist = []
    CS_list = []
    CS_newlist = []
    action_dflist = []
    prediction_dflist = []
    obs_dflist = []
    allobs = {}
    allobslist = []
    
    start = datetime.now()
    
    actions_dict = {0:"up", 1:"down", 2:"left", 3:'right', 4:'stayput'}
    R, S = initialize_matrices(nrow, ncol) 
    print(A, B)
    
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent(1, len(directions), lr_nn, n, keras_loss, keras_opt, act)
    print(model.summary())
    # Target Model (updated every 100 steps)
    #target_model = agent(nrow*ncol, len(directions))
    target_model = agent(1, len(directions), lr_nn, n, keras_loss, keras_opt, act)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen = maxlen) #deque(maxlen=50_000)

    steps_to_update_target_model = 0
    R1 = []
    R2 = [0]
    R2_obs = []
    R_forgif = []
    final_reward_list = []
    reward_list = []
    all_policy_list = []

    for episode in range(max_iterations):
        nodes = create_nodes_dict(nrow, ncol)  # get coordinates for all nodes in the grid

        total_training_rewards = 0
        print("Episode number:", episode)
        if episode == 0:
            cs_local = cs
        else:
            locations = [i.location for i in cs_local[0:-1]]
            new_locations, r_, c_ = create_cs_locations(locations, ncs, nrow, ncol, drone_max_range, nodes, origin)
            cs_local = charging_stations.create_cs(new_locations)
        #observation = []
        #for i in range(Ncs):
        #    ##random CS placement ### NOT NEEDED SINCE CHARGING STATION WAS ALREADY PLACED
        #    cs_locations, row, col = create_cs_locations(cs_locations, Ncs, nrow, ncol, drone_max_range,
        #                                                 nodes, origin)
        #    #observation.append(cs_locations[-1].encode(S))

        #cs = charging_stations.create_cs(cs_locations)
        observation = cs_local[-1].encode(S)
        old_location = cs_local[-1].location
        #observation = [i.encode(S) for i in cs_local] ###do I need to only take observation for cs_local[-1]?
        #old_location = [i.location for i in cs_local]

        done = False
        #print("print_check_type", type(observation), type(observation[0]), type(observation[1]))
        
        epsilon = 0 + episode*1e-3 ### edit this to allow more exploitation or exploration
        ###keeping in mind the relatively low number of episodes
        if not R2_obs:
            R2_obs.append(observation)

        while not done:
            steps_to_update_target_model += 1
            if steps_to_update_target_model % epoch_len == 0: #10 == 0: ## should be % 10 in full model
                if episode == 0:
                    cs_local = cs    
                else:
                    locations = [i.location for i in cs_local[0:-1]]
                    new_locations, r_, c_ = create_cs_locations(locations, ncs, nrow, ncol, drone_max_range, nodes, origin)
                    cs_local = charging_stations.create_cs(new_locations)
                #cs_locations = []
                #for i in range(Ncs):
                #    cs_locations, row, col = create_cs_locations(cs_locations, Ncs, nrow, ncol, drone_max_range,
                #                                                 nodes, origin)
                #cs = charging_stations.create_cs(cs_locations) 
                observation = cs_local[-1].encode(S)
                old_location = cs_local[-1].location
                #observation = [i.encode(S) for i in cs_local]
                #old_location = [i.location for i in cs_local]
                
            print(steps_to_update_target_model)
            print("Observation:", observation)

            random_number = np.random.rand() #random number 0 to 1
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number >= epsilon: # Explore
                action = []
                for i in range(1):
                    action.append(np.random.choice(directions))
                print("Action:", action)
            else: # Exploit best known action
                predicted = model.predict([observation]).flatten() #Ncs >= 1
                print("Prediction: ", predicted)
                action_index = np.argmax(predicted)
                #convert predicted node into actions
                action = translate_action(action_index, 1, directions) 
                print("Action:", action)
                
            #make sure all cs commands work for all charging stations 
            #this will have to be updated for multiple charging stations
            new_observation = []
            #cs_latlonlocations = []
            #for i in range(len(cs_local)): #use len of cs instead of len of action
                #move charging stations according to action
                #cs_local[i].move(action[i], S)    


            cs_local[-1].move(action[0], S)
            new_observation = cs_local[-1].encode(S)
            new_location = cs_local[-1].location
            #new_observation = [i.encode(S) for i in cs_local]
            #new_location = [i.location for i in cs_local]
            
            #need a compute_DQN_reward function separate from compute_reward
            #compute a single reward given all charging stations
            ### why does reward go to zero after adding > 2 charging stations?
            if reward_func == "area":
                reward = r.get_reward_AREA(cs_local, prob_demand, bufferdist, origins, bufferdist/1000)
                utilization = None
                routes_df = pd.DataFrame()
                routes_df['client'] = None
                routes_df['store'] = None
                success = 0
            elif reward_func == "success":
                #reward = r.get_reward_shallow(cs, demand) ### NB I had to change this? get_reward_success does not exist?
                reward, utilization, routes_df, success = r.get_reward_success(cs_local, demand)
                success = 0
            elif reward_func == "utilization":
                reward, utilization, routes_df, success = r.get_reward_utilization(cs_local, demand)

            reward_dflist.append(reward)
            utilization_dflist.append(utilization)
            client_dflist.append(list(routes_df.client))
            store_dflist.append(list(routes_df.store))
            success_dflist.append(success)
            #prediction_dflist.append(predicted)
            obs_dflist.append(observation)
            
            #POLICY
            #for each node, reshape into an observation, predict the action, save in a list

            if (steps_to_update_target_model == 1) or (steps_to_update_target_model == epoch_len):
                policy_list = []
                for i in range(len(nodes)):
                    #row, col = obj.converttogrid(nodes[i], nrow, ncol) #convert to row col
                    #obs = (row*S.shape[0])+col #encode
                    p = model.predict([i]).flatten()
                    action_index = np.argmax(p.copy())
                    #action = translate_action(action_index, 1, directions)
                    policy_list.append(action_index)
                all_policy_list.append(policy_list)
            else:
                all_policy_list.append(all_policy_list[-1])

            R1.append(reward)
            if reward > R2[-1]: 
                R2.append(reward)
                R2_obs.append(observation)

            else:
                R2.append(R2[-1])
                #R_forgif.append(R)
            print("Reward:", reward)
            replay_memory.append([observation, action, reward, new_observation, done])
            
            CS_list.append(old_location) #CS_list.append(observation)
            action_dflist.append(action)
            CS_newlist.append(new_location) #CS_newlist.append(new_observation)

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % steps == 0 or done: #4 == 0 or done:
                train(replay_memory, model, target_model, done, lr_train, discount, batchsize)

            observation = new_observation
            old_location = new_location
            total_training_rewards += reward # no need to sum, single value, no longer a list
            print("TTR:", total_training_rewards)
            
            if steps_to_update_target_model >= epoch_len:#10: #length of epoch ## should be 100 in full model
                done = True

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1
                
                if steps_to_update_target_model >= epoch_len: # 10: #length of epoch ## should be 100 in full model
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
            
    #end = datetime.now()
    
    #timetaken = end - start
    #print("TIME TAKEN: " + str(timetaken)) #0:33:23.710360 #1:09:19.525174 for 50 iterations
    
    rewarddf.insert(0, "reward", reward_dflist)
    #rewarddf.insert(1, "Location", CS_latlon_list) #
    rewarddf.insert(1, "Location", CS_list)
    rewarddf.insert(2, "Action", action_dflist)
    #rewarddf.insert(3, "New_Location", CS_latlon_newlist) #
    rewarddf.insert(3, "New_Location", CS_newlist)
    rewarddf.insert(4, "utilization", utilization_dflist)
    rewarddf.insert(5, 'client', client_dflist)
    rewarddf.insert(5, 'store', store_dflist)
    rewarddf.insert(6, 'success', success_dflist)
    #rewarddf.insert(7, 'prediction', prediction_dflist)
    rewarddf.insert(8, 'policy', all_policy_list)
    rewarddf.to_csv(r"results_testing/Reward_Locations_deep.csv")
    
    predictionsdf.insert(0, "Location", CS_list)
    predictionsdf.insert(1, "observation", obs_dflist)
    predictionsdf.insert(2, "New_Location", CS_newlist)
    #predictionsdf.insert(3, 'prediction', prediction_dflist)
    predictionsdf.insert(3, 'action', action_dflist)
    predictionsdf.insert(4, 'policy', all_policy_list)
    predictionsdf.to_csv(r"results_testing/Prediction_Locations_deep.csv")
    
    #reward_alldf.insert(0, "reward", R1)
    #reward_alldf.to_csv(r"results_testing/Reward_Locations_deep_ALL.csv")
    
    #reward_bestdf.insert(0, "reward", R2)
    #reward_bestdf.to_csv(r"results_testing/Reward_Locations_deep_best.csv")
    
    ###should I return the last CS or the best CS?
    #return CS_newlist[-1] ### last CS #observation, R1, R2, R2_obs, R_forgif, R
    j = R1.index(max(R1))
    return CS_newlist[j], predictionsdf ### best CS

def cs_placement(demand, ncs, nrow, ncol, max_iteration, max_iterations, lr_nn, n, lr_train, discount, batchsize, maxlen, steps, epoch_len, keras_loss, keras_opt, act, reward_func, prob_demand, bufferdist, origins):
    #max_iterations, lr_nn, n, lr_train, discount, batchsize, maxlen, steps, epoch_len, keras_loss, keras_opt, act, reward_func[2], prob_demand, bufferdist, origins
    action = {0: "up", 1: "down", 2: "left", 3: 'right', 4: 'stayput'}
    directions = [0, 1, 2, 3, 4]

    origin = [40.41101822870703, -79.91054922095948]  # Waterfront
    AB_latlon = pd.read_csv('data/AB_latlon.csv')
    A = ast.literal_eval(AB_latlon.loc[0, 'location'])
    B = ast.literal_eval(AB_latlon.loc[1, 'location'])
    nodes = create_nodes_dict(nrow, ncol) # get coordinates for all nodes in the grid

    drone_max_range = 5
    R, S = initialize_matrices(nrow, ncol)
    all_cs_locations = []
    all_rewards = []
    best_cs_locations = [0]
    best_rewards = [0]

    cs = []
    #np.random.seed(1990)
    while len(cs) < ncs:
        cs_locations = [i.location for i in cs]
        all_cs_locations.append(cs_locations)
        all_rewards.append(r.get_reward_shallow(cs, demand)) ### first time, this gives us without any charging stations
        cs_locations, row, col = create_cs_locations(cs_locations, ncs, nrow, ncol, drone_max_range, nodes, origin)
        cs = charging_stations.create_cs(cs_locations)
        
        #deep
        #for i in range(max_iteration):
        cs_newlist, predictions = rl_DQN_indiv(cs, ncs, demand, origin, A, B, nrow, ncol, drone_max_range, directions, 
                                               max_iterations, lr_nn, n, lr_train, discount, batchsize, maxlen, steps, epoch_len,
                                               keras_loss, keras_opt, act, reward_func, prob_demand, bufferdist, origins)
        #print("CS_newlist!!: ", cs_newlist)
        cs_locations[-1] = cs_newlist
        cs = charging_stations.create_cs(cs_locations)
        #cs[-1] = charging_stations.create_cs(cs_newlist[0])
        print("cs:", cs)
        print("cs_locations:", cs_locations)
        
        #for i in range(max_iteration):
            #shallow ###unused
            #forward = np.random.choice(directions) 
            #cs[-1].move(forward, S)
        
    cs_locations = [i.location for i in cs]
    all_cs_locations.append(cs_locations)
    all_rewards.append(r.get_reward_shallow(cs, demand))
        #if all_rewards[-1] > best_rewards[-1]:
        #    best_rewards.append(all_rewards[-1])
        #    best_cs_locations.append(cs_locations)

        #cs[-1].location = best_cs_locations[-1][-1]

    df = pd.DataFrame()
    df['all_cs_locations'] = all_cs_locations #should be len = len(ncs)+1
    df['all_rewards'] = all_rewards #should be len = len(ncs)+1

    df.to_csv("results_testing/rewards_locations.csv", index=False)
    #print(all_rewards)
    
'''   
    #POLICY #option 2
    #for each thing we have observed, calculate the reward if 
    dictionary = predictions.set_index('observation')['Location'].to_dict()
    obs_set = set(predictions['observation'])
    obs_list = np.sort(list(obs_set))
    action_list = []
    for i in predictions['action']:
        action_list.append(int(i[0]))
    
    location_string = []
    for i in predictions['Location']:
        location_string.append(str(i))
    arrays = [predictions['observation'], location_string]
    index = pd.MultiIndex.from_arrays(arrays, names=('Obs', 'Loc'))    
    df = pd.DataFrame(action_list, index=index)
    policy = df.groupby(level=0).last() ##TODO
    #then just need to match obs to location using dictionary and we can use this location for plotting?
    location_list = []
    for i in obs_list:
        location_list.append(dictionary[i])
    policy.insert(1, "Location", location_list)
    policy.to_csv(r"results_testing/policy.csv")
    #currently this is only for the locations that we observe, what about all the others? #TODO
'''

def main():
    origin = [40.41101822870703, -79.91054922095948]  # Waterfront

    demand = pd.read_csv("data/demand_county_1000.csv")
    demand['origin_lat'] = origin[0]
    demand['origin_lon'] = origin[1]

    print(demand)
    # Get the cs_locations todo: we're randomly generating it now, change to import it from somewhere
    # cs_locations should be a list with [lat, lon] of each charging station
    n_cs = 1 ###should be 10
    nrow = 30
    ncol = 30
    max_iteration = 2 ###should be 200
    epoch_len = 50 #50

    max_iterations = int(max_iteration/n_cs) #20 #episodes
    
    lr_nn= 0.001
    n = [12, 12] #[24, 12]
    lr_train = 0.7
    discount = 0.618 #0.9
    batchsize = 64 #64*2
    maxlen = 50_000
    steps = 4

    drone_max_range = 5
    
    keras_loss = tf.keras.losses.Huber()
    keras_opt = tf.keras.optimizers.Adam
    act = 'relu'
    
    AB_latlon = pd.read_csv("data/AB_latlon.csv")
    A = ast.literal_eval(AB_latlon.loc[0, 'location'])
    B = ast.literal_eval(AB_latlon.loc[1, 'location'])
    reward_func = "success" #= ["area",'success','utilization']
    allegheny_censustracts = pd.read_csv("data/allegheny_censustracts.csv")
    allegheny_censustracts['pvals'] = allegheny_censustracts['demand_scaled']/sum(allegheny_censustracts['demand_scaled'])
    prob_demand = allegheny_censustracts

    bufferdist = drone_max_range*1000
    origins = [origin]
    start = datetime.now()
    cs_placement(demand, n_cs, nrow, ncol, max_iteration, max_iterations, lr_nn, n, lr_train, discount, batchsize, maxlen, steps, epoch_len,
                 keras_loss, keras_opt, act, reward_func, prob_demand, bufferdist, origins)
    end = datetime.now()
    duration = end - start
    duration = duration.total_seconds() / 60
    print("Duration: %.2f min"%(duration))

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()