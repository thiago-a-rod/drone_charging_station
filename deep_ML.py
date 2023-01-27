"""
This module computes Reward Matrix (R) using a deep Q network (DQN)
"""
import numpy as np
import pandas as pd
import ast
import shallow_ML
import charging_stations
import routing
#import visualization as vis
#import reward_shallow
#import reward_deep
#import reward_global
import reward as r
import geopy.distance as geo

import gym
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
from datetime import datetime

np.float32
#RANDOM_SEED = 5
#tf.random.set_seed(RANDOM_SEED)

#%#%#%# NOTES/TIPS #%#%#%#
# search for ### to find places I need to update
#do i need to update S anywhere? no
#make sure the NN model is doing what we want it to do/has architecture we want
        # use model.summary()

def initialize_matrices(nrow, ncol):
    R = np.zeros((nrow, ncol))
    S = np.zeros((nrow, ncol)) #states
    return R, S

def agent(state_shape, action_shape, lr_nn, n, keras_loss, keras_opt, act):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    ### should we update this architecture or add more layers?
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
    

#def rl_DQN(Ncs, M, nrow, ncol, directions, max_iterations): #old reward
def rl_DQN(Ncs, demand, origin, A, B, nrow, ncol, drone_max_range, directions, max_iterations, lr_nn, n, lr_train,
           discount, batchsize, maxlen, steps, epoch_len, keras_loss, keras_opt, act, reward_func,
           prob_demand, bufferdist, origins): #new reward
    
    rewarddf = pd.DataFrame()
    reward_bestdf = pd.DataFrame()
    reward_alldf = pd.DataFrame()
    
    reward_dflist = []
    success_dflist = []
    utilization_dflist = []
    client_dflist = []
    store_dflist = []
    CS_list = []
    CS_newlist = []
    action_dflist = []
    
    start = datetime.now()
    
    actions_dict = {0:"up", 1:"down", 2:"left", 3:'right', 4:'stayput'}
    R, S = initialize_matrices(nrow, ncol) 
    print(A, B)
    
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent(Ncs, len(directions)**Ncs, lr_nn, n, keras_loss, keras_opt, act)
    print(model.summary())
    # Target Model (updated every 100 steps)
    #target_model = agent(nrow*ncol, len(directions))
    target_model = agent(Ncs, len(directions)**Ncs, lr_nn, n, keras_loss, keras_opt, act)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen = maxlen) #deque(maxlen=50_000)

    steps_to_update_target_model = 0
    R1 = []
    R2 = [0]
    R2_obs = []
    R_forgif = []
    final_reward_list = []
    reward_list = []

    for episode in range(max_iterations):
        total_training_rewards = 0
        print("Episode number:", episode)
        #multiple CS
        cs_locations = []
        #cs = []
        #observation = []
        nodes = shallow_ML.create_nodes_dict(nrow, ncol)  # get coordinates for all nodes in the grid

        for i in range(Ncs):
            ##random CS placement
            # row = np.random.randint(0, nrow)
            # col = np.random.randint(0, ncol)
            #cs_locations.append(charging_stations.converttolatlon(row, col, nrow, ncol))
            #cs_locations.append([40.41101822870703, -79.91054922095948])
            cs_locations, row, col = shallow_ML.create_cs_locations(cs_locations, Ncs, nrow, ncol, drone_max_range,
                                                                    nodes, origin)
            #observation.append(cs_locations[-1].encode(S))

        cs = charging_stations.create_cs(cs_locations)
        observation = [i.encode(S) for i in cs]
        old_location = [i.location for i in cs] ###TODO

        done = False
        #print("print_check_type", type(observation), type(observation[0]), type(observation[1]))
        
        epsilon = 0 + episode*1e-3
        if not R2_obs:
            R2_obs.append(observation)

        while not done:
            steps_to_update_target_model += 1
            if steps_to_update_target_model % epoch_len == 0:#10 == 0: ### should be % 10 in full model
                cs_locations = []
                #observation = []
                # for i in range(Ncs):
                #     ##random CS placement
                #     row = np.random.randint(0, nrow)
                #     col = np.random.randint(0, ncol)
                #     cs_locations.append(charging_stations.converttolatlon(row, col, nrow, ncol))
                #     #observation.append(cs_locations[-1].encode(S))
                for i in range(Ncs):
                    cs_locations, row, col = shallow_ML.create_cs_locations(cs_locations, Ncs, nrow, ncol,
                                                                            drone_max_range,
                                                                            nodes, origin)
                cs = charging_stations.create_cs(cs_locations) 
                observation = [i.encode(S) for i in cs]
                old_location = [i.location for i in cs] ###TODO
                
            print(steps_to_update_target_model)
            print("Observation:", observation)

            random_number = np.random.rand() #random number 0 to 1
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number >= epsilon: # Explore
                action = []
                for i in range(Ncs):
                    action.append(np.random.choice(directions))
                #action = np.random.choice(directions) #select random direction - for all NcS!
                print("Action:", action)
            else: # Exploit best known action
                predicted = model.predict([observation]).flatten() #Ncs >= 1
                #predicted = model.predict(np.asarray(observation)).flatten()
                #print(" Predicted", predicted)
                action_index = np.argmax(predicted)
                #convert predicted node into actions
                action = translate_action(action_index, Ncs, directions) 
                print("Action:", action)
                
            #make sure all cs commands work for all charging stations 
            #this will have to be updated for multiple charging stations
            new_observation = []
            #cs_latlonlocations = []
            for i in range(len(cs)): #use len of cs instead of len of action
                #move charging stations according to action
                cs[i].move(action[i], S)    
            new_observation = [i.encode(S) for i in cs]
            new_location = [i.location for i in cs] ###TODO ???
            
            #need a compute_DQN_reward function separate from compute_reward
            #compute a single reward given all charging stations
            ### give this function the right inputs
            if reward_func == "area":
                reward = r.get_reward_AREA(cs, prob_demand, bufferdist, origins, bufferdist/1000)
                utilization = None
                routes_df = pd.DataFrame()
                routes_df['client'] = None
                routes_df['store'] = None
                success = 0
            elif reward_func == "success":
                reward, utilization, routes_df, success = r.get_reward_success(cs, demand)
            elif reward_func == "utilization":
                reward, utilization, routes_df, success = r.get_reward_utilization(cs, demand)

            reward_dflist.append(reward)
            utilization_dflist.append(utilization)
            client_dflist.append(list(routes_df.client))
            store_dflist.append(list(routes_df.store))
            success_dflist.append(success)

            R1.append(reward)
            if reward > R2[-1]: 
                R2.append(reward)
                R2_obs.append(observation)
                #update R
                # for i in range(len(cs)):
                #     row_final, col_final = charging_stations.converttogrid(cs[i].location, nrow, ncol)
                #     R[row_final][col_final] = reward
                # R_forgif.append(R)
            else:
                R2.append(R2[-1])
                #R_forgif.append(R)
            print("Reward:", reward)
            replay_memory.append([observation, action, reward, new_observation, done])
            
            CS_list.append(old_location) #CS_list.append(observation) ###TODO
            action_dflist.append(action)
            CS_newlist.append(new_location) #CS_newlist.append(new_observation) ###TODO

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % steps == 0 or done: #4 == 0 or done:
                train(replay_memory, model, target_model, done, lr_train, discount, batchsize)

            observation = new_observation
            total_training_rewards += reward # no need to sum, single value, no longer a list
            print("TTR:", total_training_rewards)
            
            if steps_to_update_target_model >= epoch_len:#10: #length of epoch ### should be 100 in full model
                done = True

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1
                #final_reward_list.append(total_training_rewards) #sum of reward in the epoch to print
                #final_reward_list.append(max(reward_list)) #max of the reward in the epoch to print
                #final_reward_list = R1 #plot all of the reward for every i in every epoch

                if steps_to_update_target_model >= epoch_len: # 10: #length of epoch ### should be 100 in full model
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
            
    end = datetime.now()
    
    timetaken = end - start
    print("TIME TAKEN: " + str(timetaken)) #0:33:23.710360 #1:09:19.525174 for 50 iterations
    
    #decode observations in CS_list
    # CS_latlon_list = []
    # CS_latlon_newlist = []
    # for i in range(len(CS_list)):
    #     row = CS_list[i]//S.shape[0]
    #     col = CS_list[i]%S.shape[1]
    #     CS_latlon_list.append(charging_stations.converttolatlon(row, col, nrow, ncol))
    #     row2 = CS_newlist[i]//S.shape[0]
    #     col2 = CS_newlist[i]%S.shape[1]
    #     CS_latlon_newlist.append(charging_stations.converttolatlon(row2, col2, nrow, ncol))
    
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
    rewarddf.to_csv(r"results/Reward_Locations_deep.csv")
    
    reward_alldf.insert(0, "reward", R1)
    reward_alldf.to_csv(r"results/Reward_Locations_deep_ALL.csv")
    
    reward_bestdf.insert(0, "reward", R2)
    reward_bestdf.to_csv(r"results/Reward_Locations_deep_best.csv")
    
    return observation, R1, R2, R2_obs, R_forgif, R

def main():
    #NB total number of CS movements = max_iterations * epoch_len
    #I got the OOM error again at episode 100 (4/10 in epoch) with 64*2 batch size amd [24, 12] nodes
    
    M = 1000
    Ncs = 10#10 #20 #number of charging stations
    directions = [0, 1, 2, 3, 4]
    max_iterations = 20 #20 #episodes
    nrow = 30 #10
    ncol = 30 #20
    
    lr_nn= 0.001
    n = [12, 12] #[24, 12]
    lr_train = 0.7
    discount = 0.618 #0.9
    batchsize = 64 #64*2
    maxlen = 50_000
    steps = 4
    epoch_len = 50
    drone_max_range = 10
    
    keras_loss = tf.keras.losses.Huber()
    keras_opt = tf.keras.optimizers.Adam
    act = 'relu'
    
    origin = [40.41101822870703, -79.91054922095948] # Waterfront mall
    demand = pd.read_csv("data/demand_county_" + str(M) + ".csv")
    demand['origin_lat'] = origin[0]
    demand['origin_lon'] = origin[1]

    demand_out_range = demand.copy()
    print(demand_out_range)

    for i in range(len(demand)):
        d_client = [demand.loc[i,'lat'], demand.loc[i,'lon']]
        o_store = [demand.loc[i, 'origin_lat'], demand.loc[i, 'origin_lon']]
        dist_od = geo.distance(d_client, o_store).km
        if dist_od < drone_max_range:
            demand_out_range = demand_out_range.drop(i)

    demand = demand_out_range.copy()
    
    AB_latlon = pd.read_csv("data/AB_latlon.csv")
    A = ast.literal_eval(AB_latlon.loc[0, 'location'])
    B = ast.literal_eval(AB_latlon.loc[1, 'location'])
    reward_func = ["area",'success','utilization']
    allegheny_censustracts = pd.read_csv("data/allegheny_censustracts.csv")
    allegheny_censustracts['pvals'] = allegheny_censustracts['demand_scaled']/sum(allegheny_censustracts['demand_scaled'])
    prob_demand = allegheny_censustracts

    bufferdist = drone_max_range*1000
    origins = [origin]
    
    ### output inclides the observation list that needs to be decoded before it can be used in future
    finalstate, reward_list, final_reward_list, final_reward_obs, \
    R_forgif, R = rl_DQN(Ncs, demand, origin, A, B, nrow, ncol, drone_max_range, directions,
                         max_iterations, lr_nn, n, lr_train, discount, batchsize, maxlen, steps, epoch_len,
                         keras_loss, keras_opt, act, reward_func[2], prob_demand, bufferdist, origins) #new reward
    print(finalstate)
    print(final_reward_obs)
    #print("len", len(R_forgif))

    #R_final = R_forgif[-1] #save the final version of the R-matrix from the gif as a new
    
    #best location(s)
    beststate = final_reward_obs[-1]
    print(beststate) ### check the code above is visualizing in the right place #[53, 52, 62, 62]
    cs_lat = []
    cs_lon = []
    cs_latlon = []
    
    for i in range(len(beststate)):
        cs_lat.append(beststate[i] // ncol)
        cs_lon.append(beststate[i] % nrow)
        cs_latlon.append([cs_lat[-1], cs_lon[-1]])
    print(cs_latlon) ### check the code above is visualizing in the right place #[[2, 3], [2, 2], [3, 2], [3, 2]]
    
    return reward_list, final_reward_list, final_reward_obs, cs_lat, cs_lon, cs_latlon, R

if __name__ == "__main__":
    #https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()