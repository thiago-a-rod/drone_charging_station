import deep_ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
import geopy.distance as geo
import ast
import reward as REWARD
import charging_stations
plt.rcParams.update({'figure.autolayout': True, 'font.size': 14, })

def plot_analysis(df, parameter_label):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Parameter: "+parameter_label, fontsize=14)

    # successful deliveries
    sns.lineplot(x='iterations', y='success', data=df, hue='replicate', ax=ax1, legend=False)
    ax1.set_ylabel("successful deliveries", fontsize=14)
    ax1.set_xlabel("iterations", fontsize=14)
    sns.despine(top=True, right=True, ax=ax1)

    # reward
    g = sns.lineplot(x='iterations', y='reward', data=df, hue='replicate', ax=ax2, legend=False)
    ax2.set_ylabel("reward", fontsize=14)
    ax2.set_xlabel("iterations", fontsize=14)
    sns.despine(top=True, right=True, ax=ax2)



    plt.savefig("results/hyperparameters/"+parameter_label+".png")


def main():
    directions = [0,1,2,3,4]
    M = 1000
    Ncs = 10 # Number of Charging Stations
    drone_max_range = r = 10 # km
    nrow = 30  # number of rows (grid)
    ncol = 30  # number of columns (grid)
    max_iterations = 1  # number of episodes
    epoch_len = 1  # number of iterations for each episode
    lr_nn = 0.001  # Learning rate TODO: ask @Marissa, define what this learning rate do
    n = [12, 12] # TODO: ask @Marissa
    lr_train = 0.7 # TODO: ask @Marissa
    discount = 0.618
    batchsize = 64  # 64*2
    maxlen = 50_000
    steps = 4
    keras_loss = tf.keras.losses.Huber()
    keras_opt = tf.keras.optimizers.Adam
    act = 'relu'

    origin = [40.41101822870703, -79.91054922095948]
    demand = pd.read_csv("data/demand_county_" + str(M) + ".csv")
    demand['origin_lat'] = origin[0]
    demand['origin_lon'] = origin[1]
    demand_out_range = demand.copy()
    print(len(demand))

    for i in range(len(demand)):
        d_client = [demand.loc[i, 'lat'], demand.loc[i, 'lon']]
        o_store = [demand.loc[i, 'origin_lat'], demand.loc[i, 'origin_lon']]
        dist_od = geo.distance(d_client, o_store).km
        if dist_od < drone_max_range:
            demand_out_range = demand_out_range.drop(i)

    demand = demand_out_range.copy()
    print(len(demand))


    AB_latlon = pd.read_csv("data/AB_latlon.csv")
    A = ast.literal_eval(AB_latlon.loc[0, 'location'])
    B = ast.literal_eval(AB_latlon.loc[1, 'location'])

    reward_func = ["area", 'success', 'utilization']
    allegheny_censustracts = pd.read_csv("data/allegheny_censustracts.csv")
    allegheny_censustracts['pvals'] = allegheny_censustracts['demand_scaled'] / sum(
        allegheny_censustracts['demand_scaled'])
    prob_demand = allegheny_censustracts

    bufferdist = r*1000
    origins = [origin]

    time = []
    variable_id = []


    epoch_len = 100
    max_iterations = 300
    reward_functions = ['area']
    reward_func = 'success'
    replicates = 3
    Ncs = 2
    Episodes = [100]#[100, 300, 600, 1000, 1500]
    Epochs = [300]#[300, 100, 50, 30, 20]
    LAYERS = [3, 4, 5, 3, 4, 5 ]
    NODES = [24, 24, 24, 12, 12, 12]
    CHAR = [2]
    n = [12, 12]

    for var in range(len(reward_functions)):
        reward_func = reward_functions[var]
        #epoch_len = Epochs[var]
        #max_iterations = Episodes[var]
        #parameter_label = 'Ncs_' + str(CHAR[var]) + "_"+"iteration_"
        #n = [NODES[var] for _ in range(LAYERS[var])]
        #print(n)
        parameter_label = 'reward_' + str(reward_functions[var])  + "_iteration_"
        reward, success, iterations, replicate_id = [], [], [], []

        for rep in range(replicates):
            file_name = parameter_label + str(rep)
            print("=======>>>> replicate #", rep)
            start = datetime.now()
            observation, R1, R2, R2_obs, R_forgif, R = deep_ML.rl_DQN(Ncs, demand, origin, A, B, nrow, ncol,
                       drone_max_range, directions, max_iterations, lr_nn, n,
                       lr_train, discount, batchsize, maxlen, steps, epoch_len, keras_loss, keras_opt, act, reward_func,
                       prob_demand, bufferdist, origins)
            end = datetime.now()
            duration = end - start
            duration = duration.total_seconds() / 60
            df = pd.read_csv('results/Reward_Locations_deep.csv')
            df['duration'] = duration
            success_area = []
            if reward_func == "area":
                for i in range(len(df)):
                    location_area = ast.literal_eval(df.loc[i,'Location'])
                    cs = charging_stations.create_cs(location_area)
                    reward_local, utilization, routes_df, success_local = REWARD.get_reward_success(cs, demand)
                    success_area.append(success_local)
                df['success'] = success_area
            df.to_csv("results/hyperparameters/"+file_name+".csv")


if __name__ == "__main__":
    # https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()