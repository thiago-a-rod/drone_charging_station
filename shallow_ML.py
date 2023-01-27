import numpy as np
import pandas as pd
import charging_stations as obj
import ast
import reward
import geopy.distance as geo
import charging_stations


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


def initialize_matrices(nrow, ncol, n_actions):
    V = np.zeros((nrow, ncol))
    R = np.zeros((nrow, ncol))
    P = np.zeros((nrow, ncol))
    Q = np.zeros((nrow*ncol, len(n_actions)))
    S = np.zeros((nrow, ncol)) #states
    return V, R, P, Q, S


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


def cs_placement(demand, ncs, nrow, ncol, max_iteration):
    action = {0: "up", 1: "down", 2: "left", 3: 'right', 4: 'stayput'}
    directions = [0, 1, 2, 3]

    origin = [40.41101822870703, -79.91054922095948]  # Waterfront
    AB_latlon = pd.read_csv('data/AB_latlon.csv')
    A = ast.literal_eval(AB_latlon.loc[0, 'location'])
    B = ast.literal_eval(AB_latlon.loc[1, 'location'])
    nodes = create_nodes_dict(nrow, ncol) # get coordinates for all nodes in the grid

    drone_max_range = 5
    V, R, P, Q, S = initialize_matrices(nrow, ncol, directions)
    all_cs_locations = []
    all_rewards = []
    best_cs_locations = [0]
    best_rewards = [0]

    cs = []
    #np.random.seed(1990)
    while len(cs) < ncs:
        cs_locations = [i.location for i in cs]
        all_cs_locations.append(cs_locations)
        all_rewards.append(reward.get_reward_shallow(cs, demand))
        cs_locations, row, col = create_cs_locations(cs_locations, ncs, nrow, ncol, drone_max_range, nodes, origin)
        cs = charging_stations.create_cs(cs_locations)
        for i in range(max_iteration):
            forward = np.random.choice(directions)
            cs[-1].move(forward, S)
            cs_locations = [i.location for i in cs]
            all_cs_locations.append(cs_locations)
            all_rewards.append(reward.get_reward_shallow(cs, demand))
            if all_rewards[-1] > best_rewards[-1]:
                best_rewards.append(all_rewards[-1])
                best_cs_locations.append(cs_locations)

        cs[-1].location = best_cs_locations[-1][-1]


    df = pd.DataFrame()
    df['all_cs_locations'] = all_cs_locations
    df['all_rewards'] = all_rewards

    df.to_csv("results/shallow_rewards_locations.csv", index=False)
    #print(all_rewards)


def main():
    origin = [40.41101822870703, -79.91054922095948]  # Waterfront

    demand = pd.read_csv("data/demand_county_1000.csv")
    demand['origin_lat'] = origin[0]
    demand['origin_lon'] = origin[1]

    print(demand)
    # Get the cs_locations todo: we're randomly generating it now, change to import it from somewhere
    # cs_locations should be a list with [lat, lon] of each charging station
    n_cs = 10
    nrow = 30
    ncol = 30
    max_iteration = 200
    cs_placement(demand, n_cs, nrow, ncol, max_iteration)






if __name__ == "__main__":
    main()