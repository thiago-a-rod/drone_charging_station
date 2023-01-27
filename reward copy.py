import routing
import numpy as np
import pandas as pd
import ast
import charging_stations
import geopandas as gpd
from datetime import datetime
import shapely.wkt
import cs_connected
import matplotlib.pyplot as plt

def get_reward(cs, demand):
    utilization, routes_df = routing.routing_simulation(cs, demand)
    success = utilization['store']
    print("sucess", success)

    del utilization['client']
    del utilization['store']
    print(utilization)
    if (len(cs) > 0):
        R = (sum(utilization.values())/(len(demand)*len(cs)))*(success/len(demand))
        R = success
        return R, utilization, routes_df, success
    else:
        return 0

def get_reward_shallow(cs, demand):
    utilization, routes_df = routing.routing_simulation(cs, demand)
    success = utilization['store']
    print("sucess", success)
    print(utilization)
    del utilization['client']
    del utilization['store']
    print(utilization)
    return success
    # if (len(cs) > 0):
    #     R = (sum(utilization.values())/(len(demand)*len(cs)))*(success/len(demand))
    #     return R
    # else:
    #     return 0


def get_reward_AREA(cs, prob_demand, bufferdist, origins, r):
    reward = 0


    censusgeo = []
    for i in range(len(prob_demand)):
        censusgeo.append(shapely.wkt.loads(prob_demand['geometry'][i]))
    prob_demand_gpd = gpd.GeoDataFrame(prob_demand, crs = 'EPSG:4269', geometry = censusgeo) 
    prob_demand_ct = prob_demand_gpd['geometry'].to_crs({"proj" : "cea"}) 
    
    #exclude_cs_not_connected #Thiago DO #NB: this needs to be an multipolygon area
    centers_connected = cs_connected.get_valid_cs(cs, origins, r)  # list with lat lon of centers of circles with radius r

    xloc = []
    yloc = []
    xloc2 = [origins[0][0]]
    yloc2 = [origins[0][1]]
    for j in range(len(centers_connected)):
        xloc.append(centers_connected[j][0])
        yloc.append(centers_connected[j][1])
    for j in range(len(cs)):
        xloc2.append(cs[j].location[0])
        yloc2.append(cs[j].location[1])
    
    cs_con = gpd.GeoDataFrame(geometry=gpd.points_from_xy(yloc, xloc, crs="EPSG:4269"))
    cs_all = gpd.GeoDataFrame(geometry=gpd.points_from_xy(yloc2, xloc2, crs="EPSG:4269"))
    
    cs_buffer = cs_con.to_crs('+proj=cea').buffer(bufferdist).to_crs("EPSG:4269")
    cs_buffer_all = cs_all.to_crs('+proj=cea').buffer(bufferdist).to_crs("EPSG:4269")
    
    cs_buffer2 = cs_buffer.to_crs({"proj" : "cea"})
    area_sum = cs_buffer2[0]
    for i in range(len(cs_buffer2)-1):
        area_sum = area_sum.union(cs_buffer2[i+1])
    for i in range(len(prob_demand)):
        reward += area_sum.intersection(prob_demand_ct[i]).area*(prob_demand['pvals'][i]/prob_demand['ALAND'][i])
        #reward += exclude_cs_not_connected.intersection(prob_demand_ct[i]).area*(prob_demand['pvals'][i]/prob_demand['ALAND'][i])
    
    #visualization to check what areas are used to calculate reward
    cs_buffer_all.plot(alpha = .4, color = "orange")
    for i in range(len(cs_buffer_all)):
        plt.text(cs_buffer_all.centroid.x[i], cs_buffer_all.centroid.y[i], str(i))
    plt.title("All Charging Stations")
    
    cs_buffer_all[1:].plot(alpha = .4, color = "red")
    plt.title("All Charging Stations + Origin")
    
    cs_buffer_plot = cs_buffer2.to_crs("EPSG:4269")
    cs_buffer_plot.plot(alpha = .4)
    plt.title("Selected Charging Stations + Origin")
        
    return reward


def main():
    origin = [40.41101822870703, -79.91054922095948]  # Waterfront
    origins = [origin]  # include all origins here
    demand = pd.read_csv("data/demand_county_1000.csv")
    demand['origin_lat'] = origin[0]
    demand['origin_lon'] = origin[1]

    print(demand)
    # Get the cs_locations todo: we're randomly generating it now, change to import it from somewhere
    # cs_locations should be a list with [lat, lon] of each charging station

    n_cs = 20
    nrow = 30
    ncol = 30

    cs_locations = []
    np.random.seed(1990)  # comment this later
    for i in range(n_cs):
        row = np.random.randint(0, nrow)
        col = np.random.randint(0, ncol)
        cs_locations.append(charging_stations.converttolatlon(row, col, nrow, ncol))

    print("Demand size = ", len(demand))
    print("Charging stations = ", n_cs)
    cs = charging_stations.create_cs(cs_locations)

    start = datetime.now()
    reward = get_reward(cs, demand)
    end = datetime.now()
    time_reward = end - start
    print("Reward: ",reward)
    print("Time for Reward: ", time_reward)
    
    allegheny_censustracts = pd.read_csv("data/allegheny_censustracts.csv")
    allegheny_censustracts['pvals'] = allegheny_censustracts['demand_scaled']/sum(allegheny_censustracts['demand_scaled'])

    r = 5 # km <-- drone range
    start = datetime.now()
    reward2 = get_reward_AREA(cs, allegheny_censustracts, 5000, origins, r)
    end = datetime.now()
    time_areareward = end - start
    print("Area Reward: ", reward2 )
    print("Time for Area Reward: ", time_areareward)

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
