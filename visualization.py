#visualizations

## Dependencies
#import numpy as np
import pandas as pd 
import geopandas as gpd
import contextily as cx

import matplotlib.pyplot as plt
#import geopy.distance as geo
import imageio
import os
import ast

#%%/
#methods
#make line graphs
def rewardgraphs(reward_list, name):
    plt.figure()
    plt.plot(reward_list)
    plt.ylabel("Reward")
    plt.xlabel("Episodes*MaxIterations")
    plt.xlabel("Iterations")
    plt.savefig(name + ".png")
    plt.close()
    
#save gif images
def gifimage(w, ML_type, filenames):
    plt.savefig(ML_type + "_" + f'{w}.png')
    plt.close()
    filename = ML_type + "_" + f'{w}.png'
    filenames.append(filename)

#built gif
def buildgif(filenames, finalfilename):
    with imageio.get_writer(finalfilename, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

#calls the previously defined gif methods to plot graphs and make gifs
def gif_all(origin, A, B, initial_loc, loc_list, filenames, image_name, gif_name):
    # convert the list of [[x1,y1], [x2, y2], ...] into a list of [x1, x2...] and [y1, y2...]
    xloc = []
    yloc = []
    loc_lit = ast.literal_eval(initial_loc)
    xloc_int = []
    yloc_int = []
    for j in range(len(loc_lit)):
        xloc_int.append(loc_lit[j][0])
        yloc_int.append(loc_lit[j][1])
    xloc.append(xloc_int)
    yloc.append(yloc_int)
    
    for i in range(len(loc_list)):
        loc_lit = ast.literal_eval(loc_list[i])
        xloc_int = []
        yloc_int = []
        for j in range(len(loc_lit)):
            xloc_int.append(loc_lit[j][0])
            yloc_int.append(loc_lit[j][1])
        xloc.append(xloc_int)
        yloc.append(yloc_int)
        
    for i in range(len(xloc)):
        store = gpd.GeoDataFrame(geometry=gpd.points_from_xy([origin[1]], [origin[0]], crs="EPSG:4269"))
        cs = gpd.GeoDataFrame(geometry=gpd.points_from_xy(yloc[i], xloc[i], crs="EPSG:4269"))
        
        plt.figure(figsize = (10, 6))
        ax = plt.subplot()
        plt.ylim([A[0], B[0]])
        plt.xlim([A[1], B[1]])
        cs.plot(marker = "o", color = "black", ax = ax)
        store.plot(marker = "*", color = "red", ax = ax)
        #cx.add_basemap(ax, crs="EPSG:4269")#, source= cx.providers.Stamen.Toner)
        
        gifimage(i, image_name, filenames)
        #gifimage(i, "results/gifimagesbackup_all/deep_all", filenames)
        #gifimage(i, "results/gifimagesbackup_step/deep_step", filenames)
    
    buildgif(filenames, gif_name)
    

# final map of CS locations for best reward
def finalmap(origin, loc_list, A, B, bufferdist=5000): #buffer distance = range of drone in meters
    
    xloc = []
    yloc = []
    loc_lit = ast.literal_eval(loc_list)
    for j in range(len(loc_lit)):
        xloc.append(loc_lit[j][0])
        yloc.append(loc_lit[j][1])

    cs = gpd.GeoDataFrame(geometry=gpd.points_from_xy(yloc, xloc, crs="EPSG:4269"))
    store = gpd.GeoDataFrame(geometry=gpd.points_from_xy([origin[1]], [origin[0]], crs="EPSG:4269"))
    
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    plt.ylim([A[0], B[0]])
    plt.xlim([A[1], B[1]])
    cs.plot(marker = "o", color = "blue", ax = ax, label = "charging stations")
    store.plot(marker = "*", color = "red", ax = ax, label = "origin")
    cx.add_basemap(ax, crs="EPSG:4269")#, source= cx.providers.Stamen.Toner)
    
    #plot bufferring
    cs_buffer = cs.to_crs('+proj=cea').buffer(bufferdist).to_crs("EPSG:4269")
    cs_buffer.plot(alpha=.4, ax = ax, label = "CS range")
    
    #plot numbers for each charging station
    for i in range(len(cs_buffer)):
        plt.text(cs_buffer.centroid.x[i], cs_buffer.centroid.y[i], str(i))
    
    plt.legend()
    
    plt.savefig('results/finalmap.png')
    plt.close()
    
    ### this was a test for new reward function get_reward_AREA()
    #cs_buffer2 = cs_buffer.to_crs({"proj" : "cea"})
    #area_sum = cs_buffer2[0]
    #for i in range(len(cs_buffer2)-1):
    #    area_sum = area_sum.union(cs_buffer2[i+1])
    
    #plot the same figure but with probability as a background
    allegheny_censustracts = pd.read_csv("data/allegheny_censustracts.csv")
    import shapely.wkt
    censusgeo = []
    for i in range(len(allegheny_censustracts)):
        censusgeo.append(shapely.wkt.loads(allegheny_censustracts['geometry'][i]))
    allegheny_censustracts = gpd.GeoDataFrame(allegheny_censustracts, crs = 'EPSG:4269', geometry = censusgeo) 
    
    allegheny_censustracts['pvals'] = allegheny_censustracts['demand_scaled']/sum(allegheny_censustracts['demand_scaled'])
    
    variable = 'pvals'#'demand_scaled'
    vmin, vmax = 0.0, 1.1
    fig, ax = plt.subplots(1, figsize=(10, 6))
    allegheny_censustracts.plot(column=variable, cmap= 'Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = [] #empty array for the data range
    cbar = fig.colorbar(sm) #add the colorbar to the figure
    #plt.ylim([A[0], B[0]])
    #plt.xlim([A[1], B[1]])
    cs.plot(marker = "o", color = "blue", ax = ax, label = "charging stations")
    store.plot(marker = "*", color = "red", ax = ax, label = "origin")
    plt.legend()
    
    plt.savefig('results/finalmap_withprob.png')
    plt.close()
    
    #plot the deliveries, with the unsuccessful deliveries a different colour
    from shapely.geometry import Point
    demand = pd.read_csv("data/demand_county_1000.csv")
    householdgeo = [Point(xy) for xy in zip(demand['lon'], demand['lat'])]
    demand_geodf = gpd.GeoDataFrame(demand, crs = 'EPSG:4269', geometry = householdgeo)
    
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    plt.ylim([A[0], B[0]])
    plt.xlim([A[1], B[1]])
    cs.plot(marker = "o", color = "blue", ax = ax, label = "charging stations")
    store.plot(marker = "*", color = "red", ax = ax, label = "origin")
    demand_geodf.plot(ax = ax, markersize = 10, color = 'black', marker = 'o', label = 'deliveries')
    cx.add_basemap(ax, crs="EPSG:4269")
    plt.legend()
    
    plt.savefig('results/finalmap_withdeliveries.png')
    plt.close()
 
def main():

     # read in CSV files and other info
     rewards = pd.read_csv("results/Reward_Locations_deep.csv")
     AB_latlon = pd.read_csv(r"data/AB_latlon.csv")
     A = ast.literal_eval(AB_latlon.loc[0, 'location'])
     B = ast.literal_eval(AB_latlon.loc[1, 'location'])
     origin = [40.41101822870703, -79.91054922095948] 

     # pull out relevant series
     reward_list_all = list(rewards['reward'])
     locations_pre_list_all = list(rewards['Location'])
     locations_post_list_all = list(rewards['New_Location'])
     actions_list_all = list(rewards['Action'])

     reward_list_step = [0]
     locations_pre_list_step = []
     locations_post_list_step = []
     actions_list_step = []

     for i in range(len(reward_list_all)):
         if reward_list_all[i] > reward_list_step[i]: 
             reward_list_step.append(reward_list_all[i])
             locations_pre_list_step.append(locations_pre_list_all[i])
             locations_post_list_step.append(locations_post_list_all[i])
             actions_list_step.append(actions_list_all[i])
         else:
             reward_list_step.append(reward_list_step[-1])
    
     #'''
     # plot line graph for all and step reward reward
     rewardgraphs(reward_list_all, "results/DQN_allrewardgraph")
     rewardgraphs(reward_list_step, "results/DQN_steprewardgraph")
     
     #make gifs
     filenames_all = []
     gif_all(origin, A, B, locations_pre_list_all[0], locations_post_list_all, 
             filenames_all, "results/deep_all", "results/deep_all_csloc.gif")
     
     filenames_step = []
     gif_all(origin, A, B, locations_pre_list_step[0], locations_post_list_step, 
             filenames_step, "results/deep_step", "results/deep_step_csloc.gif")
     #'''     
     finalmap(origin, locations_post_list_step[-1], A, B, bufferdist=5000)
 
if __name__ == "__main__":
     main()