'''
This module provides the Drone and Charging Station classes.
'''
import pandas as pd
import numpy as np
import geopy.distance as geo
import ast
import time
import DijkstraAlgo as da
from collections import Counter
import charging_stations
import multiprocess


def distances_cs(cs, R=10):
	dist = np.zeros((len(cs), len(cs)))
	for i in range(len(cs)):
		for j in range(len(cs)):
			d = geo.distance(cs[i].location, cs[j].location).km
			if d > R:
				d = np.inf
			dist[i, j] = round(d, 2)
	return dist


def distance_origin_dest(cs, dist, demand_unit, R=10):
	origin = [demand_unit.origin_lat, demand_unit.origin_lon]
	destination = [demand_unit.lat, demand_unit.lon]
	dist_origin = []
	dist_dest = []
	for i in range(len(cs)):
		d = geo.distance(cs[i].location, origin).km
		o = geo.distance(cs[i].location, destination).km
		if d > R:
			d = np.inf
		dist_origin.append(d)
		if o > R:
			o = np.inf
		dist_dest.append(o)

	if geo.distance(origin, destination).km > R:
		d_od = np.inf
	else:
		d_od = geo.distance(origin, destination).km

	D = np.c_[np.array(dist_origin), dist, dist_dest]
	D = np.vstack([np.array([0] + dist_origin +[d_od]), D, np.array([0] + dist_dest + [d_od])])

	return D


def subrouting(cs, demand, dist):
	demand.reset_index(inplace=True)
	origin = 1  # Always the first column
	destination = len(cs) + 2  # Always the last column
	labels = ['store'] + [str(i) for i in range(len(cs))] + ['client']
	fail = 0
	cs_used = []
	routes = []
	distances = []
	successful_deliveries = []
	successful_origins = []
	fail = 0
	for i in range(len(demand)):

		print("Progress: %.2f %%" % (100 * i / len(demand)), end='\r')
		D = distance_origin_dest(cs, dist, demand.loc[i, :])
		x = da.DijkstraAlgorithm()
		try:
			x.dijkstraWithPath(D, origin, destination)
			shortest_path = x.path()
			distance = x.distance()
			shortest_path_label = []
			for j in shortest_path:
				shortest_path_label.append(labels[j - 1])

			# print("\nThe shortest route: ")
			# print(shortest_path_label)  # It will print the path
			# print("Shortest distance: {:.3f}".format(*distance))  # It will print the distance

			cs_used.extend(shortest_path_label)
			routes.append(shortest_path_label)
			distances.append(distance)
			successful_deliveries.append((demand.lat[i], demand.lon[i]))
			successful_origins.append((demand.origin_lat[i], demand.origin_lon[i]))
		except:
			# print("Unsuccessful delivery", end='\r')
			fail += 1

	success = len(demand) - fail
	print("Unsuccessful deliveries:", fail)
	print("Successful deliveries:", success)
	#print("--- %.2f seconds ---" % (time.time() - start))
	utilization = Counter(cs_used)
	# print(utilization)
	routes_df = pd.DataFrame()
	routes_df['route'] = routes
	routes_df['distance'] = distances
	routes_df['store'] = successful_origins
	routes_df['client'] = successful_deliveries
	return utilization, routes_df


def routing_simulation(cs, demand):
	dist = distances_cs(cs)
	start = time.time()
	fail = 0

	n_core = multiprocess.cpu_count() - 1
	sub_size = len(demand)//n_core
	sub_size_remainder = len(demand)%n_core

	demand_split = []
	for i in range(n_core-1):
		sub_demand = demand.loc[i*sub_size:(i+1)*sub_size-1,:].copy()
		demand_split.append(sub_demand)
	demand_split.append(demand.loc[(n_core-1)*sub_size:,:].copy())
	sub_utilization = []
	sub_routes = []

	pool = multiprocess.Pool(n_core)
	processes = [pool.apply_async(subrouting, args=(cs, sub_demand, dist)) for sub_demand in demand_split]

	for process in processes:
		utilization, routes_df = process.get()
		sub_utilization.append(utilization)
		sub_routes.append(routes_df)


	# for sub_demand in demand_split:
	# 	utilization, routes_df = subrouting(cs, sub_demand, dist)
	# 	sub_utilization.append(utilization)
	# 	sub_routes.append(routes_df)

	keys = ["store", "client"] + [str(i) for i in range(len(cs))]
	values = [0 for i in range(len(cs)+2)]

	utilization = {i:j for i, j, in zip(keys, values)}
	routes_df = pd.DataFrame()

	for i in range(n_core):
		if sub_utilization[i] is not None:
			for key in keys:
				utilization[key] += sub_utilization[i][key]
			routes_df = pd.concat([routes_df, sub_routes[i]], ignore_index=True)

	success = len(demand) - fail
	print("Unsuccessful deliveries:", fail)
	print("Successful deliveries:", success)
	print("--- %.2f seconds ---" % (time.time() - start))

	return utilization, routes_df



def main():
	origin = [40.41101822870703, -79.91054922095948]  # Waterfront

	# Get the demand
	demand = pd.read_csv("data/demand_county_1000.csv")
	demand['origin_lat'] = origin[0]
	demand['origin_lon'] = origin[1]

	print(demand)
	# Get the cs_locations todo: we're randomly generating it now, change to import it from somewhere
	# cs_locations should be a list with [lat, lon] of each charging station

	AB_latlon = pd.read_csv('data/AB_latlon.csv')
	A = ast.literal_eval(AB_latlon.loc[0, 'location'])
	B = ast.literal_eval(AB_latlon.loc[1, 'location'])

	n_cs = 60
	nrow = 50
	ncol = 50

	cs_locations = []
	np.random.seed(1990)
	for i in range(n_cs):
		row = np.random.randint(0, nrow)
		col = np.random.randint(0, ncol)
		cs_locations.append(charging_stations.converttolatlon(row, col, nrow, ncol))
	print("Demand size = ", len(demand))
	print("Charging stations = ", n_cs)
	cs = charging_stations.create_cs(cs_locations)
	utilization, routes_df = routing_simulation(cs, demand)
	print(routes_df)
	print(utilization)

	routes_df.to_csv("data/routes.csv", index=False)
	df_ut = pd.DataFrame(utilization.values(), index=utilization.keys())
	df_ut.to_csv('data/utilization.csv')


if __name__=="__main__":
	main()