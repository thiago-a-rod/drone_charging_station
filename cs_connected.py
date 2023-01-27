import numpy as np
import geopy.distance as geo


def in_range_connected(cs, origin, connected, r):
    for i in range(len(cs)):
        print(r)
        d = geo.distance(cs[i], origin).km
        if d < 2 * r:
            connected.append(i)
    return list(set(connected))


def get_connected(cs, origin, origins, r):
    #local_cs = cs.copy()
    local_cs = [i.location for i in cs]
    local_cs.extend(origins)
    connected = []
    inspected = []
    answer = set()
    loop = True

    origin = local_cs.index(origin)
    while loop:
        if (set(inspected) == set(connected)) and (len(answer) > 0):
            loop = False
        elif (inspected != connected) or (inspected == []):

            connected = in_range_connected(local_cs, local_cs[origin], connected, r)
            answer = answer.union(set(connected))
            inspected.append(origin)
        if len(set(inspected)) < len(set(connected)):
            to_inspect = list(set(connected).difference(set(inspected)))
            origin = to_inspect[0]
        # print(connected, inspected, origin)
    connected = [local_cs[i] for i in connected]
    return connected


def get_valid_cs(cs, origins, r):
    centers_connected = []
    for i in range(len(origins)):
        centers_connected = centers_connected + get_connected(cs, origins[i], origins, r)

    unique = []
    for i in range(len(centers_connected)):
        if centers_connected[i] not in unique:
            unique.append(centers_connected[i])
    return unique


