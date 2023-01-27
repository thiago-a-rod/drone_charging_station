import numpy as np
import geopy.distance as geo
import pandas as pd
import ast


def converttolatlon(gridNS, gridEW, nrow, ncol):
    AB_latlon = pd.read_csv('data/AB_latlon.csv')
    A = ast.literal_eval(AB_latlon.loc[0, 'location'])
    B = ast.literal_eval(AB_latlon.loc[1, 'location'])
    lat = A[0] + ((gridNS/nrow)*(B[0]-A[0]))
    lon = A[1] + ((gridEW/ncol)*(B[1]-A[1]))
    return [lat, lon]


def converttogrid(cs_location, nrow, ncol):
    AB_latlon = pd.read_csv('data/AB_latlon.csv')
    A = ast.literal_eval(AB_latlon.loc[0, 'location'])
    B = ast.literal_eval(AB_latlon.loc[1, 'location'])
    gridNS = nrow*(cs_location[0] - A[0])/(B[0]-A[0])
    gridEW = ncol*(cs_location[1] - A[1])/(B[1]-A[1])
    return gridNS, gridEW


class Charging_Station:
    def __init__(self, location):
        self.location = location

    def move(self, direction, S):
        row, col = converttogrid(self.location, S.shape[0], S.shape[1])
        row = int(round(row,0))
        col = int(round(col,0))
        print("<<<<<<<>>>>>>>>", row,col)
        if direction == 0:  # up
            if (row != S.shape[0] - 1):
                self.location = converttolatlon(row + 1, col, S.shape[0], S.shape[1])
        elif direction == 1:  # down
            if (row != 0):
                self.location = converttolatlon(row - 1, col, S.shape[0], S.shape[1])

        elif direction == 2:  # left
            if (col != 0):
                self.location = converttolatlon(row, col -1, S.shape[0], S.shape[1])
        elif direction == 3:  # right
            if (col != S.shape[1] - 1):
                self.location = converttolatlon(row, col + 1, S.shape[0], S.shape[1])
        elif direction == 4:  # stayput
            pass

    def encode(self, S):
        row, col = converttogrid(self.location, S.shape[0], S.shape[1])
        i = row
        i *= S.shape[0]
        i += col
        return i

def create_cs(cs_location):
    '''
    cs_location: [list] with latitude and longitude of the charging stations
    return: [list] with charging stations
    '''
    cs = []
    ncs = len(cs_location)
    for i in range(ncs):
        cs.append(Charging_Station(cs_location[i]))
    return cs