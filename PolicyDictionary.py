# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 08:27:17 2022

@author: webbe
"""

import numpy as np
import pandas as pd
import ast
import random

def policydict(dictionary, node, action, reward):
    if not bool(dictionary): #if it is empty
        dictionary = {node: {'action':action, 'reward': reward}}
    else:
        if node in list(dictionary.keys()) and reward > dictionary[node]['reward']: #replace
            dictionary[node]['action'] = action #replace action
            dictionary[node]['reward'] = reward #replace reward
        else: #add a new entry
            dictionary[node] =  {'action':action, 'reward': reward}
    return dictionary

def returnpolicy(dictionary):
    dictionary = dict(sorted(dictionary.items())) #sort the dictionary after it's done
    visited_nodes = list(dictionary.keys()) #get the list of visited nodes
    actions = [] #get the list of actions for each node that we visited
    for i in visited_nodes:
        actions.append(dictionary[i].get('action'))
    
    return visited_nodes, actions
    
def main():
    dictionary = dict()
    nodes = list(range(0, 10))

    for i in range(15):    
        node = np.random.choice(nodes) #randomly select one of the nodes ##observation
        action = np.random.choice([0, 1, 2, 3, 4]) #action ##action
        reward = np.random.rand() #calculate reward ##reward
        dictionary = policydict(dictionary, node, action, reward) #store in dictionary
          
    visited_nodes, actions = returnpolicy(dictionary)
    
    print(dictionary)
    print(visited_nodes)
    print(actions)
    
    
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()