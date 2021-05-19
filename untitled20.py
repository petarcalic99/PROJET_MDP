#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:01:53 2021

@author: charlotte
"""
import time
import mdptoolbox
import numpy as np
#P, R = mdptoolbox.example.forest()
P = np.array(( ((0,0,1),(0,0,0 ),(0,0,0)) , ((0,1,0),(0,0,0 ),(0,0,0)) , ((0,0,0),(1,0,0 ),(0,0,0)), ((0,0,0.3333),(0,0,0.3333 ),(0,0,0.3333))  ))
print(P)

R = np.array(((2, 1, -1000, -1000), (-1000, -1000,2,-1000),(-1000,-1000,-1000,3)))
#print(P)
#print(R)



vi = mdptoolbox.mdp.ValueIteration(P, R, 1 ,0.01,700)
start_time = time.time()
vi.run()
print("--- %s seconds ---" % (time.time() - start_time))

print(vi.policy)
print(vi.V)