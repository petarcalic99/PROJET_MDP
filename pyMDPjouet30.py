#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Marmote and MarmoteMDP and pyMarmoteMDP are free softwares: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#Marmote is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with MarmoteMDP. If not, see <http://www.gnu.org/licenses/>.

#Copyright 2019 Emmanuel Hyon, Alain Jean-Marie
"""
 @brief An example to enumerate a state space of two dimensions
 @author Hyon, Lip6
 @date Nov 2020
 @version 1.1
 
 Cet exemple de test fait créer un MDP tout simple

"""
from pyMarmoteMDP import *
import time

critere = "min"
epsilon = 0.0001
maxIter = 1000
gamma = 0.5

dimSS = 4
dimSA = 3
actionSpace =marmoteInterval(0,dimSA-1)
stateSpace = marmoteInterval(0,dimSS-1)

print("#")

#vector to store the dimension
trans=sparseMatrixVector(dimSA)

P0 = sparseMatrix(dimSS,dimSS)
P0.addToEntry(0,0,0)
P0.addToEntry(0,1,0.875)
P0.addToEntry(0,2,0.0625)
P0.addToEntry(0,3,0.0625)
P0.addToEntry(1,0,0)
P0.addToEntry(1,1,0.75)
P0.addToEntry(1,2,0)
P0.addToEntry(1,3,0.25)
P0.addToEntry(2,0,0)
P0.addToEntry(2,1,0)
P0.addToEntry(2,2,0.5)
P0.addToEntry(2,3,0.5)
P0.addToEntry(3,0,0)
P0.addToEntry(3,1,0)
P0.addToEntry(3,2,0)
P0.addToEntry(3,3,1)
trans[0] = P0

P1= sparseMatrix(dimSS, dimSS);
P1.addToEntry(0,0,0.875)
P1.addToEntry(0,1,0)
P1.addToEntry(0,2,0.125)
P1.addToEntry(0,3,0)
P1.addToEntry(1,0,0)
P1.addToEntry(1,1,0.75)
P1.addToEntry(1,2,0.125)
P1.addToEntry(1,3,0.125)
P1.addToEntry(2,0,0.8)
P1.addToEntry(2,1,0)
P1.addToEntry(2,2,0.2)
P1.addToEntry(2,3,0)
P1.addToEntry(3,0,0)
P1.addToEntry(3,1,0)
P1.addToEntry(3,2,0)
P1.addToEntry(3,3,1)
trans[1] = P1

P2 = sparseMatrix(dimSS, dimSS);
P2.addToEntry(0,0,0)
P2.addToEntry(0,1,0)
P2.addToEntry(0,2,1)
P2.addToEntry(0,3,0)
P2.addToEntry(1,0,1)
P2.addToEntry(1,1,0)
P2.addToEntry(1,2,0)
P2.addToEntry(1,3,0)
P2.addToEntry(2,0,0)
P2.addToEntry(2,1,1)
P2.addToEntry(2,2,0)
P2.addToEntry(2,3,0)
P2.addToEntry(3,0,0)
P2.addToEntry(3,1,0)
P2.addToEntry(3,2,0)
P2.addToEntry(3,3,1)
trans[2] = P2



Reward  = sparseMatrix(dimSS, dimSA);
Reward.addToEntry(0,0,0)
Reward.addToEntry(0,1,4000)
Reward.addToEntry(0,2,6000)


Reward.addToEntry(1,0,1000)
Reward.addToEntry(1,1,4000)
Reward.addToEntry(1,2,6000)

Reward.addToEntry(2,0,3000)
Reward.addToEntry(2,1,4000)
Reward.addToEntry(2,2,6000)

Reward.addToEntry(3,0,3000)
Reward.addToEntry(3,1,4000)
Reward.addToEntry(3,2,6000)



print("Debut de la construction MDP")
mdp1 = totalRewardMDP(critere,stateSpace,actionSpace,Reward)
#mdp1 = discountedMDP(critere, stateSpace, actionSpace, trans, Reward,gamma)

print("Fin de la construction MDP\n")

print("Affichage MDP")
mdp1.writeMDP()

print("Calcul iteration valeur")
#call the function to solve the MDP.       Ajout de time
start_time = time.time()
optimum = mdp1.valueIteration(epsilon, maxIter)
print("--- %s seconds ---" % (time.time() - start_time))



print("********************************")
print("Solution iteration valeur")
optimum.writeSolution()

print("********************************")