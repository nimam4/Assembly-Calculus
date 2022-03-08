#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#Generates Figure 2g (assembly convergence) of Papadimitriou et al. PNAS 2020 
#Writes ACT config file at the end 
#To generate other figures of the same paper (pattern completion, pattern associations), modify this top-level test file

import cortex; 
import numpy as np; 
import matplotlib.pyplot as plt
import lib

np.random.seed(5); 

simTime = 100; 

#nInputs = 10000; 
#nNeurons = 10000; 
#connP = 0.01

nInputs = 1000; 
nNeurons = 1000; 
connP = 0.01

kWinners = 100; 
#betas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]; 
betas = [0.01]; 
plot_x = range(0, 100, 1); 
plot_y = []; 
              
#Input
input1 = np.zeros(nInputs); 
activeInputIDs = np.random.choice(range(nInputs), kWinners, replace=False); 
input1[activeInputIDs] = 1; 

for beta in betas: 
    a1 = cortex.area(N=nNeurons, p=connP, k=kWinners, beta=beta, nInputs=nInputs); 
    for i in range(0, simTime): 
        a1.update(input1); 
        if((i+1)%10 ==0): 
            print(i+1);
    s = lib.findTotalSupportOverTime(a1.spikes, plot_x); 
    plot_y.append(s); 

#%%
plt.figure(figsize=(12, 8)); 
for i in range(0, len(plot_y)): 
    #plt.plot(plot_x, plot_y[i], label=betas[i]); 
    plt.plot(plot_x, plot_y[i]); 

plt.xlabel('times projected');
plt.ylabel('total support'); 
#plt.legend(); 
          
#%% ACT config

wMatrix = np.zeros((nNeurons+nInputs, nNeurons+nInputs), dtype=int); 
for i in range(0, nNeurons): 
    wMatrix[i][a1.neurons[i].presynapticIDs] = 1; 

#Write file
file1 = open("assemblyConfig.act","w"); 
file1.write("template<pint I, O> defproc neuron (bool in[I]; bool out[O]);\n\n"); 
file1.write("defproc design () \n"); 
file1.write("{\n"); 
#Neuron initialization
for i in range(0, nNeurons): 
    nIn = sum(wMatrix[i]); 
    nOut = sum(wMatrix[:, i]); 
    file1.write("    neuron<" + str(nIn) + "," + str(nOut) + "> n" + str(i) + ";\n"); 
#Neuron connections
wireCnt = np.zeros(nNeurons+nInputs, dtype=int);           #fanout count
for postID in range(0, nNeurons):
    preIDs = np.where(wMatrix[postID]==1)[0];
    for j in range(0, len(preIDs)): 
        file1.write("    n" + str(postID) + ".in[" + str(j) + "] = n" + str(preIDs[j]) + ".out[" + str(wireCnt[preIDs[j]]) + "];\n");  
        wireCnt[preIDs[j]] += 1; 

file1.write("\n}"); 
file1.close(); 