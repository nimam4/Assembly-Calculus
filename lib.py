#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np; 

def findAssemblyIDs(spikes): 
    neuronIDs = np.array([]); 
    for i in range(0, len(spikes)): 
        neuronIDs = np.concatenate((neuronIDs, spikes[i])); 
        neuronIDs = np.unique(neuronIDs);
    supportSize = len(neuronIDs); 
    neuronIDs = neuronIDs.astype('int')
    return neuronIDs, supportSize;                    

def findTotalSupportOverTime(spikes, times): 
    #times = range(0, len(spikes), 10);
    supportSizes = []; 
    for t in times: 
        _, s = findAssemblyIDs(spikes[0:t+1]); 
        supportSizes.append(s); 
    return supportSizes

def jaccardSimilarity(l1, l2):
    set1 = set(l1);
    set2 = set(l2);
    intersectionSize = len(set.intersection(set1, set2));
    unionSize = len(set.union(set1, set2));
    return intersectionSize/float(unionSize);

def findSimilarity(sample1, sample2): 
    s = jaccardSimilarity(sample1, sample2); 
    return s; 

def findSimilarityArray(sample1, sample2): 
    a = np.where(sample1==1)[0]; 
    b = np.where(sample2==1)[0]; 
    s = jaccardSimilarity(a, b); 
    return s; 

def findSImatrix(testSet, trainingSet): 
    SImatrix = np.zeros((len(testSet), len(trainingSet))); 
    for i in range(0, len(testSet)): 
        for j in range(0, len(trainingSet)): 
            s = findSimilarity(testSet[i], trainingSet[j]); 
            SImatrix[i][j] = s; 
    return SImatrix

def findSImatrixED(testSet, trainingSet): 
    SImatrix = np.zeros((len(testSet), len(trainingSet))); 
    for i in range(0, len(testSet)): 
        for j in range(0, len(trainingSet)):
            d = np.linalg.norm(testSet[i]-trainingSet[j]); 
            SImatrix[i][j] = d; 
    return SImatrix

def classify(testSet, testSetLabels, trainingSet): 
    SImatrix = findSImatrix(testSet, trainingSet); 
    pValues = []; 
    nCorrect = 0; 
    for i in range(0, len(SImatrix)):
        bestMatchID = np.argmax(SImatrix[i, :]); 
        #bestMatchID = np.argmin(SImatrix[i, :]); 
        pValues.append(bestMatchID); 
        if(bestMatchID==testSetLabels[i]):
            nCorrect+=1; 
    percentCorrect = float(nCorrect)/len(testSet); 
    return percentCorrect, pValues, SImatrix


def classifyRaw(testSet, testSetLabels, trainingSet): 
    SImatrix = findSImatrixED(testSet, trainingSet); 
    pValues = []; 
    nCorrect = 0; 
    for i in range(0, len(SImatrix)):
        #bestMatchID = np.argmax(SImatrix[i, :]); 
        bestMatchID = np.argmin(SImatrix[i, :]); 
        pValues.append(bestMatchID); 
        if(bestMatchID==testSetLabels[i]):
            nCorrect+=1; 
    percentCorrect = float(nCorrect)/len(testSet); 
    return percentCorrect, pValues, SImatrix


def generateRandomBinaryVector(n, k):
    randomV = np.zeros(n); 
    activeInputIDs = np.random.choice(range(n), k, replace=False); 
    randomV[activeInputIDs] = 1;
    return randomV

def addImpulseNoise(trainingData, kWinners, nTest, p=0.5): 
    testData = []; 
    testDataLabels = []; 
    nNoise = int(p*kWinners); 
    for i in range(0, len(trainingData)):
        for j in range(0, nTest): 
            testData.append(np.copy(trainingData[i])); 
            testDataLabels.append(i); 
            zeroIDs = np.where(trainingData[i]==0)[0]; 
            nonzeroIDs = np.where(trainingData[i]==1)[0]; 
            affectedIDs0 = np.random.choice(zeroIDs, nNoise, replace=False); 
            affectedIDs1 = np.random.choice(nonzeroIDs, nNoise, replace=False);                    
            testData[-1][affectedIDs0] = 1; 
            testData[-1][affectedIDs1] = 0; 
    testData = np.asarray(testData); 
    return testData, testDataLabels

def generateRandomData(nDim, kWinners, nTrainingSamples, nTestSamples, impulseNoise=0.1): 
    trainingData = [];
    for i in range(0, nTrainingSamples): 
        input1 = generateRandomBinaryVector(nDim, kWinners); 
        trainingData.append(input1);
    trainingData = np.asarray(trainingData); 
    labelData = trainingData; 
    testData, testDataLabels = addImpulseNoise(trainingData, kWinners, nTestSamples, p=impulseNoise);
    return trainingData, labelData, testData, testDataLabels; 

