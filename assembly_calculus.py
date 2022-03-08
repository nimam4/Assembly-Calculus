import numpy as np

class neuron:
    def __init__(self, idx):
        self.ID = idx; 
        self.winit = 0.1; 
        self.preSpikes = np.array([]);          #stores IDs of presynaptic spikes for learning
        self.monitor = []; 
        
    def initConnections(self, presynapticIDs): 
        self.presynapticIDs = presynapticIDs; 
        M = len(presynapticIDs); 
        self.Ws = np.full((M), self.winit, dtype='float64'); 
        self.renormalizeWs(); 

    def renormalizeWs(self): 
        self.Ws = self.Ws/np.sum(self.Ws);

    def learn(self, beta):
        spikeIDs = np.where(self.preSpikes==1)[0]; 
        self.Ws[spikeIDs] = (1+beta)*self.Ws[spikeIDs]; 
        self.renormalizeWs(); 

    def update(self, spikeArray):
        self.preSpikes = spikeArray[self.presynapticIDs]; 
        activity = np.multiply(self.Ws, self.preSpikes);
        activity = np.sum(activity); 
        return activity

class area: 
    def __init__(self, N=1000, p=0.1, k=10, beta=0.1, nInputs=1000, p2=0.1): 
        self.nNeurons = N; 
        self.p = p; 
        self.p2 = p2;           #afferent input connection probability
        self.k = k; 
        self.beta = beta; 
        self.nInputs = nInputs; 
        self.spikes = [[]];                       #stores history of spikes
        self.monitor = []; 
        self.timestamp = 0; 

        self.initNeurons(); 
                                            
    def initNeurons(self):
        self.neurons = []; 
        for i in range(self.nNeurons): 
            self.neurons.append(neuron(idx=i)); 
            recurrentIDs = np.random.choice(range(self.nNeurons), int(self.p*self.nNeurons), replace=False); 
            afferentIDs = self.nNeurons + np.random.choice(range(self.nInputs), int(self.p2*self.nInputs), replace=False);
            presynapticIDs = np.concatenate((recurrentIDs, afferentIDs)); 
            self.neurons[i].initConnections(presynapticIDs); 
    
    def computeWTA(self, netActivity): 
        netActivity = np.asarray(netActivity); 
        sortedActivity = (-netActivity).argsort();
        spikes = sortedActivity[:self.k];
        return spikes; 
    
    def reset(self): 
        self.spikes[-1] = []; 
    
    def update(self, afferentSpikes, driverIDs=[]):
        self.timestamp += 1; 
        netActivity = []; 
        recurrentSpikes = np.zeros(self.nNeurons);  
        recurrentSpikes[self.spikes[-1]] = 1;      #set previous spike IDs to 1
        spikeArray = np.concatenate((recurrentSpikes, afferentSpikes)).astype(int); 
                
        #Update neurons
        for i in range(self.nNeurons): 
            a = self.neurons[i].update(spikeArray); 
            netActivity.append(a);  
                              
        #Forced drivers
        for i in driverIDs:
            netActivity[i] = 100000; 
                            
        #Find spikes of this timestep
        spikeIDs = self.computeWTA(netActivity);
        self.spikes.append(spikeIDs); 
        #Learning
        if(self.beta>0):
            for i in spikeIDs:  
                self.neurons[i].learn(self.beta); 
                pass; 
