import glob
import numpy as np
import csv
from scipy.special import logsumexp
import math
from load import *
import os
import time
import sys
import argparse




class Model():
        
    def __init__(self, *args, npzFile = None, num_HiddenState = 4, num_observationClass = 5, **kwargs):
        # Define the initial (A,B, and pi)
        # Default values
        self.num_HiddenState = num_HiddenState
        self.num_observations = num_observationClass

        # print(type(args))
        print(args)
        # if args
        if npzFile is not None:
            
            print("test")
            self.E = npzFile['Emission']
            self.T = npzFile['Transmission']
            self.P = npzFile['Prior']


        else:
            print("train")
            self.P = self.prior(num_HiddenState)
            self.E = self.Emission(num_HiddenState, num_observationClass)
            self.T = self.Transition(num_HiddenState)

        print(self.T)
        self.hidden = num_HiddenState
        self.observed = num_observationClass
        print("Prior",self.P.shape)
        print("Emission",self.E.shape)
        print("Transition",self.T.shape)

    
    # def __init__(self, num_HiddenState = 4, num_observationClass = 5):
 

    # def __init__(self, Emissions, Transitions, Priors, num_HiddenState=4, num_observationClass=5):
    #     # Define the initial (A,B, and pi)
    #     self.num_HiddenState = num_HiddenState
    #     self.num_observations = num_observationClass
    #     self.P = self.prior(num_HiddenState)
    #     self.E = self.Emission(num_HiddenState, num_observationClass)
    #     self.T = self.Transition(num_HiddenState)
    #     self.hidden = num_HiddenState
    #     self.observed = num_observationClass
    #     print("Prior",self.P.shape)
    #     print("Emission",self.E.shape)
    #     print("Transition",self.T.shape)


    def prior(self, num_HiddenState):
        p = np.divide(np.ones(num_HiddenState),num_HiddenState)

#             p[0] = 1
        return p

    def Emission(self, num_HiddenState, num_observationClass):
        e = np.zeros((num_HiddenState,num_observationClass))
        e[:,:] = 1/num_observationClass


        return e

    def Transition(self, num_HiddenState):
        '''
        First, we will try creating an ergodic transition matrix. If I have time, will try a left-right model
        '''
        # t = np.full((num_HiddenState,num_HiddenState),1/num_HiddenState)
        t = np.zeros((num_HiddenState, num_HiddenState))
        
        


        # This will create a left-right matrix. Will try this later, but for now ergodic
        
        for i in range(num_HiddenState-2):
            t[i,i] = 1/3
            t[i,i+1] = 1/3
            t[i,i+2] = 1/3

        # Final initializations
        t[num_HiddenState-2,num_HiddenState-2] = 1/2
        t[num_HiddenState-2,num_HiddenState-1] = 1/2

        t[num_HiddenState-1,num_HiddenState-1] = 1


        return t
    
    def forwardInduction(self, sequence, reverse = False):
        print("too much T",self.T.shape)
        print("Emissions", self.E.shape)
        alpha = np.zeros((len(self.P),len(sequence)))
        alpha[:,0] = self.P * self.E[:,0]
#             print(alpha)

        # Now the real fun begins
        for i,state in enumerate(sequence,1):
            alpha[:,i] = logsumexp(np.multiply(alpha[:,i-1] , self.E[:,i]))
        print(alpha)        


    def initialization(self, sequence):
        '''
        The forward calculation of the algorithm
        '''
        print(sequence.shape)
        alpha = np.zeros((self.num_HiddenState,len(sequence)))
        alpha[:,0] = np.multiply(self.P, self.E[:,0])
        print(alpha.shape)
        
        
        for t,time in enumerate(sequence[1:],1):
            
            for j,state_out in enumerate(range(self.num_HiddenState)): 
                alpha[j,t] = logsumexp([np.multiply(alpha[i,t-1],self.T[j,i]) for i in range(self.num_HiddenState)]) \
                + math.log(0.2) # I need to make this actually relate to the observations.            
        print("\nalpha: ",alpha)
        return alpha

    def initializeBeta(self,sequence):
        '''
        The backwards calculation of the forward-backward algorithm
        '''
        beta = np.zeros((self.num_HiddenState,len(sequence)))
        beta[:,-1] = np.ones((self.num_HiddenState,))
        
        for t,_ in enumerate(sequence[:-1],1):
            t = len(sequence) - 1 - t
            
            for j,_ in enumerate(range(self.num_HiddenState)): 
                beta[j,t] = logsumexp([np.multiply(beta[i,t+1],self.T[j,i]) for i in range(self.num_HiddenState)]) \
                + math.log(0.2) # I need to make this actually relate to the observations.
                
        print("\nbeta: ",beta)
        return beta

    def optimalseq(self, sequence, alpha, beta):
        '''
        This will return the optimal state sequence given these observations.
        
        sequence: refers to observation sequence
        
        returns:
        optSeq: This is the optimal sequence
        
        '''
        gamma = np.zeros((self.num_HiddenState,len(sequence)))
        
        for t, _ in enumerate(sequence):
            normalization = np.sum(np.multiply(alpha[:,t],beta[:,t]))
            tmp = np.multiply(alpha[:,t],beta[:,t])
            gamma[:,t] = np.divide(tmp,normalization)
        print("\nGamma: ", gamma[:,:5])
        optSeq = np.argmax(gamma,axis=0)
        print("\noptSeq: ", optSeq)
        
        return optSeq, gamma

    def EM(self, sequence, alpha, beta, optimal_sequence,gamma):
        '''
        This will return 
        
        The output of xi should be the size of the sequence
        '''
    #     xi = np.zeros((len(optimal_sequence)-1,))
        xi = np.zeros((self.num_HiddenState, self.num_HiddenState,len(optimal_sequence)-1))
    #     print(xi.shape)
        
        for t in range(len(optimal_sequence)-1):
            # Hidden states that are inferred for current and one ahead in time
            ot = optimal_sequence[t]
            ot1 = optimal_sequence[t+1]
            
            # Observed sequences
            st = sequence[t]
            st1 = sequence[t+1]


            # Very long list comprehension, that forms that array
            tmp = np.array([alpha[i,t] * self.T[i,j] * self.E[optimal_sequence[t+1],j] * beta[j, optimal_sequence[t+1]] \
            for j in range(self.num_HiddenState) for i in range(self.num_HiddenState)])
            normalization = np.sum(tmp)

            xi[:,:,t] = np.reshape(tmp/normalization,(4,4))
            
            
    #         This gamma isn't used. Hope that is ok!
        newgamma = np.sum(xi,axis= 0)
        
        
        # Update initial probability
        print("\nupdated P: ",gamma[:,0])
        self.P = gamma[:,0]
    
        # Update transition probabilities
        
        self.T = np.sum(xi,axis = 2) / np.sum(gamma[:,:-1], axis = 1)
    #         print("Updated T: ",np.sum(xi,axis = 2)[i,:] / np.sum(gamma[:,:-1], axis = 0)[i])
        print("\ntmp: ",self.T)

        # Update emission probabilities
        print("\nUpdated E: ", np.sum(gamma, axis = 0) / np.sum(gamma, axis = 0))
    #     self.E = np.sum(gamma, axis = 0) / np.sum(gamma, axis = 0)

        return xi,gamma
        



    def baumWelch(self,sequence, testtime = False):
        '''
        Describes doing the forward-backward calculation, followed by determining the optimal sequence that is 
        described by this
        
        Model: the HMM model that is used, which includes in hidden_states, # different observations
        sequence: The sequence that acts as your input
        '''
        # o = set(sequence)
        print("sequence: ",sequence.shape)
        alpha = self.initialization(sequence)
        beta = self.initializeBeta(sequence)
        optSeq, gamma = self.optimalseq(sequence, alpha, beta)
        if testtime:
            return optSeq
        else:
            xi,_ = self.EM(sequence, alpha,beta,optSeq, gamma)
            pass
    def train(self):
        '''
        Training phase. Output is a numpy file with multiple arrays, containing the parameters the parameters


        '''
        truestart = time.time()
        num_clusters = 50 # number of discrete values

        beat_three = fileNames[0:5]; beat3_data = np.array([fileReader(beat3) for beat3 in beat_three])
        beat_four = fileNames[5:10]; beat4_data = np.array([fileReader(beat4) for beat4 in beat_four])
        circle = fileNames[10:15]; circle_data = np.array([fileReader(circ) for circ in circle])
        eight = fileNames[15:20]; eight_data = np.array([fileReader(e) for e in eight])
        inf = fileNames[20:25]; inf_data = np.array([fileReader(i) for i in inf])
        wave = fileNames[25:30]; wave_data = np.array([fileReader(wv) for wv in wave])
        # model.baumWelch(np.array([5,6,7,8,9,8]))
        data = [beat3_data,beat4_data,circle_data,eight_data,inf_data,wave_data]




        for d in data:
            start = time.time()
            for i,motion in enumerate(d):

                print("wave_data: {}".format(i),motion)
                self.baumWelch(motion)
                
            end = time.time()
            print("Motion time: ",end-start)
            time.sleep(1)
        np.savez("HMM_params.npz", Emission = self.E, Transmission = self.T, Prior = self.P)
        pass

            
            
                





if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Writing a training data set, and updating the model parameters 'HMM_params.npz' ", action="store_true")
    parser.add_argument("--test", help = "Testing the data",action="store_true")
    args = parser.parse_args()
    # thresh =0 /
    if args.train:
        model = Model()
        model.train()
    else:
        truestart = time.time()
        npzfile = np.load("HMM_params.npz")
        # print(len(npzfile))
        # Initialize the class
        
        model = Model(npzFile= npzfile)

        files = glob.glob('./2019Proj2_test/*.txt')
        print("Test time....")
        time.sleep(5)
        for file in files:
            # print(file)
            sequence = fileReader(file)
            model.baumWelch(sequence, testtime = True)
        trueend = time.time()
        print("Overall Test time: ", trueend-truestart)



 



    