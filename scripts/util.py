
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_signals(signals,labels):

    '''

    Auxiliary function to plot all signals.

    input:
        - signals: signals to plot
        - labels: labels of input signals

    output:
        - display plot

    '''

    alphas=[1,0.45,0.45,0.45,0.45]      ### just some opacity values to facilitate visualization

    lenght=np.shape(signals)[1]         ### time lenght of original and filtered signals

    fig = plt.figure()

    for j,sig in enumerate(signals):    ### iterates on all signals

        plt.plot(range(lenght),sig,'-o', label=labels[j],markersize=2,alpha=alphas[j])

    plt.grid()

    plt.ylabel('RSSI')
    plt.xlabel('time')
    plt.legend()
    plt.show()

    return

def gray_filter(signal, N=15):

    '''

    Implementation of Gray filter.
    Takes a signal and filter parameters and return the filtered signal.

    input:
        - signal: signal to be filtered
        - N: window size of input signal

    output:
        - filtered signal

    '''

    predicted_signal=[]

    for j in range(0,np.shape(signal)[0],N):                                    ### iterates on the entire signal, taking steps by N (window size)

        N=np.minimum(N,np.shape(signal)[0]-j)                                   ### just in case we are at signal final and N samples are not available

        R_0=np.zeros((N))
        R_0[:]=signal[j:j+N]                                                    ### saves in R_0 signal values of corresponding window size

        R_1=[]

        for i in range(N):

            R_1.append((np.cumsum(R_0[0:i+1]))[i])                              ### calculates R_1


        ######## calculates gray filter solution ##############################

        ### for further details about filter resolution check kayacan2010

        B=(np.matrix([np.ones((N-1)),np.ones((N-1))])).T

        for k in range(N-1):

            B[k,0]=-0.5*(R_1[k+1]+R_1[k])

        X_n=np.matrix(np.asarray(R_0[1:])).T

        _ = np.matmul(np.linalg.inv(np.matmul(B.T,B)),(np.matmul(B.T,X_n)))

        a=_[0,0]
        u = _[1,0]

        #######################################################################

        X_=R_0[0]
        predicted_signal.append(X_)                                                  ###
        for i in range(1,N):                                                         ### update predicted signal with this window calculation
            predicted_signal.append((((R_0[0]-u/a)*np.exp(-a*(i-1)))*(1-np.exp(a)))) ###

    return predicted_signal

def fft_filter(signal, N=8, M=2):

    '''

    Implementation of Fourier filter.
    Takes a signal and filter parameters and return the filtered signal.

    input:
        - signal: signal to be filtered
        - N: window size of input signal
        - M: samples of fft signal to preserve (remember fft symmetry)

    output:
        - filtered signal

    '''

    predicted_signal = []

    for j in range(0, np.shape(signal)[0], N):      ### iterates on the entire signal, taking steps by N (window size)

        N = np.minimum(N, np.shape(signal)[0] - j)  ### just in case we are at signal final and N samples are not available

        R_0=np.zeros((N))
        R_0[:]=signal[j:j+N]                        ### saves in R_0 signal values of corresponding window size

        R_0_fft=np.fft.fft(R_0)                     ### fft of signal window

        for k in range(int(N/2)):                   ### it keeps M samples of fft and sets the rest to zero
            R_0_fft[M+k]=0                          ### remember fft symmetry
            R_0_fft[-1-M-k]=0

        R_0_ifft=np.fft.ifft(R_0_fft)               ### inverse fft

        for i in range(0, N):
            predicted_signal.append(R_0_ifft[i])    ### update predicted signal with this window calculation

    return predicted_signal

def kalman_block(x,P,s, A,H,Q, R):

    '''

    Prediction and update in Kalman filter

    input:
        - signal: signal to be filtered
        - x: previous mean state
        - P: previous variance state
        - s: current observation
        - A, H, Q, R: kalman filter parameters

    output:
        - x: mean state prediction
        - P: variance state prediction

    '''

    #### check laaraiedh2209 for further understand these equations ##############

    x_mean = A * x + np.random.normal(0, Q, 1)
    P_mean = A * P * A + Q

    K = P_mean * H * (1 / (H * P_mean * H + R))
    x = x_mean + K * (s - H * x_mean)
    P = (1 - K * H) * P_mean

    ##############################################################################

    return x,P

def kalman_filter(signal, A,H,Q, R):

    '''

    Implementation of Kalman filter.
    Takes a signal and filter parameters and returns the filtered signal.

    input:
        - signal: signal to be filtered
        - A, H, Q, R: kalman filter parameters

    output:
        - filtered signal

    '''

    predicted_signal = []

    x=signal[0]                                 ### takes first value as first filter prediction
    P=0                                         ### set first covariance state value to zero

    predicted_signal.append(x)
    for j,s in enumerate(signal[1:]):           ### iterates on the entire signal, except the first element

        x,P=kalman_block(x,P,s, A,H,Q, R)       ### calculates next state prediction

        predicted_signal.append(x)              ### update predicted signal with this step calculation

    return predicted_signal

def choose_particle(particles):

    '''

    Takes an array of particles and returns an element according to weights distribution

    input:
        - particles: array of particles

    output:
        - chosen particle

    '''

    prob_distribution = []

    ######### calculates sum of weights to normalize wheight vector in next step #####

    sum_weights = 0
    for p in particles: sum_weights+=p['weight']

    ##################################################################################

    for p in particles:
        prob_distribution.append(float(p['weight']/sum_weights))

    ######### choose particle according to weights distribution ######################

    a=np.random.choice(particles,1,replace=False,p=prob_distribution)

    ##################################################################################



    return (a[0]['value'][0])


def particle_filter(signal,quant_particles,A=1,H=1,Q=1.6,R=6):

    '''

    Implementation of Particles filter.
    Takes a signal and filter parameters and return the filtered signal.

    input:
        - signal: signal to be filtered
        - quant_particles: filter parameter - quantity of particles
        - A, H, Q, R: kalman filter parameters

    output:
        - filtered signal

    '''


    predicted_signal=[]

    rang=10                                                             ### variation range of particles for initial step

    x=signal[0]                                                         ### takes first value as first filter prediction
    P=0                                                                 ### set first covariance state value to zero

    predicted_signal.append(x)

    min_weight_to_consider = 0.07                                       ### defines some needed constants in algorithm
    min_weight_to_split_particle = 5                                    ###

    for j,s in enumerate(signal[1:]):                                   ### iterates on the entire signal, except the first element

        range_ = [predicted_signal[j-1] - rang,                         ###
                                predicted_signal[j-1] + rang]           ### set variation range for first step sampling

        particles = []

        for particle in range(quant_particles):                         ### loop on all particles

            input=np.random.uniform(range_[0],range_[1])                ### sample particle value from variation range
            weight=1/np.abs(input-x)                                    ### particle weight

            if weight>min_weight_to_consider:                           ### it only iterates on particles which weights
                                                                        ### are greater than _min_weight_to_consider_

                x_, P = kalman_block(input, P, s, A, H, Q, R)           ### calculates next state prediction

                weight = 1 / np.abs(s - x_)                             ### prediction weight
                particles.append({'value':x_,'weight': weight})

                ### for particles with greater weights, it creates other particles in the 'neighborhood' ###########################

                if weight > min_weight_to_split_particle:

                    input=input +np.random.uniform(0,5)
                    x_, P = kalman_block(input, P, s, A, H, Q, R)

                    weight = 1 / np.abs(s - x_)
                    particles.append({'value': x_, 'weight': weight})

                ###################################################################################################################

        x=choose_particle(particles)                                     ### choose a particle, according to weight distribution

        predicted_signal.append(x)                                       ### update predicted signal with this step calculation

    return predicted_signal