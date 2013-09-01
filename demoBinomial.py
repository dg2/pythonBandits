'''
Demo: Basic Binomial Bandits via Randomized Assignment

The main idea of randomized assignment is to, at each step, select
an arm by sampling from a discrete distribution which assigns to each 
arm the estimated probability of that arm being the optimal one (given 
the observations so far)

Dario Garcia
'''
import scipy.stats as stats
import numpy as np
from numpy.random import rand  as rand

# Experiment setup
NUM_OBS = 100
NUM_ARMS = 4
NUM_SAMPLES = 1000
# Success probability of each arm 
#theta = [0.65,0.55,0.4,0.2]
theta = stats.uniform.rvs(0,1,size=NUM_ARMS)
print theta

prior = np.ones(NUM_ARMS)/float(NUM_ARMS)
post = prior
positives = np.zeros(NUM_ARMS)
tries = np.zeros(NUM_ARMS)
reward_vector = np.zeros(NUM_OBS)
regret = np.zeros(NUM_OBS)

for it in range(NUM_OBS):
    print "="*20
    print "Iteration %d" % it
    # Select an arm
    cum_post = np.cumsum(post)
    aux = rand(1)
    chosen_arm = np.searchsorted(cum_post,aux)
    print 'Choosing arm %d' % chosen_arm

    # Obtain the reward
    reward = stats.bernoulli.rvs(theta[chosen_arm])
    reward_vector[it] = reward
    positives[chosen_arm] += reward
    tries[chosen_arm] += 1

    # Update posterior
    # Our magnitude of interest is P(theta_i = max(theta))
    # We can find it by simulation
    # First, we sample from each arm according to the posterior probability
    s = np.zeros((NUM_SAMPLES,NUM_ARMS))
    for arm in range(NUM_ARMS):
        s[:,arm] = stats.beta.rvs(positives[arm]+1,tries[arm]-positives[arm]+1, size=NUM_SAMPLES)
    # Now we count how many times each arm is the 'winner' and use that
    # as a Monte Carlo approximation to the actual probability
    aux = np.argmax(s,axis=1)
    aux = np.array([np.sum(aux==i) for i in range(NUM_ARMS)])
    post = aux/float(np.sum(aux))
    print post
        
    # Report
    regret[it] = np.sum(tries*(np.max(theta)-theta))
    print "Cumulative reward: %d" % np.sum(reward_vector[:it+1])
    print "Cumulative expected regret: %.3f" % regret[it]
