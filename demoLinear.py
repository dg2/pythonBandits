'''
Demo: Basic Probit Bandits via Randomized Assignment

The main idea of randomized assignment is to, at each step, select
an arm by sampling from a discrete distribution which assigns to each 
arm the estimated probability of that arm being the optimal one (given 
the observations so far)

Here we implement that idea for a generalized linear bandit (probit model) 
with side information (i.e. contextual bandit). Inspired by 'A Modern Bayesian Look at the multi-armed bandit' (Scott 2010)

Dario Garcia
'''
# TODO:
# 1 - Only resample from the active arm
# 2 - Fast version: Get one sample from each arm and decide according to that single sample
# 3 - Learning can be made sequential: We only need to update the posterior with
# the new observation, not use all the history (we only need to use the posterior
# sample mean and covariance as priors for the next iteration)
import scipy.stats as stats
import numpy as np
from numpy.random import rand  as rand
import bayesProbit
import pylab as pl

# Experiment setup
# Success probability model for each arm

def experiment(NUM_ARMS = 3, D = 2, NUM_OBS = 100, NUM_SAMPLES = 500, BURNOUT = 200, prior_sigma = 10):
    # Generate the set of coefficients for each arm
    coef = np.random.randn(D,NUM_ARMS)
    print coef
    # Initialize vectors
    prior = np.ones(NUM_ARMS)/float(NUM_ARMS)
    post = prior
    regret = np.zeros(NUM_OBS)
    random_regret = np.zeros(NUM_OBS)
    cumulative_reward = 0
    theta_mu = np.zeros(D)
    theta_sigma = prior_sigma*np.eye(D)
    arm_data = dict()
    for arm in range(NUM_ARMS):
        arm_data[arm] = {'feat':np.empty((D,0)),'reward':np.empty((0))}
    play_counts = np.zeros(NUM_ARMS)

    # Main loop
    for it in range(NUM_OBS):
        print "="*20
        print "Iteration %d" % it
        
        # Generate features
        feat = rand(D,1)
        theta = stats.norm.cdf(np.dot(feat.T, coef))[0] # Probit model
        print theta
    
        # Select an arm
        # We estimate the posterior probability of each arm being the optimal one
        # for the observed feature by using samples from the posterior of the 
        # probit regression parameters
        # We sample from the posterior of P(\theta_i|y) for each arm
        s = np.zeros((NUM_ARMS,NUM_SAMPLES-BURNOUT))
        for arm in range(NUM_ARMS):
            print 'Arm %i' % arm
            # If we do not have any observations, do some sensible initialization
            if (arm_data[arm]['feat'].shape[1]==0):
                print 'Sampling from the prior'
#                samples = np.random.multivariate_normal(theta_mu, theta_sigma, size = NUM_SAMPLES).T
                samples = np.zeros((D,NUM_SAMPLES))
            else:
                samples = bayesProbit.gibbsSampling(arm_data[arm]['feat'], arm_data[arm]['reward'], theta_mu, theta_sigma, NUM_SAMPLES = NUM_SAMPLES)
            # Show information about the posterior based on the samples,
            # after discarding the burnout period
#            theta_mu = np.mean(samples[:,BURNOUT:], axis = 1)
#            theta_sigma = np.cov(samples[:,BURNOUT:])
            print 'Real mu: ' 
            print coef[:,arm]
            print 'Estimate: '
            print np.mean(samples[:,BURNOUT:], axis = 1)
            print np.cov(samples[:,BURNOUT:])

            # Project the feature point
            s[arm,:] = np.dot(feat.T,samples[:,BURNOUT])
        
        # Estimate the probabilities of being the optimal arm
        aux = np.argmax(s,axis = 0)
        aux = np.array([np.sum(aux==i) for i in range(NUM_ARMS)])
        post = aux/float(np.sum(aux))
        # Sample according to those probabilities
        cum_post = np.cumsum(post)
        aux = rand(1)
        chosen_arm = np.searchsorted(cum_post,aux)[0]
        print 'Choosing arm %d' % chosen_arm    
        play_counts[chosen_arm] += 1
        # Obtain the reward
        reward = stats.bernoulli.rvs(theta[chosen_arm])    
        cumulative_reward += reward
        arm_data[chosen_arm]['feat'] = np.hstack((arm_data[chosen_arm]['feat'],feat))
        arm_data[chosen_arm]['reward'] = np.hstack((arm_data[chosen_arm]['reward'],[reward]))

        # Report
        regret[it] = max(theta)-theta[chosen_arm]
        random_regret[it] = max(theta)-np.mean(theta)
        print "Cumulative reward: %d" % cumulative_reward
        print "Cumulative expected regret: %.3f" % np.sum(regret)
        print "Cumulative expected random regret: %.3f" % np.sum(random_regret)
        print "Play counts: "
        print play_counts
        pl.plot(cumsum(regret))
        pl.plot(cumsum(random_regret))
        pl.legend(['Regret','Random regret'])
        
