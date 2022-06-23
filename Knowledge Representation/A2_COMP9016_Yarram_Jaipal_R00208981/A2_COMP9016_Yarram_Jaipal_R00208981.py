# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:28:41 2021

@author: reddy
"""
# importing the required functionalities from AIMA
from learning import DataSet, err_ratio
from probabilistic_learning import *
from probability import *

# 1.2 BAYESIAN NETWORKS

def bayesian_network():
    print('\n ********************* BAYESIAN NETWORKS *****************************')
    # creating bayes nodes for the network.
    GlobalWarming_node = BayesNode('GlobalWarming', ['FossilFuels', 'Traffic'], 
                       {(True, True): 0.98,(True, False): 0.80, (False, True): 0.67, (False, False): 0.20})
    
    RenewableEnergy_node = BayesNode('RenewableEnergy', ['GlobalWarming'], {True: 0.90, False: 0.40})
    
    AI_node =BayesNode('AI', ['RenewableEnergy'], {True: 0.5, False: 0.1})
    
    Emp_node =BayesNode('Employed', ['AI'], {True: 0.3, False: 0.01})
    
    
    FossilFules_node = BayesNode('FossilFuels', '', 0.8)
    Traffic_node = BayesNode('Traffic', '', 0.6)
    
    print()
    print("********* Renewable Energy Node *************")
    print(RenewableEnergy_node)
    print(RenewableEnergy_node.p(True, {'GlobalWarming': False, 'Fossil Fuels': False}))
    
    print()
    print("********* Global Worming Node *************")
    print(GlobalWarming_node)
    print(GlobalWarming_node.p(False, {'FossilFuels': True, 'Traffic': True}))
    
    T, F = True, False
    # creating a Bayes Network with the help of predescribed nodes.
    bayesNet = BayesNet([
                ('FossilFuels', '', 0.8),
                ('Traffic', '', 0.6),
                ('GlobalWarming', 'FossilFuels Traffic',
                 {(T, T): 0.98, (T, F): 0.80, (F, T): 0.67, (F, F): 0.27}),
                ('RenewableEnergy', 'GlobalWarming', {T: 0.90, F: 0.40}),
                ('AI', 'RenewableEnergy', {T: 0.5, F: 0.1}),
                ('Employed', 'AI', {T: 0.5, F: 0.01})
                ])
    print()
    print("******** Bayes Network ************")
    print(bayesNet)
    
    
    print()
    print("******** Query ****************")
    print("Query :FossilFuels (Traffic -- False , AI -- True , Renewable Enery -- True)")
    Query = enumeration_ask('FossilFuels',{'Traffic': False, 'AI': True, 'RenewableEnergy': True}, bayesNet)
    print (Query.show_approx())
    
    
bn = bayesian_network()

# 1.3 NAIVE BAYES LEARNER
print("\n\n********************* NAIVE BAYES LEARNER *********************")
lenses = DataSet(name="lenses")

# Here we showcase the to total number of attributes (features + target).
print("Dataset - attrs:", lenses.attrs)

# This is the target Class or label
print("Dataset - target:", lenses.target)

# This will be the column for feature data
Tcolumn=lenses.inputs
print("Dataset - inputs:",Tcolumn)

# Get the Label Class from the dataset
targetV = lenses.values[lenses.target]

#calculated label class distribution.
target_distribution = CountingProbDist(targetV)

'''we initialized a dictionary of CountingProbDist objects, one for each class and feature'''
attributedistribution = {(gv, attribute): CountingProbDist(lenses.values[attribute])
              for gv in targetV
              for attribute in lenses.inputs}


for example in lenses.examples:
        targetval = example[lenses.target]
        target_distribution.add(targetval)
        for attribute in lenses.inputs:
            attributedistribution[targetval, attribute].add(example[attribute])


for target in target_distribution.dictionary.keys():
    print ('“Prior” probabilities for each of the classes {} is {}'.format(target, target_distribution[target]))

FDistribution = { f : CountingProbDist(lenses.values[f])
                 for f in lenses.inputs}

 # probability of evidence
print("\n\n2. Probability of evidence.")
for f in lenses.inputs:
    print("\nProbability of evidence for unique values of feature {} is: ".format(f))
    for val in FDistribution[f].dictionary.keys():
          print(" Value: {} ; Probabilty: {}".format(val,FDistribution[f][val]))
          
    # probability of likelihood of evidences
print("\n\n3. Probability of likelihood of evidences (numerator).")
for featuredata in attributedistribution:
    print("\nFeature {} with class {} :".format(featuredata[1],featuredata[0]))
    for unique in attributedistribution[featuredata].dictionary.keys():
        print("Likelihood of evidence for value {}: {}".format(unique,attributedistribution[featuredata][unique]))

print()
print("******************************************************************")
#enses = DataSet(name="lenses",exclude=[0])
print('test',lenses.inputs)
print('test',lenses.target)
NBL = NaiveBayesLearner(lenses, continuous=False)
print("Discrete Classifier")
a = NBL([1,1, 2, 2, 2])
if a == 1:
    print ("The patient should be fitted with hard contact lenses")
elif a == 2:
    print ("The patient should be fitted with soft contact lenses")
else:
    print("The patient should not be fitted with contact lenses")


b = NBL([2,2, 1, 1,1])
if b == 1:
    print ("The patient should be fitted with hard contact lenses")
elif b == 2:
    print ("The patient should be fitted with soft contact lenses")
else:
    print("The patient should not be fitted with contact lenses")

c = NBL([3,2, 2, 1,3])
if c == 1:
    print ("The patient should be fitted with hard contact lenses")
elif c == 2:
    print ("The patient should be fitted with soft contact lenses")
else:
    print("The patient should not be fitted with contact lenses")
    
print("Error ratio for adaboost: ", err_ratio(NBL, lenses))
print()