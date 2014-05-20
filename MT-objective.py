from ImprovedIBM2.IBM2 import *#Alignment
from ImprovedIBM2.eval_alignment import *
from fastAlignment.fastAlignment import * #Alignment
from fastAlignment.eval_alignment import *
#from Improved import Alignment
from time import time
from bo import BO
import numpy as np
import os
import sys
import subprocess

class IBM2Objective(object):
    def __init__(self):
        self.domain = np.transpose(np.array([[0.01, 0.2],[0.0001,0.002],[0.01,0.2]]))
        self.ndim = 3
        
    def map_params(self, x):
        params = x.ravel()
        return params
    
    def __call__(self, x):
        
        params = self.map_params(x)
        return runIBM2(params)
        
        
        
class FastAlignmentObjective(object):
    def __init__(self):
        self.domain = np.transpose(np.array([[0.01, 0.2],[0.0001,0.002],[0.1,20]]))
        self.ndim = 3
        
    def map_params(self, x):
        params = x.ravel()
        return params
    
    def __call__(self, x):
        
        params = self.map_params(x)
        return runfastAlignment(params)


class CdecObjective(object):
    def __init__(self):
        self.domain = np.transpose(np.array([[0.01, 0.2],[0.0001,0.002]]))
        self.ndim = 2
        
    def map_params(self, x):
        params = x.ravel()
        return params
    
    def __call__(self, x):   
        params = self.map_params(x)
        print str(params[0])+ ' ' +str(params[1])
        
        
        pipe_in,pipe_out,pipe_err= os.popen3('/home/brian/workspace/cdec/word-aligner/fast_align -i /home/brian/workspace/cdec/training.es-en -d -v -o -H -x /home/brian/workspace/cdec/test.es-en'+
                      ' -prob_align_null '+str(params[0]) +' -a '+str(params[1]) , 'wr' )
        '''
        
        p = subprocess.Popen('/home/brian/workspace/cdec/word-aligner/fast_align -i /home/brian/workspace/cdec/training.es-en -d -v -o -H -x /home/brian/workspace/cdec/test.es-en'+
                      ' -prob_align_null '+str(params[0]) +' -a '+str(params[1]) ,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        (pipe_in,pipe_out,pipe_err)=(p.stdin, p.stdout, p.stderr)
        '''
        line = ''
        elements = []
        while 1:
            line = pipe_err.readline()
            if '' == line:
                break
            elements = line.split()
            #print "BO INFO: "+ line
        # TOTAL LOG PROB -18901.8
        
        
        score = 0
        if elements[len(elements)-1]=='-inf':
            score = -9999999
        else:
            score = float(elements[len(elements)-1])
                
        print 'Result Likelihood:'+str(score)
        
        #score -1000
        return score
    
    
if __name__ == '__main__':
    
    # IBM2
    '''
    x0 = np.random.randint(1,20,(20,1))*1.0/1000 
    x1 = np.random.randint(1,20,(20,1))*1.0/10000
    x2 = np.random.randint(1,20,(20,1))*1.0/100
    
    # fastAlignment
    x0 = np.random.randint(1,20,(200,1))*1.0/1000 
    x1 = np.random.randint(1,20,(200,1))*1.0/10000
    x2 = np.random.randint(1,200,(200,1))*1.0/10
    '''
    #cdec
    x0 = np.random.randint(1,20,(200,1))*1.0/1000 
    x1 = np.random.randint(1,20,(200,1))*1.0/10000
    
    #x = np.vstack((x0.T,x1.T,x2.T))
    x = np.vstack((x0.T,x1.T))
    x = x.T
    
    objective = CdecObjective()
    
    bo = BO(objective, noise=1e-1)

    for _ in xrange(50):
        bo.optimize(num_iters=1)

        # Get predictions for plotting
        #y_hat, y_hat_var = bo.predict(x, predict_variance=True)
        #y_hat_upper_bound = y_hat + 1.96 * np.sqrt(y_hat_var)
        #y_hat_lower_bound = y_hat - 1.96 * np.sqrt(y_hat_var)
        #ei = bo.expected_improvement(bo.grid)
        print 'Best Param:' + str(bo.best_param)
        print 'Best Value:' + str(bo.best_value)
    print "Optimization finished."
    print "Best parameter settings found:"
    # objective.map_params(...) will attach names to the parameters so you can tell what they are
    print(objective.map_params(bo.best_param))
    print "With cross validation accuracy: {}".format(bo.best_value)

    
