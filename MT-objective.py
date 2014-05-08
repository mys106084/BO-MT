from ImprovedIBM2.IBM2 import *#Alignment
from ImprovedIBM2.eval_alignment import *
#from Improved import Alignment
from time import time
from bo import BO
import numpy as np

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
        
        '''
        params = self.map_params(x)
        x0 = params[0]
        x1 = params[1]
        x2 = params[2]
        
        myAlignment = Alignment(x0,x1,x2)
        myAlignment.Inputcorpus()
        myAlignment.EM_IBM2()
        myAlignment.Dev_IBM2()
        score =  Objective_fscore(open('data//dev.key'),open('data//dev.out'))
        print 'Current F1 score:' + str(score)
        return score
        '''
        
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


if __name__ == '__main__':
    
    # IBM2
    '''
    x0 = np.random.randint(1,20,(20,1))*1.0/1000 
    x1 = np.random.randint(1,20,(20,1))*1.0/10000
    x2 = np.random.randint(1,20,(20,1))*1.0/100
    '''
    # fastAlignment
    x0 = np.random.randint(1,20,(20,1))*1.0/1000 
    x1 = np.random.randint(1,20,(20,1))*1.0/10000
    x2 = np.random.randint(1,200,(200,1))*1.0/10
    
    x = np.vstack((x0.T,x1.T,x2.T))
    x = x.T
    
    objective = IBM2Objective()
    
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

    
