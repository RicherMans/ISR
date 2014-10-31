'''
Created on Sep 15, 2014

@author: richman
'''
import numpy as np
import math
import abc
from multiprocessing import process,Queue

class GMMEstimator(object):
    '''
    classdocs
    '''
    gaussians = 0
    iterations = 100
    threshold = 0.001
    
    def __init__(self,label, gaussians, threshold=0.001):
        '''
        Constructor
        '''
        self.gaussians = gaussians
        self.label = label
        
    def _multipdf(self, x, mue, sigma):
        if self.dim == len(mue) and (self.dim, self.dim) == sigma.shape:
            det = np.linalg.det(sigma)
#             Check if sign is nonnegative
            if det <= 0:
                raise ValueError("Sigma: %s Negative determinate : %.2f" % (sigma, det))
            n_const = 1. / (math.pow(2.*np.pi, float(self.dim) / 2.) * math.pow(det, 0.5))
            x_mu = np.matrix(x - mue)
            sigma_inv = np.matrix(np.linalg.inv(sigma))
            result = n_const * math.pow(np.e, -0.5 * (x_mu * sigma_inv * x_mu.T))
            return result
        else:
            raise ValueError("dimensions do not match")
    
    def _posteriorocc(self, datapoint, weights, mues, sigmas, gaussind):
#         Use tis in e step so that dont need to rewrte it for the prodictor
        nom = weights[gaussind] * self._multipdf(datapoint, mues[gaussind], sigmas[gaussind])
        denom = 0
        for gauss in range(self.gaussians):
            denom += weights[gauss] * self._multipdf(datapoint, mues[gauss], sigmas[gauss])
        return float(nom /denom)
    
    
    def _estep(self, data, weights, mues, sigmas):
        '''E-Step, which calculates the posterior occupancy for each datapoint in data '''  
        gammas = np.zeros((self.gaussians,len(data)))
        for gaussind in range(self.gaussians):
            for datap in range(len(data)):
                gammas[gaussind][datap] = self._posteriorocc(data[datap], weights, mues, sigmas, gaussind)
        return gammas
    
    
    def _async_posteriorocc(self, datapoint, weights, mues, sigmas, gaussind,gammas, dataind):
        self.threadpool.put(self._posteriorocc(datapoint, weights, mues, sigmas, gaussind))
        
    def _auxf(self, gammas, weights, data, mues, sigmas):
        '''
        Calculates the auxilliary function
        gammas are all posterior occupancies, dimensions are [ gauss samplesize ]
        weights are all components weights, dimensions are [ gauss ]
        data is the observed data with dimensions of [ samplesize featuresize ]
        mues are all the means, dimensions are [ gauss featuresize ] 
        sigmas are all covariance matrices, dimensions are [ gauss featuresize featuresize ]
        
        '''
        occ_log_weight = 0
        logdet_norm = 0                
        for datap in range(len(data)):
            for gauss in range(self.gaussians):
                occ_log_weight += gammas[gauss][datap] * np.log(weights[gauss])
                x_mue = np.array(data[datap] - mues[gauss])
                sigma_inv = np.linalg.inv(sigmas[gauss])
                logdet_norm += gammas[gauss][datap] * (np.log(np.linalg.det(sigmas[gauss])) + np.dot(np.dot(x_mue.T, sigma_inv), x_mue))
        return occ_log_weight - 0.5 * logdet_norm
    
    
    def fit(self, data,updater='MLE'):
        ''' Estimates a model for the given data, data is a two dimensional array of 
        dimensions [sample_size feature_size]
        Updater can be given as 'MLE' or 'MAP'. MLE estimates are less stable than MAP ones
        '''
        
#         Dimensionality of data vectors are needed here
        self.dim = len(data[0])
        
        mues = np.random.sample((self.gaussians, self.dim))
#         We use qr decomposition to get a matrix q, which is orthogonal and invertible, so
        qrinps = np.random.sample((self.gaussians, self.dim, self.dim))
        qs = []
        
        for qrinp in qrinps:
            qs.append(np.diag(np.diag(qrinp)))
        sigmas = np.array(qs)
#         Initialize the weights uniformly and then apply the probability constraint
        weights = np.ones(self.gaussians)
        sumweights=sum(weights)
        for i in range(len(weights)):
            weights[i] = weights[i] /sumweights
        objfimp_old = 0
        objfimp_new = -100
        
        updater=MLEUpdater(self.gaussians)
        
        for iter in range(self.iterations):
            if (np.abs(objfimp_new - objfimp_old) < self.threshold):
                break
            if math.isnan(objfimp_new - objfimp_old):
                break
            gammas = self._estep(data, weights, mues, sigmas)
            objfimp_old = self._auxf(gammas, weights, data, mues, sigmas)
            updater.update(gammas, weights, data, mues, sigmas)
            objfimp_new = self._auxf(gammas, weights, data, mues, sigmas)
            print "Iteration %i done \n Obj Improvement is : %.5f , LLK : %.5f" % (iter + 1, np.abs(objfimp_new - objfimp_old),objfimp_new)
        self._mues = mues
        self._sigmas = sigmas
        self._weights = weights
        self._gammas = gammas
        print "Training of GMM %s finished!"%(self.label)
        return self
        
        
    def predict(self, data):
        if self._gammas is None:
            raise ValueError("You must first use fit then use predict on that estimated model!")
        ret=[]
        for datap in range(len(data)):
#             Calculate prosterior occupancy and take the maxi value
            tmpmax = 0
            maxi = -10
            for gauss in range(self.gaussians):
                tmpmax = float(self._posteriorocc(data[datap], self._weights, self._mues, self._sigmas, gauss))
                if tmpmax > maxi:
                    maxi = tmpmax
            ret.append((maxi,data[datap],self.label))
        return ret
    
class Updater(object):
    
    __metaclass__=abc.ABCMeta
    
    def __init__(self,gaussians):
        self.gaussians = gaussians
    
    @abc.abstractmethod
    def update(self,gammas, weights, data, mues, sigmas):
        raise NotImplementedError("Implement this class")
    
class MAPUpdater(Updater):
    
    def __init__(self, gaussians):
        Updater.__init__(self, gaussians)
    
    def update(self, gammas, weights, data, mues, sigmas):
        gammaclasssum=[]
        
        mue_mle=[]
        data_mue = 0
        for gauss in range(self.gaussians):
            mue_tmp = 0
            tmpgammaclassum = 0
            data_mue = 0
            for datap in range(len(data)):
                mue_tmp += gammas[gauss][datap] * data[datap]
                tmpgammaclassum += gammas[gauss][datap]
                data_mue += data[datap]
            gammaclasssum.append(tmpgammaclassum)
            mue_mle.append(np.array(mue_tmp/tmpgammaclassum))
            mues[gauss]=((mue_mle[gauss]*gammaclasssum[gauss]+0.01*data_mue)/(0.01*gammaclasssum[gauss]))
        for gauss in range(self.gaussians):
            sigma_class = 0
            for datap in range(len(data)):
                x_mue = np.matrix(data[datap] - mues[gauss])
                sigma_class += gammas[gauss][datap] * np.dot(x_mue.T, x_mue)
                sigma_class = np.diag(np.diag(sigma_class))
            x_mue_mle=np.matrix(mue_mle[gauss] - data_mue)
            x_mue_mle_sq=x_mue_mle.T*x_mue_mle
            mid=(0.01*gammaclasssum[gauss])/(0.01+gammaclasssum[gauss])
            sigmas[gauss] = np.diag(np.diag(mid + x_mue_mle_sq))
        sumclasses = sum(gammaclasssum)
        for gauss in range(self.gaussians):
            weights[gauss] = gammaclasssum[gauss] / sumclasses
    
            
            
            
        
        
        
class MLEUpdater(Updater):
    
    def __init__(self, gaussians):
        Updater.__init__(self, gaussians)
    
    def update(self, gammas, weights, data, mues, sigmas):
        ''' updates the parameters, the M-Step in EM
            data is the current data point
            mues are the means with dimension [ gauss featuresize ]
            sigmas has dims [ gauss featuresize featuresize ]
        '''
        gammaclasssum = []
        for gauss in range(self.gaussians):
            mue_tmp = 0
            tmpgammaclassum = 0
            for datap in range(len(data)):
                mue_tmp += gammas[gauss][datap] * data[datap]
                tmpgammaclassum += gammas[gauss][datap]
            gammaclasssum.append(tmpgammaclassum)
            mues[gauss] = np.array(mue_tmp/tmpgammaclassum)
        
        for gauss in range(self.gaussians):
            sigma_tmp = 0
            for datap in range(len(data)):
                x_mue = np.matrix(data[datap] - mues[gauss])
                sigma_tmp += gammas[gauss][datap] * np.dot(x_mue.T, x_mue)
                sigma_tmp = np.diag(np.diag(sigma_tmp))
            sigmas[gauss] = 1. / gammaclasssum[gauss] * sigma_tmp
        
        sumclasses = sum(gammaclasssum)
        for gauss in range(self.gaussians):
            weights[gauss] = gammaclasssum[gauss] / sumclasses
    
