import numpy as np
import re

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

class StringKernel:
    
    def __init__(self,lambda_param=1, k=1):
        self.l = lambda_param
        self.k = k
    
    def handle_one_example(self, s,t, vocabulary,lambda_param):
        total = 0
        for word in vocabulary:
            for _ in find_all(s,word): #re.finditer(word,s):
                for _ in find_all(t,word):#re.finditer(word,t):
                    total += lambda_param ** (2 * len(word))
        return total
    
    def K_prime(self,i, sx, t):
        if i == 0:
            return 1
        if min(len(sx)-1, len(t)) < i:
            return 0
        x = sx[-1]
        s = sx[:-1]
        total = self.l * self.K_prime(i, s,t)
        for j in range(1,len(t)):
            if t[j] == x:
                power = len(t) - j + 2
                total += self.K_prime(i-1, s, t[:j-1]) * (self.l ** power)
        return total
                          
    def K(self, i, sx, t):
        if min(len(sx)-1, len(t)) < i:
            return 0
        x = sx[-1]
        s = sx[:-1]
        total = self.l * self.K(i, s,t)
        for j in range(1,len(t)):
            if t[j] == x:
                total += self.K_prime(i-1, s, t[:j-1]) * (self.l ** 2)
        return total
    
    def K_hat(self, s,t, diagonal_s, diagonal_t):
        return self.K(self.k, s,t) / np.sqrt(diagonal_s * diagonal_t) 
                          
    
    def build_kernel(self, X1, X2):    
        N = X1.shape[0]
        M = X2.shape[0]
        result = np.zeros((N,M))
        similarite_diagonal = {}
        similarite_diagonal[1] = {}
        similarite_diagonal[2] = {}
        for i in range(N):
            similarite_diagonal[1][i] = self.K(self.k, X1[i],X1[i])
        for j in range(M):
            similarite_diagonal[2][j] = self.K(self.k, X2[j],X2[j])
            
        for i in range(N):
            for j in range(M):
                simi = similarite_diagonal[1][i]
                simj = similarite_diagonal[2][j]
                result[i,j] = self.K_hat(X1[i],X2[j], simi,simj)
        return result