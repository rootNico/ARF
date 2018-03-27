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
    
    def handle_one_example(self, s,t, vocabulary,lambda_param):
        total = 0
        for word in vocabulary:
            for _ in find_all(s,word): #re.finditer(word,s):
                for _ in find_all(t,word):#re.finditer(word,t):
                    total += lambda_param ** (2 * len(word))
        return total
    
    def build_kernel(self, vocabulary, lambda_param=1):    
        def kernel(X1,X2):
            N = X1.shape[0]
            M = X2.shape[0]
            result = np.zeros((N,M))
            for i in range(N):
                for j in range(M):
                    result[i,j] = self.handle_one_example(X1[i],X2[j], vocabulary, lambda_param)
            return result
        
        return kernel