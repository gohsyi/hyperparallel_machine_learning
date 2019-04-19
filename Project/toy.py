import time
import numpy as np
import tensorflow as tf
from multiprocessing import Pool

def run(fn):
    time.sleep(2)
    print(fn.n)

class N:
    def __init__(self, n):
        self.n = tf.constant([1,2])

if __name__ == "__main__" :
    startTime = time.time()
    testFL = [N(n) for n in range(5)]
    pool = Pool(4)
    pool.map(run, testFL)
    # pool.close()
    # pool.join()
    endTime = time.time()
    print("time :", endTime - startTime)
