import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def plotnormalfn(mu=0.0,variance=1.0):
  sigma = math.sqrt(variance)
  x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
  plt.plot(x, stats.norm.pdf(x, mu, sigma))
  plt.show()
  p = 0.0
  n = len(x)
  for i in range(n-1):
    p += (x[i+1]-x[i]) * stats.norm.pdf(x[i], mu, sigma)
  print("p: ",p)
