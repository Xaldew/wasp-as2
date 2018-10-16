import numpy as np
import matplotlib.pyplot as plt

from GPyOpt.methods import BayesianOptimization
from matplotlib2tikz import save as tikz_save

def f(x):
    return (6*x - 2)**2 * np.sin(12*x-4)


domain = [{ 'name': 'var_1',
            'type': 'continuous',
            'domain': (0, 1)}]

myBopt = BayesianOptimization(f=f, domain=domain)
myBopt.run_optimization(max_iter=5)
myBopt.plot_acquisition()

"""
tikz_save("example_prior.tex", 
            figure=fig1, 
            figureheight='\\figureheight', 
            figurewidth='\\figurewidth')

tikz_save("example_post.tex", 
            figure=fig2, 
            figureheight='\\figureheight', 
            figurewidth='\\figurewidth')
"""





