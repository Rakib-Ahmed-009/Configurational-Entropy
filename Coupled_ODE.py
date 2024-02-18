import autograd.numpy as np
from autograd import grad, elementwise_grad, jacobian, hessian
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt

def ode(t, C):
    Ca, Cb = C
    d2Cadt = -(2/t) * dCadt - 2*Ca*(9/5*(Ca-1)**2+1/5*(Cb-1)**2-0.4) + (Ca**2+Cb**2)*18/5*(Ca-1)
    d2Cbdt = -(2/t) * dCbdt - 2*Cb*(9/5*(Ca-1)**2+1/5*(Cb-1)**2-0.02) + (Ca**2+Cb**2)*2/5*(Cb-1)
    return [d2Cadt, d2Cbdt]

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    "see https://arxiv.org/pdf/1710.05941.pdf"
    return x / (1.0 + np.exp(-x))

def C(params, inputs):
    "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)
    return outputs

jac = jacobian(C, 1)
hes = hessian(C, 1)
def dCdt(params, t):
    i = np.arange(len(t))
    return jac(params, t)[i, :, i].reshape((len(t), 2))
def d2Cdt(params, t):
    i = np.arange(len(t))
    return hes(params, t)[i, :, i].reshape((len(t), 2))

# initial guess for the weights and biases
t = np.linspace(1, 5, 25)
params = init_random_params(0.1, layer_sizes=[1, 8, 2])
i = 0    # number of training steps
N = 501  # epochs for training
et = 0.0 # total elapsed time
def objective(params, step):
    Ca, Cb = C(params, t).T
    dCadt, dCbdt, dCcdt = dCdt(params, t).T
    d2Cadt, d2Cbdt, d2Ccdt = d2Cdt(params, t).T
    z1 = np.sum((d2Cadt -(-(2/t) * dCadt - 2*Ca*(9/5*(Ca-1)**2+1/5*(Cb-1)**2-0.4) + (Ca**2+Cb**2)*18/5*(Ca-1))**2))
    z2 = np.sum((d2Cbdt-(-(2/t) * dCbdt - 2*Cb*(9/5*(Ca-1)**2+1/5*(Cb-1)**2-0.02) + (Ca**2+Cb**2)*2/5*(Cb-1)))**2)
    ic1 = (dCadt[0]-0)**2
    ic2 = (dCbdt[0]-0)**2
    ic3 = (Ca[24]-0)**2
    ic4 = (Cb[24]-0)**2
    return z1 + z2 + ic1 + ic2 + ic3 + ic4

def callback(params, step, g):
    if step % 100 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step)))

objective(params, 0)  # make sure the objective is scalar

import time
t0 = time.time()

params = adam(grad(objective), params,
              step_size=0.001, num_iters=N, callback=callback)

i += N
t1 = (time.time() - t0) / 60
et += t1

plt.plot(t, C(params, t),'--')
plt.legend(['PhiA', 'PhiB'])
plt.xlabel('Time')
plt.ylabel('$\Phi$')
plt.show()

