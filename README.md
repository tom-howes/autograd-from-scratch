# Autograd From Scratch

Custom PyTorch operators implemented as `torch.autograd.Function` subclasses, where **both the
forward and the backward pass are written by hand** and every gradient is derived analytically
rather than left to autograd.

The point was not to build a faster layer. It was to stop treating `loss.backward()` as a magic
incantation, by deriving the partial derivatives on paper and then checking the implementation
against reference tensors.

---

## The operators

### Fully connected: `fully_connected.py`

Forward:

$$y = xW + b$$

Backward, given the incoming gradient $\frac{\partial z}{\partial y}$:

$$\frac{\partial z}{\partial W} = x^\top \frac{\partial z}{\partial y}
\qquad
\frac{\partial z}{\partial b} = \sum_{t=1}^{T} \frac{\partial z}{\partial y}
\qquad
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} W^\top$$

The bias gradient sums over the batch dimension because the same bias is broadcast across every
example in the batch.

### Generalized logistic: `generalized_logistic.py`

Forward, with three learnable shape parameters, the lower asymptote $l$, the upper asymptote $u$
and the growth rate $g$:

$$y = l + \frac{u - l}{1 + e^{-gx}}$$

This one needs four partials, not one. Writing $\sigma = \dfrac{1}{1 + e^{-gx}}$:

$$\frac{\partial y}{\partial x} = \frac{(u-l)\,g\,e^{-gx}}{\left(1 + e^{-gx}\right)^2}
\qquad
\frac{\partial y}{\partial l} = 1 - \sigma
\qquad
\frac{\partial y}{\partial u} = \sigma
\qquad
\frac{\partial y}{\partial g} = \frac{(u-l)\,x\,e^{-gx}}{\left(1 + e^{-gx}\right)^2}$$

The gradient with respect to $g$ is the awkward one, since $g$ sits inside the exponent and the
quotient rule has to be carried through it.

### Mean squared error: `mean_squared_error.py`

Forward, over a $T \times n$ batch:

$$y = \frac{1}{Tn}\sum (x_1 - x_2)^2$$

Backward:

$$\frac{\partial y}{\partial x_1} = \frac{2(x_1 - x_2)}{Tn}
\qquad
\frac{\partial y}{\partial x_2} = -\frac{2(x_1 - x_2)}{Tn}$$

---

## The layers

`fully_connected_layer.py` wraps the operator in an `nn.Module` and initialises the weights with
Xavier uniform sampling, drawing from $\left[-\sqrt{6/(n_\text{in}+n_\text{out})},\
\sqrt{6/(n_\text{in}+n_\text{out})}\right]$ to keep activation variance stable across depth.

`generalized_logistic_layer.py` exploits the fact that the generalized logistic **contains both
tanh and sigmoid as special cases**, so one operator covers both activations:

| Activation | $l$ | $u$ | $g$ |
|---|---|---|---|
| `tanh`    | -1 | 1 | 2 |
| `sigmoid` |  0 | 1 | 1 |

`create_net.py` assembles an `nn.Sequential` from a specification: a list of hidden layer widths
and a matching list of activation names, with a linear output layer on the end.

## Training

`train.py` runs SGD with momentum and weight decay, a `StepLR` schedule with gamma decay,
cross-entropy loss and shuffled mini-batches, tracking loss and accuracy per epoch and plotting
both curves at the end.

## Running it

```bash
pip install torch matplotlib
python demo_xor.py
```

`demo_xor.py` generates a noisy XOR dataset in memory and trains a small network built from these
components. XOR is the useful smoke test here because it is not linearly separable, so it only
converges if the non-linearity and its gradient are both correct. A two-hidden-layer network of
16 tanh units reaches roughly 95% training accuracy in 40 epochs.

## Provenance

These components were written for **CS7150 Deep Learning** at Northeastern University. The
implementations, the derivations and the layer design are mine. The course's verification harnesses
and datasets are deliberately not included here.
