---
layout: post
title: "Approximating the Gaussian - Part 2"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction

This is a continuation of our previous post,
[Approximating the Gaussian - Part 1]({% post_url 2021-07-07-gaussian-approximation-part-1%}).

To briefly recap, we
are trying to approximate a standard Gaussian function, $$f(x) = e^{-x^2}$$, via a series of simpler functions,
$$f_m(x)$$ with the condition that $$f_m(x)$$ have elementary anti-derivatives. In Part 1, we chose
$$f_m(x) = \frac{\alpha_m}{\beta_m + x^2}$$ and wrote

$$
e^{-x^2} = \sum_{m=1}^M\frac{\alpha_m}{\beta_m + x^2}
$$

The coefficients $$\alpha_m$$ and $$\beta_m$$ were obtained via the non-linear least squares (NLLS) regression
procedure implemented using gradient descent in Python.

The trouble with our expansion was the degeneracy of our component functions. Our computed coefficients, while giving
nice fits, converged to wildly different values for every run because of their random initialization.

We concluded Part 1 by mentioning that the cure of degeneracy was to somehow distinguish the component functions.
That is, the component function must be structurally different for every value of $$m$$. In this post we propose
different component functions and obtain a different approximation to the Gaussian function.

## New component functions

Remember that since the Gaussian is defined at $$x\to \pm\infty$$, our component functions must also be so. In
addition, the fits will be better if the derivatives match at $$x = 0$$. A simple but somewhat less elementary
function satisfying these conditions is

$$
f_m(x) = \frac{\sin(mx)}{mx}
$$

This is the familiar [sinc function](https://en.wikipedia.org/wiki/Sinc_function). In fact the sinc function looks
like the Gaussian around $$x = 0$$. But it has wiggles and dips below zero. It will be interesting to see how well we
can approximate the Gaussian using a 'sum of sincs'.

With these new component functions, our series expansion of the Gaussian will be

$$
\begin{equation}
e^{-x^2} = \sum_{m=1}^M\alpha_m \frac{\sin(mx)}{mx}
\label{eq:exp}
\end{equation}
$$

Note that our component functions are now distinguished. This means that we need precise amounts of the component
functions to construct our Gaussian. Because $$\sin(mx)$$ and $$\sin(nx)$$ have different frequencies, deficiencies
from $$f_{m}(x)$$ cannot be offset by $$f_n(x)$$. This was not true for our earlier component functions
$$f_m(x) = \frac{\alpha_m}{\beta_m + x^2}$$; slight decrease in $$\alpha_m$$ here could be offset by equal increase in
$$\alpha_n$$. This led to degeneracy (non-unique solutions) and we hope our new component functions will lead to
unique expansion coefficients.

## Invariance condition

Before we solve for the expansion coefficients $$\alpha_m$$, we can state the invarience condition they must obey.
Recall that
$$\int_{-\infty}^{\infty}e^{-x^2} = \sqrt{\pi}$$ and $$\int_{-\infty}^{\infty}\frac{\sin(mx)}{mx} = \frac{\pi}{m}$$
we integrate both sides of equation $$\eqref{eq:exp}$$ to get

$$
\begin{equation}
\sqrt{\pi} = \sum_{m=1}^M\frac{\pi\alpha_m}{m}
\end{equation}
$$

which leads to an invariance condition

<div class="boxed">
$$
\begin{equation}
\sum_{m=0}^{M}\frac{\alpha_m\sqrt{\pi}}{m} = 1
\label{eq:cc}
\end{equation}
$$
</div>

The interpretaion of the invariance condition is simple. Regardless of how we compute the expansion coefficients
(using gradient descent, analytical formulas, or some unexplainable magic), they must satisfy equation
$$\eqref{eq:cc}$$. We'd derived a similar invariance condition in
[Part 1]({% post_url 2021-07-07-gaussian-approximation-part-1%}). In the following, we will be checking the quality
of our solution by computing the left hand side of equation $$\eqref{eq:cc}$$ and checking how closely it approaches
$1.0$.

## Computing the fit

In Part 1, we implemented a NLLS fitting procedure using gradient descent in Python. The optional last section of
Part 1 contains the details of the implementation (albeit using $$f_m(x) = \frac{\alpha_m}{\beta_m + x^2}$$). The
entire code for fitting and plotting can be found in our
[Github repo](https://github.com/rohan-kekatpure/blog/blob/master/gaussian_approximation/fit_gaussian_sinc.py).

We first try the fit using $$8$$ component functions. In the figure below, the left panel shows how the coefficients
$$\alpha_m$$ converge to their final values. The inset to the left panel shows the value of the invariance condition.
With $$M = 8$$ component functions, we have about $$6\%$$ error. The right panel compares the approximated function
(in red) to the Gaussian function (in black).

<figure>
    <img src="{{site.url}}/assets/img/gaussian_sinc_approx_m8.gif" alt='map' style='margin: 10px;' height="300"/>
</figure>

The explicit expansion with the first five terms looks like

$$
e^{-x^2} \approx 0.2091\frac{\sin(x)}{x} + 0.4090\frac{\sin(2x)}{2x} + 0.2767 \frac{\sin(3x)}{3x} +
0.0895\frac{\sin(4x)}{4x} + 0.0149\frac{\sin(5x)}{5x} + \cdots
$$

Repeated runs with different random initializations converge to the same coefficients. This confirms our
main hypothesis that the expansion using the sum of sincs is non-degenerate.

Notice that while the main lobe of the Gaussian is quite well approximated by the sum of sincs, there are decaying
but visible blips at $x = \pm k\pi$. Increasing the number of component functions does not help. In fact the
coefficients $$\alpha_m$$ after $$m = 5$$ decay exponentially to $0$. The following figure confirms this.

<figure>
    <img src="{{site.url}}/assets/img/gaussian_sinc_approx_combined.gif" alt='map' style='margin: 10px;' height="300"/>
</figure>

The error in the invariance condition (i.e. the fit quality) also does not dip below $6\%$ regardless of the number
of component functions chosen. There are only two possibilities as to why the fit isnt better. It is either
least-squares heuristic or the form of our components functions. Let us dive deeper into it.

## The problem is _linear_

The component functions chosen in Part 1 gave rise to a true non-linear least squares problem. An observant reader
might have noticed that with our new component functions the problem is ordinary linear least squares (OLS) in
disguise. To see that, we will rewrite equation $$\eqref{eq:exp}$$. At any given point $$x_i$$ we have

$$
e^{-x_i^2} = \alpha_1\frac{\sin(x_i)}{x_i} + \alpha_2\frac{\sin(2x_i)}{2x_i}
+ \alpha_3 \frac{\sin(3x_i)}{3x_i} + \cdots
$$

which is of the form

$$
y_i = \alpha_1 z_{1i} + \alpha_2 z_{2i} + \alpha_3 z_{3i} + \cdots
$$

Thus, for all points $$x_i$$ we can form the matrix equation

$$
\begin{equation}
y = Z\alpha
\label{eq:neq1}
\end{equation}
$$

where $$y$$ is the column vector of Gaussian function evaluated at all values $$x_i$$, $$\alpha$$ is the column
vector of unknown expansion coefficients and $$Z$$ is the
[design matrix](https://en.wikipedia.org/wiki/Design_matrix).
Equation $$\eqref{eq:neq1}$$ are simply the normal equations of an OLS problem with the solution

$$
\begin{equation}
\alpha = \left(Z^TZ\right)^{-1}Z^Ty
\end{equation}
$$

The normal equations can be implemented straightforwardly in Python

```python
def _fit_sinc_linear_regression(x, num_base_functions):
    M = num_base_functions
    x = np.clip(x, 1e-6, None)
    mvals = np.arange(1, M + 1, dtype=float).reshape(-1, 1)
    mx = (mvals * x).T
    Z = np.sin(mx) / mx
    y = np.exp(-x * x)
    ztz = Z.T @ Z
    zty = Z.T @ y
    alphas = np.linalg.inv(ztz) @ zty
    return alphas
```

We get good agreement between the expansion coefficients computed using the normal equations and gradient descent.
The numerical comparison between gradient descent and normal equations is given below.

| Coefficient      | Gradient descent | Normal equations     |
| :-:        |    :-   |          :- |
| $$\alpha_1$$      | $0.209173$       | $0.209172559$   |
| $$\alpha_2$$   | $0.4090211$        |  $0.409018987$      |
| $$\alpha_3$$   | $0.276756$       | $0.276752323$      |
| $$\alpha_4$$   | $0.0895346$       | $0.089527342$      |
| $$\alpha_5$$   | $0.0148756$        | $0.014865627$      |

This is a comforting sanity check on our gradient descent implementation. But it doesn't really shed light on why our
error is stuck at $6\%$ and why the coefficients decay rapidly down to zero for $$m > 5$$. For that we need an
analytical understand of the problem.

## Analytical solution


