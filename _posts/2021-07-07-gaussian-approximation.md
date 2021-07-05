---
layout: post
title: "Approximating the Gaussian"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction

I often tinker with the
[Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function)
searching for elementary ways
to [evaluate its definite integral]({% post_url 2021-06-18-solving-definite-integrals-with-plancherels-theorem%}).
Arguably, this exercise has limited practical value, but it is good entertainment.

Recently, I thought of using a well-known math trick of expressing a complicated function as a sum of more
elementary functions:

$$
\begin{equation}
e^{-x^2} = f_1(x) + f_2(x) + \cdots + f_M(x) = \sum_{m = 1}^Mf_m(x)
\label{eq:gexp}
\end{equation}
$$

The main idea being that _if_ we could express the Gaussian in the above form and _if_ the functions $$f_m(x)$$ had
closed-form anti-derivatives then we could obtain interesting approximations to the Gaussian integral:

$$
\begin{equation}
\int_{-\infty}^{\infty}e^{-x^2} dx = \sum_{m = 1}^M \int_{-\infty}^{\infty}f_m(x) dx
\end{equation}
$$

While the main thought is straightforward, it turns out that selecting the functions $$f_m(s)$$ is tricker than
expected.

## Choice of the basis functions $$f_m(x)$$

Two plausible candidates $$f_m(x)$$ for the proposed expansion in Equation $$\eqref{eq:gexp}$$
are [Taylor polynomials](https://en.wikipedia.org/wiki/Taylor_series)
and [generalized orthogonal basis functions](https://en.wikipedia.org/wiki/Generalized_Fourier_series).
Both ideas fail because of the requirement on the $$f_m(x)$$ to be integrable between the limits $$\pm\infty$$.
The polynomials in the Taylor expansion blow up at $$\pm\infty$$ and most common Fourier basis functions are oscillatory
at $$\pm\infty$$. The second observation begs an interesting side question:

>Are there examples of orthogonal basis functions integrable between $$\pm\infty$$ or must the basis functions be
oscillatory and of finite magnitude at $$\pm\infty$$?

The answer is that there are orthonormal basis functions that decay to zero at infinity. The wave functions
of bound states in Quantum Mechanics are, by construction, required to (a) decay to zero at $$\pm\infty$$ and (2)
orthonormal. The eigenfunctions of the one-dimensional
[quantum harmonic oscillator](https://bit.ly/2UYezKA)
are an explicit example. Unfortunately, these functions do not have elementary anti-derivatives and are not useful
for simplifying the Gaussian integral. An interesting follow-up question then is:

>can we construct potential wells which, when plugged into Schrodinger's equation, yield orthonormal solution sets
whose eigenfunctions have elementary anti-derivatives? More generally, can we construct potential wells given an
orthonormal basis set of functions?

Unfortunately, answering this extremely interesting question is beyond my current mathematical ability.

Having exhausted the obvious solutions, we turn back to the problem of approximating the Gaussian. The above
discussion has made it clear that our functions $$f_m(x)$$ should satisfy a few requirements:

<ol>
<li> The functions must decay to zero away from the origin: $\lim_{x\to\pm\infty}f_m(x) \to 0 $ </li>
<li> $f_m(x)$ must have antiderivatives in terms of elementary functions  </li>
<li> Derivative of $f_m(x)$ must vanish at $x = 0$: $f_m'(0) = 0$ </li>
</ol>

Since $$\frac{d}{dx}e^{-x^2} = 0$$ at $$x = 0$$, the third requirement makes it easier to match the
function with $$e^{-x^2}$$ around $$x = 0$$. One of the simplest set of functions with the above requirements are

$$
f_m(x) = \frac{\alpha_m}{\beta_m + x^2}
$$

With these functions, our proposed decomposition of $$e^{-x^2}$$ in terms of $$f_m(x)$$ can be written as

$$
\begin{equation}
e^{-x^2} = \sum_{m=1}^M\frac{\alpha_m}{\beta_m + x^2}
\label{eq:fm}
\end{equation}
$$

where $$\alpha_m$$ and $$\beta_m$$ will be determined by obtaing the best fit of the right-hand
side (RHS) expansion with the left-hand side (LHS) function. Notice that if we were lucky enough to be able to use
generalized Fourier expansion, we could have obtained closed-form formulas for $$\alpha_m$$ and $$\beta_m$$.  But
we're not lucky. We're forced to use a fitting procedure to determine $$\alpha_m$$ and $$\beta_m$$. Our best bet is
to see if there is at least some exploitable structure in $$\alpha_m$$ and $$\beta_m$$.

## Conservation condition on $$\alpha_m$$ and $$\beta_m$$

In the next section we will write code to obtain the fitting parameters. Assume for now that we have all
$$\alpha_m$$ and $$\beta_m$$ and integrate both sides of equation $$\eqref{eq:fm}$$ to get

$$
\begin{align}
\int_{-\infty}^{\infty}e^{-x^2} dx &= \sum_{m = 1}^M \int_{-\infty}^{\infty}\frac{\alpha_mdx}{\beta_m + x^2}
 \nonumber \\[0.2in]
&= \sum_{m = 1}^M \frac{\pi\alpha_m}{\sqrt{\beta_m}}
\end{align}
$$

Since the integral of the LHS is constant (in fact it is equal to $$\sqrt{\pi}$$) the integral of the RHS must also
be constant. This gives us the conservation condition:

$$
\begin{equation}
\sum_{m=0}^{M}\frac{\pi\alpha_m}{\sqrt{\beta_m}} = \text{ constant }
\end{equation}
$$

Since we know that the constant is equal to $$\sqrt{\pi}$$ we can slightly simplify the condition so that it is
easier to remember:

<div class="boxed">
$$
\begin{equation}
\sum_{m=0}^{M}\frac{\alpha_m\sqrt{\pi}}{\sqrt{\beta_m}} = 1
\label{eq:cc}
\end{equation}
$$
</div>

The boxed equation above expresses a remarkable fact. No matter what our fit gives us, the computed coefficients
must always satisfy equation $$\eqref{eq:cc}$$.

## Computing the fitting coefficients

The number of basis functions is a free parameter and we can start things with $$M=20$$. The figure below shows the
convergence of the fitting procedure. The coefficients start with random initial values. As the gradient descent
iterations progress, the coefficients converge to their final values. Their trajectories look cool, but have no
identifiable pattern.

In the inset we show the conservation condition. As the fit converges, the conservation condition
$$\sum_{m=1}^M\frac{\alpha_m\sqrt{\pi}}{\sqrt{\beta_m}}$$ approaches $1.0$. Due to the nature of the approximation
the final value is close to $0.92$ in stead of $1.0$.

The right hand side shows how the quality of the final fit the Gaussian obtained using $20$ component functions $$f_m
(x)$$. One of the things to realize is that the tails of a Gaussian decay faster than any polynomial. It is quite
hard to approximate this fall off using rational polynomial functions like we have chosen. This fact is obvious when
we look closely at the quality of the fit at the tails.

<figure>
    <img src="{{site.url}}/assets/img/gaussian_approx_m20.gif" alt='map' style='margin: 10px;' height="300"/>
<!--     <figcaption>Figure 1. Integrand of the Gamma function and its comparison with a scaled Gaussian.</figcaption> -->
</figure>

## Solutions are not unique !

Even for a fixed value of $$M$$, the values as well as the orbits of the fitting coefficients vary wildly. The figure
below shows three successive runs of the fitting procedure for $$M=20$$. In all three cases the fit is decent. The
conservation condition is also obeyed with reasonable accuracy. Yet the values and the trajectories of the coefficients
are noticeably different in all three cases.

<figure>
    <img src="{{site.url}}/assets/img/gaussian_approx_m20_combined.gif" alt='map' style='margin: 10px;' height="300"/>
</figure>

In ML parlence, our problem has multiple minima of comparable qualities. We suffer from the multiple minima problem for
two reasons. First, fitting Cauchy-like functions (alternatively known as Lorentzian curves) is always
problematic. Even for a single function $$f_m(x) = \frac{\alpha_m}{\beta_m + x^2}$$, many values of $$\alpha_m$$ and
$$\beta_m$$ can evaluate to the same function value.

Second, our basis functions are degenerate. That is, for a fixed
$$\beta$$ we can obtain a given function value as $$\frac{0.2}{\beta + x^2} + \frac{0.8}{\beta + x^2}$$ or as
$$\frac{0.6}{\beta + x^2} + \frac{0.4}{\beta + x^2}$$. Thus we should expect a lot of degenerate solutions. The cure
for the degeneracy is to somehow distinguish (or unique-ify) the component functions $$f_m(x)$$. This is one line of
work we're pursuing currently.

Presently, the multiple minima problem unfortunately prevents us from exploiting any structure in the coefficients.
The procedure in this post will be practically useful only if we could determine $$\alpha_m$$ and $$\beta_m$$
_without_ any elaborate fitting procedure. If we're able to obtain unique solution, then perhaps we can obtain
interesting identities we originally sought.

## More component functions = more accuracy

Despite the issues mentioned above, one aspect is consistent with our intuition. Our fits become more accurate with
more component functions. The plot below shows the fits for $$M = 8$$, $$20$$ and $$64$$ component functions.

<figure>
    <img src="{{site.url}}/assets/img/gaussian_approx_varm.gif" alt='map' style='margin: 10px;' height="300"/>
</figure>

Our proxy for accuracy is the quality of the conservation condition. For $8$, $20$ and $64$ component functions the
value of the conservation sum equals $0.90$, $0.92$ and $0.95$.

Another interesting fact is that as we add more component functions, the cloud of the coefficients
becomes 'stiffer'. The coefficients $$\alpha_m$$ and $$\beta_m$$ have to travel less distance to achieve an accurate
fit. For smaller values of $$M$$, the coefficients have to travel a bit further to obtain an accurate fit. This is an
observational fact and I do not have intuition for why this might be.

## Optional: Gradient descent implementation

This section is a bit mathematical and is optional for those unfamiliar as well as very familiar with machine
learning (ML)! The mathematical details are presented here primarily for my own notes and secondarily as an
illustration of how to set up a complete ML computation from scratch.

From an ML perspective, we have a
[nonlinear least squares](https://en.wikipedia.org/wiki/Non-linear_least_squares)
fitting problem (NLLS) on our hand.
The [Gauss-Newton method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
is perhaps the ideal choice for its solution. But we will instead go down an easier route of implementing a simple
gradient descent fitting algorithm. I may later add a section on Gauss-Newton fitting procedure.

The gradient descent is implemented in the standard way. We discretize $$e^{-x^2}$$ using $$N$$ points on the
$X$-axis. At each point $$x_i$$, we compute the squared residual between the actual function value and the estimated
function value. In other words, the squared residual is simply the squared difference between the LHS and the RHS of
equation $$\eqref{eq:fm}$$. Finally we sum up all $$N$$ residuals to obtain the total loss function.

$$
\begin{equation}
L = \sum_{i=1}^{N}\left(e^{-x_i^2} - \sum_{m=1}^M \frac{\alpha_m}{\beta_m + x_i^2}\right)^2
\end{equation}
$$

The second step is to compute the derivative of $$L$$ with respect to the parameters:

$$
\begin{align}
\frac{\partial L}{\partial\alpha_m} &= \sum_{i=1}^{N} \frac{-2}{\beta_m + x_i^2}
\left(e^{-x_i^2} - \sum_{m=1}^M \frac{\alpha_m}{\beta_m + x_i^2}\right) \\[0.2in]
\frac{\partial L}{\partial\beta_m} &= \sum_{i=1}^{N} \frac{2\alpha_m}{\left(\beta_m + x_i^2\right)^2}
\left(e^{-x_i^2} - \sum_{m=1}^M \frac{\alpha_m}{\beta_m + x_i^2}\right)
\end{align}
$$

The final step is the implementation of the gradient descent update equations:

$$
\begin{align}
\alpha_m &\leftarrow \alpha_m - \eta\frac{\partial L}{\partial\alpha_m} \\[0.1in]
\beta_m &\leftarrow \beta_m - \eta\frac{\partial L}{\partial\beta_m}
\end{align}
$$

The Python implementation of the three steps is simpler than their mathematical appearence. The code snippet below
shows the essence of the implementation. The complete code can be found in our
[Github](https://github.com/rohan-kekatpure/blog/blob/master/gaussian_approximation/gaussian.py).

```python
def compute_fit(x, num_base_functions, niters,
                learning_rate, regularization_param):
    M = num_base_functions
    reg = regularization_param
    alphas = np.random.random((M, 1))
    betas = np.random.random((M, 1))

    for t in range(niters):
        v1 = alphas / (betas + x * x)
        r = np.exp(-x * x) - v1.sum(axis=0)
        q = betas + x * x
        v3 = -2.0 / q * r
        v4 = (2.0 * alphas) / (q * q) * r
        dla = v3.sum(axis=1).reshape(-1, 1)
        dlb = v4.sum(axis=1).reshape(-1, 1)
        alphas -= learning_rate * dla
        betas -= learning_rate * dlb

    return alphas, betas
```
