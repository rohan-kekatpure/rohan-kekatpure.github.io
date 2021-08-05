---
layout: post
title: "Approximating the Gaussian - Part 1"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction

While of limited theoretical or practical value, tinkering with the
[Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function)
is good entertainment. I often play with the Gaussian in hopes of
finding elementary ways to
[evaluate its definite integral]({% post_url 2021-06-18-solving-definite-integrals-with-plancherels-theorem%}).

On one such recent excursions, I thought of trying the old math trick of expressing a function as a sum of simpler
functions:

$$
\begin{equation}
e^{-x^2} = f_1(x) + f_2(x) + \cdots + f_M(x) = \sum_{m = 1}^Mf_m(x)
\label{eq:gexp}
\end{equation}
$$

_If_ we could express the Gaussian in the above form and _if_ the functions $$f_m(x)$$ had
closed-form anti-derivatives then we could obtain interesting approximations to the Gaussian integral:

$$
\begin{equation}
\int_{-\infty}^{\infty}e^{-x^2} dx = \sum_{m = 1}^M \int_{-\infty}^{\infty}f_m(x) dx
\end{equation}
$$

While the main idea is quite innocent, it turns out that selecting the parametrized functions $$f_m(x)$$ and
evaluating their parameters is tricker than expected.

## Choice of the component functions $$f_m(x)$$

Two plausible candidates $$f_m(x)$$ for the proposed expansion in Equation $$\eqref{eq:gexp}$$
are [Taylor polynomials](https://en.wikipedia.org/wiki/Taylor_series)
and [generalized orthogonal basis functions](https://en.wikipedia.org/wiki/Generalized_Fourier_series).
Both of these candidates do not work.

Since our aim is to integrate the Gaussian between $$\pm\infty$$, each of the $$f_m(x)$$ also needs to be
integrable between $$\pm\infty$$. The Taylor terms, being polynomials, blow up at
$$\pm\infty$$.

As for the generalized Fourier expansion, the most common basis functions are oscillatory and
and finite at $$\pm\infty$$. They also do not have finite integrals between $$\pm\infty$$.

The observation about the Fourier basis functions nevertheless begs an interesting side question:

>must the basis functions always be oscillatory and of finite magnitude at $$\pm\infty$$?

The answer is no; there are many examples of orthonormal basis functions that decay to zero at infinity. In fact, the
wave functions of Quantum Mechanical bound states are guaranteed to form an orthonormal basis and decay to
zero at $$\pm\infty$$. The eigenfunctions of the one-dimensional
[quantum harmonic oscillator](https://bit.ly/2UYezKA) or
[Airy functions](https://en.wikipedia.org/wiki/Airy_function)
are some examples.

Unfortunately, these bound state wave functions do not have elementary anti-derivatives. They are not useful
for simplifying the Gaussian integral. An interesting follow-up question then is:

>can we construct Hermitian operators whose eigenfunctions have elementary anti-derivatives? More generally, given an
orthonormal basis function set, can we construct the corresponding potential well function for the Schrodinger equation?

Answering this extremely interesting question will distract us from our modest current objective.

We turn back to the problem of approximating the Gaussian. For now, we're convinced that neither the Taylor
expansion nor the generalized Fourier expansion would work. Even so, the discussion above has provided us with a
few requirements that our component functions $$f_m(x)$$ must satisfy:

<ol>
<li> $f_m(x)$ must decay to zero away from the origin: $\lim_{x\to\pm\infty}f_m(x) \to 0 $ </li>
<li> $f_m(x)$ must have antiderivatives in terms of elementary functions  </li>
<li> $f_m(x)$ must have a derivative of $0$ at $x = 0$: $f_m'(0) = 0$ </li>
</ol>

The third requirement is not essential, but makes it easier to match the function with $$e^{-x^2}$$ around $$x = 0$$.
 One of the simplest set of functions with the above requirements are

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
side (RHS) expansion with the left-hand side (LHS) function. If we had been lucky enough to be able to use the
generalized Fourier expansion, we could have obtained closed-form formulas for $$\alpha_m$$ and $$\beta_m$$.  But
we're not lucky. Our choice of $$f_m(x)$$ forces us to use a fitting procedure to determine $$\alpha_m$$ and
$$\beta_m$$. Best we can hope is for some exploitable structure in the coefficients $$\alpha_m$$ and $$\beta_m$$.

## An invariance condition

Assume for now that we have computed the coefficients $$\alpha_m$$ and $$\beta_m$$. Now integrate both sides of
equation $$\eqref{eq:fm}$$ to get

$$
\begin{align}
\int_{-\infty}^{\infty}e^{-x^2} dx &= \sum_{m = 1}^M \int_{-\infty}^{\infty}\frac{\alpha_mdx}{\beta_m + x^2}
 \nonumber \\[0.2in]
&= \sum_{m = 1}^M \frac{\pi\alpha_m}{\sqrt{\beta_m}}
\end{align}
$$

Since the integral of the LHS is constant (in fact it is equal to $$\sqrt{\pi}$$) the integral of the RHS must also
be constant. This gives us the invariance condition:

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

## Computing the fit

The fitting procedure is implemented using a simple gradient descent approach. The section at the end of this
post provides the mathematical formulation and its implementation from scratch in Python. Since the math is optional,
the details are pushed to the end. But the math and the code are available for the interested reader. Presently we
study the results of the fit.

We start our study with a small value of $$M$$. Upon performing the fit we obtain the coefficients and can write
the approximation explicitly. For $$M = 3$$ we obtain the reasonable (but uninspiring) approximation:

$$
\begin{equation}
e^{-x^2} \approx \frac{2.0188}{1.2006 + x^2} + \frac{0.4473}{1.1776 + x^2}
- \frac{3.0000}{2.9089 + x^2}
\end{equation}
$$

The invariance condition is also satisfied approximately:

$$
\begin{equation}
\frac{2.0188\sqrt{\pi}}{\sqrt{1.2006}} + \frac{0.4473\sqrt{\pi}}{\sqrt{1.1776}}
- \frac{3.0000\sqrt{\pi}}{\sqrt{2.9089}} \approx 0.87
\end{equation}
$$

To improve the fit we have to increase the number of component functions. We can study
things with $$M=20$$. The figure below shows the convergence of the fitting procedure. The coefficients start with
random initial values. As the gradient descent iterations progress, the coefficients converge to their final values.
Their trajectories look cool, but have no identifiable pattern.

In the inset we show the invariance condition. As the fit converges, the invariance condition
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
invariance condition is also obeyed with reasonable accuracy. Yet the values and the trajectories of the coefficients
are noticeably different in all three cases.

<figure>
    <img src="{{site.url}}/assets/img/gaussian_approx_m20_combined.gif" alt='map' style='margin: 10px;' height="300"/>
</figure>

In ML parlence, our problem has multiple minima of comparable qualities. We suffer from the multiple minima problem for
two reasons. First, fitting Cauchy-like functions (alternatively known as Lorentzian curves) is always
problematic. Even for a single function $$f_m(x) = \frac{\alpha_m}{\beta_m + x^2}$$, many values of $$\alpha_m$$ and
$$\beta_m$$ can evaluate to the same function value.

Second, our component functions are degenerate. That is, for a fixed
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

Our proxy for accuracy is the quality of the invariance condition. For $8$, $20$ and $64$ component functions, the
value of the invariance sum equals $0.90$, $0.92$ and $0.95$.

Another interesting fact is that as we add more component functions, the coefficients cloud
becomes stiffer. The coefficients $$\alpha_m$$ and $$\beta_m$$ have to travel less distance to achieve an accurate
fit. For smaller values of $$M$$, the coefficients have to travel a bit further to obtain an accurate fit. This is an
observational fact and I do not have intuition for why this might be.

## Next steps

The current post is a start of a fruitful direction of investigation. There are some key questions that demand
further exploration.

It is important to somehow constrain the degeneracy in the component functions. The tweaks I tried so far all lead to
degradation in the fit quality. Nonetheless, search is on for more nicer functions.

It is also necessary to try more stable fitting procedures. Most notably, the Gauss-Newton algorithm. The stability
of the procedure will expand the range of functions that can be tried.

An interesting mathematical question to explore is this. Besides the Taylor and the generalized Fourier expansions,
are there other types of basis function expansions which respect the properties that we seek in our component
functions?

We will take up the explorations of the above questions in the follow up posts. The links will be provided here
whenever applicable.

## Epilogue
The maximum number of parameters we considered was
$$128$$ ($$64$$ $$\alpha$$s and $$\beta$$s). Yet it was difficult to identify any pattern in the coefficients.
Imagine how difficult it must be to make sense of deep networks with
[hundreds of billions](https://en.wikipedia.org/wiki/GPT-3)
of parameters. Extrapolate further to biological systems and any search for a first-principles understanding may seem
hopelessly difficult.

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
