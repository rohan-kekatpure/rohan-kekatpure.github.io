---
layout: post
title: "Approximating the Gaussian with simpler bell curves"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction
In this post we are going to tinker a bit with the simple one dimensional
[Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function).
Specifically, we will examine if it is possible to break down the Gaussian into a series of simpler functions. My
original motivation for seeking a series expansion of the Gaussian was my ongoing curiosity about
Gaussian function. I'm always
[in pursuit]({% post_url 2021-06-18-solving-definite-integrals-with-plancherels-theorem%})
of simple ways to evaluate its integral (which equals $$\sqrt{\pi}$$). On one such recent excursions, I thought of
trying to break down the Gaussian into simpler functions:

$$
\begin{equation}
e^{-x^2} = f_1(x) + f_2(x) + \cdots + f_M(x) = \sum_{m = 1}^Mf_m(x)
\label{eq:gexp}
\end{equation}
$$

_If_ we could express the Gaussian in the above form and _if_ the functions $$f_m(x)$$ had
closed-form anti-derivatives, then we could obtain interesting series approximations to the Gaussian integral:

$$
\begin{equation}
\int_{-\infty}^{\infty}e^{-x^2} dx = \sum_{m = 1}^M \int_{-\infty}^{\infty}f_m(x) dx
\label{eq:ser}
\end{equation}
$$

The main idea above is quite innocent. However, it turns out that selecting the functions $$f_m(x)$$ and
evaluating their parameters is tricker than expected. This exercise will take us through a tour of linear
algebra, nonlinear least squares, gradient descent and Fourier series.

## Choice of the component functions $$f_m(x)$$

Decomposing functions into series of simpler functions is an old and a well understood concept.
Two plausible ways to obtain the expansion in Equation $$\eqref{eq:gexp}$$
are [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series)
and [Fourier expansion](https://en.wikipedia.org/wiki/Generalized_Fourier_series).

However, our goal is to obtain an approximation for the _integral_ of the Gaussian from $$-\infty$$ to $$+\infty$$.
To achieve this, the series in equation $$\eqref{eq:ser}$$ must be integrable _term by term_. That is, each
$$f_m(x)$$ needs to be integrable between $$\pm\infty$$.

The terms of the Taylor expansion, being polynomials, blow up at $$\pm\infty$$. The terms of common Fourier
expansions are oscillatory and finite at $$\pm\infty$$. They too are not integrable between $$\pm\infty$$.

An interesting side question is this:

>must the Fourier basis functions always be oscillatory and of finite magnitude at $$\pm\infty$$?

The answer is no; there are many examples of orthonormal basis functions that decay to zero at infinity. In fact, the
wave functions of Quantum Mechanical bound states are guaranteed to form an orthonormal basis and decay to
zero at $$\pm\infty$$. The eigenfunctions of the one-dimensional
[quantum harmonic oscillator](https://bit.ly/2UYezKA) or
[Airy functions](https://en.wikipedia.org/wiki/Airy_function)
are some examples.

Unfortunately, these bound state wave functions do not have elementary anti-derivatives, which is one of our
requirements. An interesting follow-up question then is:

>can we construct Hermitian operators whose eigenfunctions have elementary anti-derivatives? More generally, given an
orthonormal basis function set, can we construct the corresponding potential well function for the Schrodinger equation?

Exploring this intriguing question will distract us from our current objective.

For now, we're convinced that neither Taylor nor the Fourier expansion would work. Even so, the discussion above has
provided us with requirements that our component functions $$f_m(x)$$ must satisfy in order to be useful for
approximating the Gaussian integral:

<ol>
<li> Each $\vert f_m(x)\vert$ must decay to zero away from the origin:
$\lim_{x\to\pm\infty}\vert f_m(x)\vert \to 0$ </li>
<li> Each $f_m(x)$ must be flat at $x = 0$: $f_m'(0) = 0$ </li>
<li> Each $f_m(x)$ must have a known closed-form integral between $\pm\infty$  </li>
</ol>

Spend a moment to visualize the requirements (1) and (2). It will become clear that each $$f_m(x)$$ must
itself be a bell-like curve. The third requirement, while not necessary, helps narrow down the choice of possible
candidate functions. It is trying to make sure that our problem doesn't become a purely numerical exercise and that
we get interesting series when we plug back $$\int_{-\infty}^{\infty}f_m(x)dx $$ in equation $$\eqref{eq:ser}$$.

Our original problem can now be viewed as decomposing the Gaussian into simpler bell curves. Before reading further,
try and see if you can come up with functional forms for bell curves which satisfy the above requirements.

In theory, functions satisfying the above conditions can be constructed in an infinite number of ways. In practice I
found it harder than expected to come up with formulas for bell curves, but eventually stumbled on three alternatives.

The three functional forms belong to rational, exponential, and trigonometric class of functions. Each class reveals
unique and interesting convergence behavior to the Gaussian.

The parametrized form of our three function families can be written as follows:

<ol>
<li> Rational function bell curves: $f_m(x) = \frac{\alpha_m}{\beta_m + x^2}$. The expansion looks like
$$
e^{-x^2} = \frac{\alpha_1}{\beta_1 + x^2} + \frac{\alpha_2}{\beta_2 + x^2} + \cdots
$$
</li>
<br>
<li> Exponential function bell curves: $f_m(x) = \alpha_m\,\text{sech}^2(mx)$. The expansion looks like
$$
e^{-x^2} = \alpha_1\text{sech}^2(x) + \alpha_2\text{sech}^2(2x) + \cdots
$$
</li>
<br>
<li> Trigonometric function bell curves: $f_m(x) = \alpha_m\, \frac{\sin(mx)}{mx}$. The expansion looks like
$$
e^{-x^2} = \alpha_1\frac{\sin(x)}{x} + \alpha_2\frac{\sin(2x)}{2x} + \cdots
$$
</li>
</ol>

It is easy to check that each of the proposed $$f_m(x)$$ satisfy conditions (1) and (2).

Note that the expansions in terms of our proposed functions $$f_m(x)$$ are not guaranteed. Just because we formally
wrote the expansion does not imply that it will hold. We have not shown the set of functions $$f_m(x)$$ to be
[complete](https://mathworld.wolfram.com/CompleteOrthogonalSystem.html).
Hence there are no theoretical guarantees that the above expansions will converge to Gaussian.

Intuitively however, it must be possible to express the Gaussian as a combination of other bell curves. But how do we
choose the parameters of $$f_m(x)$$ that give best possible approximation to the Gaussian function? Had
we used the Taylor or Fourier expansions, we could have obtained closed-form formulas for the coefficients
$$\alpha_m$$ and $$\beta_m$$. Without any theory, we will be forced to use a fitting procedure.

The coefficients $$\alpha_m$$ and $$\beta_m$$ will therefore be determined using least squares fit of the right-hand
side (RHS) expansion with the left-hand side (LHS) function.

## Quality of fit

The quality of a fit is typically measured using the
[$$R^2$$ metric](https://en.wikipedia.org/wiki/Coefficient_of_determination). The $$R^2$$ metric tells us how well
the formula approximates the given data on a point-by-point basis. The values of the parameters are not usually
relevant and do not feature in an $$R^2$$ calculation.

For our current problem though, the coefficients _do_ mean something. In fact, the coefficients are bound by a
conservation law (or, alternatively, and invariance condition) that depends on the form of $$f_m(x)$$. To see this,
we can carry out the integration in equation $$\eqref{eq:ser}$$ (reproduced below)

$$
\begin{equation}
\int_{-\infty}^{\infty}e^{-x^2} dx = \sum_{m = 1}^M \int_{-\infty}^{\infty}f_m(x) dx
\end{equation}
$$

The LHS is known to be $$\sqrt{\pi}$$ and the RHS is also known for our three candidate $$f_m(x)$$:

$$
\begin{equation}
\int_{-\infty}^{\infty}f_m(x) dx =
\begin{cases}
\frac{\pi\alpha_m}{\sqrt{\beta_m}} &\qquad \text{ for  } f_m(x) = \frac{\alpha_m}{\beta_m + x^2} \\[0.1in]
\frac{2\alpha_m}{m} &\qquad \text{ for  } f_m(x) = \alpha_m\text{sech}^2(mx) \\[0.1in]
\frac{\pi\alpha_m}{m} &\qquad \text{ for  } f_m(x) = \alpha_m\frac{\sin(mx)}{mx}
\end{cases}
\end{equation}
$$

This leads to the invariance conditions for the coefficients as follows

<div class="boxed">
$$
\begin{align}
\label{eq:cc1}
\sum_{m = 1}^M\frac{\alpha_m\sqrt{\pi}}{\sqrt{\beta_m}} &= 1 &\text{ for } f_m(x) = \frac{\alpha_m}{\beta_m +
x^2}  \\[0.1in]
\label{eq:cc2}
\sum_{m = 1}^M\frac{2\alpha_m}{m\sqrt{\pi}} & = 1 &\text{ for } f_m(x) = \text{sech}^2(x) \\[0.1in]
\label{eq:cc3}
\sum_{m = 1}^M\frac{\alpha_m\sqrt{\pi}}{m} &= 1 &\text{ for } f_m(x) = \alpha_m\frac{\sin(mx)}{mx}
\end{align}
$$
</div>

The boxed equations above express a remarkable fact: regardless of the numerical value of individual
coefficients, they are, in aggregate, bounded by the conservation condition. This fact is not deep.
It is obvious from the formal series expansion in equation $$\eqref{eq:ser}$$. But it contrasts with
experience we typically have with least-squares regression. In a typical least-squares fit, the coefficients are free
to assume any value. Here we are trying to approximate a well-known function. So the coefficients have extra
constraints.

These constraints help us formulate a more interesting way to measure convergence. Once the fit is computed, we will
measure its quality not by its $$R^2$$ but by how well the computed coefficients approximate $$\sqrt{\pi}$$. The LHS of
equations $$\eqref{eq:cc1}$$, $$\eqref{eq:cc2}$$ and $$\eqref{eq:cc3}$$ express this convergence in simpler terms.
That is, we simply have to see how close the LHS approaches $1.0$ rather than having to remember $$\sqrt{\pi}$$.

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

We will keep the presentation general by retaining $$f_m(x)$$ for the $m$th component function and use specific forms
of $f_m(x)$ only when we need the final formulas.

$$
\begin{equation}
L = \sum_{i=1}^{N}\left(e^{-x_i^2} - \sum_{m=1}^M f_m(x)\right)^2
\end{equation}
$$

The second step is to compute the derivative of $$L$$ with respect to the parameters:

<div class="boxed">
$$
\begin{align}
\frac{\partial L}{\partial\alpha_m} &= \sum_{i=1}^{N} 2 \frac{\partial f_m(x)}{\partial\alpha_m}
\left(e^{-x_i^2} - \sum_{m=1}^M f_m(x)\right) \\[0.2in]
\frac{\partial L}{\partial\beta_m} &= \sum_{i=1}^{N} 2 \frac{\partial f_m(x)}{\partial\beta_m}
\left(e^{-x_i^2} - \sum_{m=1}^M f_m(x)\right)
\end{align}
$$
</div>

We can obtain explicit formulas for the gradients by substituting $$f_m(x) = \frac{\alpha_m}{\beta_m + x^2}$$:

$$ \begin{align}
\frac{\partial L}{\partial\alpha_m} &= \sum_{i=1}^{N} \frac{-2}{\beta_m + x_i^2}
\left(e^{-x_i^2} - \sum_{m=1}^M \frac{\alpha_m}{\beta_m + x_i^2}\right) \\[0.2in]
\frac{\partial L}{\partial\beta_m} &= \sum_{i=1}^{N} \frac{2\alpha_m}{\left(\beta_m + x_i^2\right)^2}
\left(e^{-x_i^2} - \sum_{m=1}^M \frac{\alpha_m}{\beta_m + x_i^2}\right)
\end{align} $$

The final step is the implementation of the gradient descent update equations:

$$
\begin{align}
\alpha_m &\leftarrow \alpha_m - \eta\frac{\partial L}{\partial\alpha_m} \\[0.1in]
\beta_m &\leftarrow \beta_m - \eta\frac{\partial L}{\partial\beta_m}
\end{align}
$$

where $\eta$ is the learning rate. The Python implementation of the three steps is simpler than their mathematical
appearence. The code snippet below shows the essence of the implementation. The complete code can be found in our
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
