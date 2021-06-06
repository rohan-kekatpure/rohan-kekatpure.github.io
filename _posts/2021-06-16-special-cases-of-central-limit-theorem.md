---
layout: post
title: "Laplace's method, Stirling's Approximation and the special cases of the Central Limit Theorem"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction

The [Central Limit Theorem]() (CLT) states that the distribution of the average value of $$n$$ random variables drawn
from any distribution with a finite mean $$\mu$$ and a variance $$\sigma^2$$approaches a (scaled) normal distribution
. In notation, if $$x_1, x_2, \ldots, x_n$$ are drawn from a distribution, and

$$
\bar{x} = \frac{1}{n} \left(x_1 + x_2 + \cdots + x_n\right)
$$

then

$$
\bar{x} \sim \mathcal{N}(\mu, \frac{\sigma^2}{n})
$$

The proof of the general version of the CLT is arrived at by starting with the
[characteristic function](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)) (CF) of the
probability distribution function (PDF). By the rules of Fourier transform, it turns out that the CF of the
sample sum is proportional to the $$n$$th power of the CF of the original random variable. From here this $$n$$th power
is manipulated to obtain a normal distribution. While the proof can be found in standard probability theory texts,
the details can seem a bit heavy without prior context (they certainly were for the author).

Our aim in this post is to use a lot less math to prove that some common distributions tend to a normal distribution.
The steps of our technique are general: they apply uniformly to all the distributions considered here and possibly
more. We will first scketch the steps of the proof and apply then systematically to prove that Binomial, Poisson and
the Gamma distributions converge to a normal distribution and extract the means and variances of the asymptotic
normal distributions they converge to.

## Steps to prove convergence to normal distribution

To prove that any function converges to a normal distribution is equivalent to proving that the logarithm of the
function converges to an inverted parabola (kind of like "$$\cap$$"). To prove that an arbitrary function is
(an inverted) parabola, we have to compute its Taylor expansion up to the quadratic term and show that the higher
order terms shrink fast. This process is the essence of the
[Laplace's method]({% post_url 2021-06-08-laplaces-method%}).

In detail, the steps we will follow in each of the following proofs are as follows:

 1. Let $$g(x)$$ be the given PDF, and $$f(x) = \log g(x)$$
 2. Solve for $$f'(x) = 0$$, call this solution $$\mu$$
 3. Verify $$f''(x_0) < 0$$
 4. Write $$f(x) \approx f(x_0) - \frac{\vert f''(\mu)\vert}{2} (x - \mu)^2$$
 5. Finally write $$g(x) \approx e^{f(\mu)}\,e^{-\frac{1}{2}\vert f''(\mu)\vert(x-\mu)^2}$$

If the given PDF $$g(x)$$ is normalized, then, in step 5 above, the factor $$e^{f(\mu)}$$ turns out to be
$$\sqrt{\frac{\vert f''(\mu)\vert}{2\pi}}$$.

## Poisson $$\to$$ Normal

The PDF of Poisson distribution is

$$
g(x) = \frac{\lambda^x e^{-\lambda}}{\Gamma(x + 1)}
$$

Now lets carry out steps 1--5 above.

$$
\begin{align}
f(x) &= \log g(x) = x\log \lambda - \lambda - \log \Gamma(x + 1) \notag \\
&\approx x\log \lambda - \lambda - (x\log x - x + \log\sqrt{2\pi x}) \notag \\
f'(x) &= \log\lambda - \log x - \frac{1}{2x} \label{eq:poissonf1} \\
f''(x) &= -\frac{1}{x} + \frac{1}{2x^2} \label{eq:poissonf2}
\end{align}
$$

Solving for $$f'(x) = 0$$ gives us $$\mu = \lambda$$. At $$x = \mu$$, $$f''(\mu) \approx -1/\lambda$$ and
$$f(\mu) = -\log\sqrt{2\pi\lambda}$$. The final expression for the quadratic expansion of $$f(x)$$ around $$x = \mu$$
is

$$
f(x) \approx - \log\sqrt{2\pi\lambda} - \frac{(x - \mu)^2}{2\lambda} \qquad\ldots\,\text{for large } \lambda
$$

and the PDF for large $$\lambda$$ is approximately

$$
g(x) \approx \frac{e^{-(x-\mu)^2/(2\lambda)}} {\sqrt{2\pi\lambda}} = \mathcal{N}(\lambda, \lambda) \qquad \blacksquare
$$

## Binomial $$\to$$ normal

The PDF of the [binomial distribution] is

$$
g(x; n, p) = \binom{n}{x}p^x (1-p)^{n - x}
$$

Let us go through our prescribed steps.

$$
\begin{align}
\begin{split}
f(x) &= \log g(x) \\
&= \log n! - \log x! - \log(n-x)! + x\log p + (n-x)\log (1-p) \\
&= \log \frac{1}{\sqrt{2\pi n x (n-x)}} + n\log n - x\log x -(n-x)\log (n-x) \\
&\quad + x\log p + (n-x)\log (1-p)
\end{split}
\end{align}
$$

Thats a long expression, but most terms will cancel out after we're done. Lets keep going. Next step is to evaluate
$$f'(x)$$ and $$f''(x)$$:

$$
\begin{align}
f'(x) &= -\frac{n}{2x(n - x)} - \log \frac{x}{n-x} + \log \frac{p}{1-p} \\
f''(x) &= \frac{1}{2x^2} -\frac{1}{2(n-x)^2} - \frac{1}{x(n-x)}
\end{align}
$$

Next we will solve $$f'(x)= 0$$. For an asymptotic solution, we can neglect the $$\frac{n}{2x(n-x)}$$ term in the
solution for $$f'(x) = 0$$. In the appendix we provide an exact numerical solution including this term and prove
that the difference is negligible. Presently, the approximate solution of $$f'(x) = 0$$ yields, $$\mu = np$$.

Evaluating $$f(x)$$ and leading term of $$f''(x)$$ at $$x = \mu = np$$ yields

$$
\begin{align}
f(\mu) &= \log\frac{1}{\sqrt{2\pi np(1-p)}}\\
f''(\mu) &= -\frac{1}{np(1-p)}
\end{align}
$$

We're now done with calculations and can write:

$$
f(x) \approx \log\frac{1}{\sqrt{2\pi np(1-p)}} - \frac{(x-np)^2}{2np(1-p)}
$$

Finally the PDF approximated for $$np \gg 1$$ is

$$
g(x) \approx \frac{ e^{-\frac{(x-np)^2}{2np(1-p)}} } {\sqrt{2\pi np (1-p)}} = \mathcal{N}(np, np(1-p))
\qquad \blacksquare
$$
