---
layout: post
title: "Proving the special cases of the Central Limit Theorem"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction

The [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) (CLT)
states that the distribution of the average value of $$n$$ random variables drawn
from any distribution (with a finite mean $$\mu$$ and a variance $$\sigma^2$$) approaches a normal distribution. In
notation, if $$x_1, x_2, \ldots, x_n$$ are drawn from a distribution, and

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
more. We will first scketch the steps of the proof and then apply systematically to prove that Poisson, the Gamma
and the Binomial distributions converge to a normal distribution. As a consequence, we will also get the means and
the variances of the asymptotic normal distributions they each converge to.

## Steps to prove convergence to normal distribution

To prove that any function converges to a normal distribution is equivalent to proving that the logarithm of the
function converges to an inverted parabola (kind of like "$$\cap$$"). To prove that an arbitrary function is
an inverted parabola, we have to compute its Taylor expansion up to the quadratic term and show that the higher
order terms shrink fast. This process is the essence of the
[Laplace's method]({% post_url 2021-06-08-laplaces-method%}).

Assume that we are given a functional form of the PDF $g(x)$ with a goal to prove that it asymptotically converges to
a normal distribution. The steps to do that are as follows:

 1. Compute the logarithm of the given PDF: $$f(x) = \log g(x)$$
 2. Compute the first and the second derivatives of $$f(x)$$: $$f'(x)$$ and $$f''(x)$$
 3. Solve $$f'(x) = 0$$, call the solution $$x = \mu$$
 4. Evaluate the $$f(\mu)$$ and $$f''(\mu)$$
 5. Write the asymptotic equations

    5a. $$f(x) \to f(\mu) -\frac{\vert f''(\mu)\vert}{2}(x-\mu)^2$$

    5b. $$g(x) \to e^{f(\mu)}\,e^{-\frac{1}{2}\vert f''(\mu)\vert(x-\mu)^2}$$

Let us make a few general remarks before beginning the derivations.

 1. Normalized PDFs often have a factorial or a Gamma function in the denominator. We use the
 [Stirling's approximation]({%post_url 2021-06-01-stirlings-approximation%}) to get an asymptotic and differentiable
  form of the Gamma function. Explicitly, here are the two equivalent expressions of the Stirling's approximation.
  (The $$O(1/n)$$ signifies that the error goes down linearly as $$n$$ increases.)

  $$
  \begin{align}
  n! &= \Gamma(n + 1) \to \sqrt{2\pi n}\left(\frac{n}{e}\right)^n\, O\left(\frac{1}{n}\right) \\
  \log n! &= \log\Gamma(n + 1) \to n\log n - n + \frac{1}{2}\log 2\pi + \frac{1}{2} \log n + O\left(\frac{1}{n}\right)
  \end{align}
  $$

 2. As a result of the Stirling's approximation, in step 5 above, the factor $$e^{f(\mu)}$$ turns out to be
 $$\sqrt{\frac{\vert f''(\mu)\vert}{2\pi}}$$. This naturally identifies with the $$\frac{1}{\sqrt{2\pi\sigma^2}}$$
 factor in the eventual expression for the normal distribution.

 3. In all of the derivations below, we will retain only the _leading_ terms whenever we need to solve any equations.
 Concretely, this will amount to neglecting the derivative of $$\frac{1}{2}\log x$$ in the
 expression of Stirling's approximation above. Since we are dealing with asymptotic convergence (i.e. for large
 $$x$$) these approximations are justified. Consider the following numerical example. Let us solve
 $$a - \log x - \frac{1}{x} = 0$$, approximately (neglecting the $$\frac{1}{x}$$ term) and exactly. For
 $$a = 3$$, then the  approximate solution is $$x = e^{a} = 20.085536923187668$$ and the exact solution is
 $$x = 20.08553688518224$$. For larger values of $$a$$ the difference is below the floating point precision. Our
 derivations will be in this regime.

## Poisson distribution

The PDF of [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) is

$$
g(x) = \frac{\lambda^x e^{-\lambda}}{x!}
$$

Now lets carry out steps 1--5 above.

 1. Take the logarithm of the PDF:

    $$
    \begin{align}
    \begin{split}
    f(x) &= \log g(x) = x\log \lambda - \lambda - \log \Gamma(x + 1)\\[0.1in]
         &= x\log \lambda - \lambda - (x\log x - x + \log\sqrt{2\pi x} + O(1/n))\\
    \end{split}
    \end{align}
    $$

 2. Compute the first and the second derivatives:

    $$
    \begin{align}
    \begin{split}
    f'(x) &= \log\lambda - \log x - \frac{1}{2x} \label{eq:poissonf1} \\
    f''(x) &= -\frac{1}{x} + \frac{1}{2x^2} \label{eq:poissonf2}
    \end{split}
    \end{align}
    $$

 3. Solve for $$f'(x) = 0$$, neglecting $$1/(2x)$$:

    $$
    \begin{equation}
    f'(x) = 0 \Rightarrow x = \mu = \lambda
    \end{equation}
    $$

 4. Evaluate $$f(\mu)$$ and $$f''(\mu)$$:

    $$
    \begin{align}
    f(\mu) &= -\log\sqrt{2\pi\lambda} \\[0.2in]
    f''(\mu) &= -1/\lambda
    \end{align}
    $$

 5. Write Taylor expansion of $$f(x)$$ and $$g(x)$$ near $$x=\mu$$:

    $$
    \begin{align}
    f(x) &\to - \log\sqrt{2\pi\lambda} - \frac{(x - \mu)^2}{2\lambda} \\[0.2in]
    g(x) &\to \frac{e^{-\frac{(x - \mu)^2}{2\lambda}}}{\sqrt{2\pi\lambda}}
    \sim \mathcal{N}(\lambda, \lambda) \qquad \blacksquare
    \end{align}
    $$

## Gamma distribution

One of the forms for the PDF of the [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) is

$$
g(x) = \frac{\beta^{\alpha + 1} x^\alpha e^{-\beta x}}{\Gamma(\alpha + 1)}
$$

 1. Take the logarithm of the PDF:

    $$
    \begin{align}
    \begin{split}
    f(x) &= \log g(x) = (\alpha + 1) \log \beta + \alpha\log x -\beta x - \log\Gamma(\alpha + 1) \\[0.1in]
    &= (\alpha + 1) \log \beta + \alpha\log x -\beta x \\[0.05in]
    &\quad - \left(\alpha\log\alpha - \alpha + \log\sqrt{2\pi\alpha} + O(1/\alpha)\right)
    \end{split}
    \end{align}
    $$

 2. Compute the first and the second derivatives:

    $$
    \begin{align}
    f'(x) &= \frac{\alpha}{x} -\beta \\
    f''(x) &= -\frac{\alpha}{x^2}
    \end{align}
    $$

 3. Solve for $$f'(x) = 0$$:

    $$
    \begin{equation}
    f'(x) = 0 \Rightarrow x = \mu = \frac{\alpha}{\beta}
    \end{equation}
    $$

 4. Evaluate $$f(\mu)$$ and $$f''(\mu)$$:

    $$
    \begin{align}
    f(\mu) &= (\alpha + 1) \log \beta + \alpha\log \frac{\alpha}{\beta} -\beta \frac{\alpha}{\beta}
        - \left(\alpha\log\alpha - \alpha + \log\sqrt{2\pi\alpha}\right) \notag \\
        &= -\log\sqrt{2\pi(\alpha/\beta^2)} \\[0.2in]
    f''(\mu) &= -\frac{\beta^2}{\alpha}
    \end{align}
    $$

 5. Write Taylor expansion of $$f(x)$$ and $$g(x)$$ near $$x=\mu$$:

    $$
    \begin{align}
    f(x) &\to \log\frac{1}{\sqrt{2\pi(\alpha/\beta^2)}} - \frac{(x-\alpha/\beta)^2}{2(\alpha/\beta^2)} \\[0.2in]
    g(x) &\to \frac{e^{-\frac{(x-\alpha/\beta)^2}{2(\alpha/\beta^2)}}}{\sqrt{2\pi(\alpha/\beta^2)}}
    \sim \mathcal{N}(\frac{\alpha}{\beta}, \frac{\alpha}{\beta^2}) \qquad\blacksquare
    \end{align}
    $$

## Binomial distribution

The PDF of the [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution) is:

$$
g(x) = \binom{n}{x}p^x (1-p)^{n - x}
$$

Let us go through our prescribed steps.

 1. Take the logarithm of the PDF:

    $$
    \begin{align}
    \begin{split}
    f(x) &= \log g(x) \\[0.1in]
    &= \log n! - \log x! - \log(n-x)! + x\log p + (n-x)\log (1-p) \\[0.1in]
    &= -\log \sqrt{2\pi n x (n-x)} + n\log n - x\log x -(n-x)\log (n-x) \\
    &\quad + x\log p + (n-x)\log (1-p) + O(1/n)
    \end{split}
    \end{align}
    $$

    Thats a long expression, but most terms will cancel out after we're done. Lets keep going.

 2. Compute the first and the second derivatives:

    $$
    \begin{align}
    f'(x) &= -\frac{n}{2x(n - x)} - \log \frac{x}{n-x} + \log \frac{p}{1-p} \\
    f''(x) &= \frac{1}{2x^2} -\frac{1}{2(n-x)^2} - \frac{1}{x(n-x)}
    \end{align}
    $$

 3. Solve for $$f'(x) = 0$$, keeping only the leading terms:

    $$
    \begin{equation}
    f'(x) = 0 \qquad \Rightarrow x = \mu = np
    \end{equation}
    $$

 4. Evaluate $$f(\mu)$$ and $$f''(\mu)$$:

    $$
    \begin{align}
    f(\mu) &= \log\frac{1}{\sqrt{2\pi np(1-p)}}\\[0.2in]
    f''(\mu) &= -\frac{1}{np(1-p)}
    \end{align}
    $$

 5. Write Taylor expansion of $$f(x)$$ and $$g(x)$$ near $$x=\mu$$:

    $$
    \begin{align}
    f(x) &\to \log\frac{1}{\sqrt{2\pi np(1-p)}} - \frac{(x-np)^2}{2np(1-p)} \\[0.2in]
    g(x) &\to \frac{e^{-\frac{(x-np)^2}{2np(1-p)}}}{\sqrt{2\pi np (1-p)}}
    \sim \mathcal{N}(np, np(1-p)) \qquad \blacksquare
    \end{align}
    $$
