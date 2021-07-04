---
layout: post
title: "Approximating the Gaussian with elementary functions"
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

No sooner had I asked myself this question, I answered it in affirmative. Basically the wave functions of bound
states in Quantum Mechanics are ready examples of orthonormal basis functions integrable between $$\pm\infty$$.
The eigenfunctions of the one-dimensional
[quantum harmonic oscillator](https://bit.ly/2UYezKA)
are one explicit example. Unfortunately, these functions do not have elementary anti-derivatives and are not useful
for simplifying the Gaussian integral. An interesting follow-up question then is:

>can we construct potential wells which, when plugged into Schrodinger's equation, yield orthonormal solution sets
whose eigenfunctions have elementary anti-derivatives? More generally, can we construct potential wells given an
orthonormal basis set of functions?

Unfortunately, answering this extremely interesting question is beyond my current mathematical ability.

Having exhausted the obvious solutions, we turn back to the problem of approximating the Gaussian.

<figure>
    <img src="{{site.url}}/assets/img/gaussian_approx_m20.gif" alt='map' style='margin: 10px;' height="300"/>
<!--     <figcaption>Figure 1. Integrand of the Gamma function and its comparison with a scaled Gaussian.</figcaption> -->
</figure>

<figure>
    <img src="{{site.url}}/assets/img/gaussian_approx_m20_combined.gif" alt='map' style='margin: 10px;' height="300"/>
</figure>

<figure>
    <img src="{{site.url}}/assets/img/gaussian_approx_varm.gif" alt='map' style='margin: 10px;' height="300"/>
</figure>
