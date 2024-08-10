---
layout: post
title: "Solving definite integrals with Plancherel's theorem"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## The Fourier transform

The
[Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform)
tells us that any signal can be represented in time or frequency. Imagine we have a pure tone
of 1000 Hz sounding with a strength of 1 decibel. We can represent this tone by tabulating its values at
_all_ times (in an infinitely large table). This is the **time representation**, also known as the time-domain signal
and written as $$x(t)$$.

Alternatively, we can represent this signal by specifying the strength of all frequencies that make up the
signal. In this case we have just 1 frequency at 1 decibel. So the **frequency representation** would be simply "1
decibel at 1000 Hz". The frequency representation of the signal, also known as the frequency-domain signal
is written as $$X(\omega)$$.

Both representations are provide the same information about the signal. In other words, an alien can construct the
signal from either from a table of $$x(t)$$ or a specification of $$X(\omega)$$. We write this equivalence as:

$$
x(t) \leftrightarrow X(\omega)
$$

$$X(\omega)$$ can be computed given $$x(t)$$ and _vice versa_

$$
\begin{align}
X(\omega) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}x(t)\, e^{-i\omega t}\, dt \\[0.1in]
x(t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}X(\omega)\, e^{i\omega t}\, d\omega \\
\end{align}
$$


## The Plancherel's theorem

Now lets talk about the total energy of the signal. In time representation, we can calculate the total energy of the
signal by adding up (or integrating) its strength at each point in time. In the frequency representation, we would add
up (integrate) the strength of each frequency component. Since the representations are identical, we should expect that
the answers obtaied from the time and the frequency representations should match. Plancherel's theorem is the
statement that they do.

In the above notation, the Plancherel's theorem states that

$$
\begin{equation}
\int_{-\infty}^{\infty} \left\vert x(t) \right\vert^2 dt =
\int_{-\infty}^{\infty} \left\vert X(\omega) \right\vert^2 d\omega
\end{equation}
$$

## Plancherel's theorem for evaluating definite integrals

Main application of the Fourier Transform is in signal processing. In fact, it is not an overstatement to say that
the Fourier Transform and its computational algorithm, the Fast Fourier Transform (FFT) underlie all modern
communication systems. There are plenty of books on this subject.

The present post, however, has a different purpose. We will twist the  mathematics of the Fourier transform to
evaluate some definite integrals. What we primarily seek from this exercise is entertainment. I'd imagine
that using Fourier Transforms to compute well-known definite integrals is of little practical use. With that
disclaimer, let us evaluate a few integrals!

The general recipe is the following:

1. Choose a function $$x(t)$$ and compute its Fourier Transform

2. Apply the Plancherel's theorem

3. Evaluate either the left- or the right hand side explicitly

4. Write down the value of the integral on the other side

## The sinc squared function

We start from the basic
[top-hat](https://en.wikipedia.org/wiki/Top-hat_filter#/media/File:Rectangular_function.svg)
function.

$$
\begin{align*}
x(t) &=
\begin{cases}
1 & -a \leq t \leq a\\[0.15in]
0 & \text{otherwise}
\end{cases} \\[0.2in]
X(\omega) &= \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty}x(t)\, e^{-i\omega t} dt =
\int_{-a}^{a}e^{-i\omega t} dt\\[0.1in]
&=\sqrt{\frac{2}{\pi}}\,\frac{\sin a\omega}{\omega}
\end{align*}
$$

Applying the Plancherel's theorem:

$$
\begin{align*}
\int_{-\infty}^{\infty} \left\vert x(t) \right\vert^2 dt &=
\int_{-\infty}^{\infty} \left\vert X(\omega) \right\vert^2 d\omega \\[0.1in]
\int_{-a}^{a} dt &=
\frac{2}{\pi}\int_{-\infty}^{\infty} \frac{\sin^2 a\omega}{\omega^2} d\omega \\[0.1in]
2a &\stackrel{u\leftarrow a\omega}{=} \frac{2a}{\pi}\int_{-\infty}^{\infty} \frac{\sin^2 u}{u^2} du \\[0.1in]
\end{align*}
$$

giving us

$$
\int_{-\infty}^{\infty}\left(\frac{\sin u}{u}\right)^2\, du= \pi
$$

But we can go further.

### Bonus: Sinc squared = Sinc

We can prove that

$$
\int_{-\infty}^{\infty}\left(\frac{\sin u}{u}\right)^2\, du = \int_{-\infty}^{\infty}\frac{\sin u}{u}\, du
$$

To do that integrate the left-hand side by parts, applying the limits:

$$
\begin{align*}
\int_{-\infty}^{\infty}\left(\frac{\sin u}{u}\right)^2\, du
&= -\frac{\sin u}{u}\bigg\vert_{-\infty}^{\infty} + \int_{-\infty}^{\infty}\frac{1}{u} 2 \sin u\cos u\, du \\
&= \int_{-\infty}^{\infty}\frac{\sin 2u}{u} \, du
\stackrel{v\leftarrow 2u}{=} \int_{-\infty}^{\infty}\frac{\sin v}{v} \, dv
\end{align*}
$$

Thus we have a neat identity

<div class="boxed">
$$
\int_{-\infty}^{\infty}\left(\frac{\sin u}{u}\right)^2\, du=
\int_{-\infty}^{\infty}\frac{\sin u}{u}\, du = \pi
$$
</div>

## The Gaussian function

$$
\begin{align*}
x(t) &= e^{-t^2} \\
X(\omega) &= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty e^{-t^2/2}\,e^{-i\omega t}dt \\
&= \frac{e^{-\omega^2/2}}{\sqrt{2\pi}}\int_{-\infty}^\infty e^{-(t + i\omega)^2/2}dt \\
&= \frac{e^{-\omega^2/2}}{\sqrt{\pi}}\int_{-\infty}^\infty e^{-u^2}du
\qquad\ldots\text{ put } t + i\omega = \sqrt{2} u\\
&= \frac{e^{-\omega^2/2}}{\sqrt{\pi}} \mathcal{J}
\end{align*}
$$

where

$$
\mathcal{J} = \int_{-\infty}^{\infty}e^{-t^2}dt
$$

Applying the Plancherel's theorem:

$$
\begin{align*}
\int_{-\infty}^\infty \vert x(t)\vert^2dt &=
\int_{-\infty}^\infty \vert X(\omega)\vert^2d\omega\\
\int_{-\infty}^\infty \left\vert e^{-t^2/2}\right\vert^2dt &=
\int_{-\infty}^\infty \left\vert \frac{e^{-\omega^2/2}}{\sqrt{\pi}}\mathcal{J}\right\vert^2d\omega\\
\int_{-\infty}^\infty e^{-t^2}dt &=
\frac{\mathcal{J}^2}{\pi}\int_{-\infty}^\infty e^{-\omega^2}d\omega\\
\mathcal{J} &= \frac{\mathcal{J}^3}{\pi}\\
\Rightarrow \mathcal{J} &= \sqrt{\pi}
\end{align*}
$$

Writing out explicitly, we have the well-known identity:

$$
\int_{-\infty}^{\infty}e^{-t^2}dt = \sqrt{\pi}
$$

Putting $$t^2 = \alpha y^2$$  we get a slightly more general identity $$\int_{-\infty}^{\infty}e^{-\alpha y^2}dy =
\sqrt{\pi/\alpha}$$. Substituting $$\alpha = i$$ (technically this step needs to be performed using contour
integration, but here the answer will be correct even without it):

$$
\begin{align*}
\int_{-\infty}^{\infty}e^{-it^2}\,dt &= \sqrt{\frac{\pi}{i}} \\
\int_{-\infty}^{\infty}\left(\cos t^2 -i\sin t^2\right)\,dt &= \sqrt{\frac{\pi}{2}} (1 + i) \\
\end{align*}
$$

Equating the real and the imaginary parts we get the limiting values of the
[Fresnel integrals](https://en.wikipedia.org/wiki/Fresnel_integral#Limits_as_x_approaches_infinity)

<div class="boxed">
$$
\int_{-\infty}^{\infty}\cos t^2 dt = \int_{-\infty}^{\infty}\sin t^2 dt  = \sqrt{\frac{\pi}{2}}
$$
</div>

## Summary and further directions

Plancherel's theorem can be used to evaluate a selected class of definite integrals. The integral must be a part of a
Fourier transform pair. For a successful application it should be possible to simplify or explicitly evaluate the
integral of the original function, or its Fourier Transform.

Fourier Transforms have a number of interesting properties that can prove useful for evaluating definite integrals.
For example, can we exploit the derivative identity for attacking other integrals or proving interesting identities
involving definite integrals?
