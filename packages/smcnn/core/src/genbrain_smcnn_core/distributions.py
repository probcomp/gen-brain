import jax
import jax.numpy as jnp
from genjax._src.core.pytree import Pytree
from genjax._src.generative_functions.distributions.distribution import exact_density
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


""" Distribution Utilities """


def labcat_generator_nonvec():
    @Pytree.partial()
    def sampler(key, probs, labels, **kwargs):
        cat = tfd.Categorical(probs=probs)
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    @Pytree.partial()
    def logpdf(v, probs, labels, **kwargs):
        w = jnp.log(jnp.sum(probs * (labels == v)))
        return w

    return exact_density(sampler, logpdf, "labeled_categorical_nonvec")


# This new implementation allows for arbitrarily structured labels, meaning
# you can use single values or N-dim vectors
def labcat_generator():
    @Pytree.partial()
    def sampler(key, probs, labels, **kwargs):
        cat = tfd.Categorical(probs=probs)
        cat_index = cat.sample(seed=key)
        return jnp.asarray(labels[cat_index])

    @Pytree.partial()
    def logpdf(v, probs, labels, **kwargs):
        vecval = jnp.atleast_2d(v)
        veclabels = jax.vmap(lambda x: jnp.atleast_2d(x))(labels)
        w = jnp.log(jnp.sum(probs * jnp.all(vecval == veclabels, axis=2).flatten()))
        return w

    return exact_density(sampler, logpdf, "labeled_categorical")


labeled_categorical = labcat_generator()


def unicat(x):
    return jnp.ones(len(x)) / len(x)


def normalize(x):
    return x / jnp.sum(x)


# probably a good idea to eventually allow in non-equi distant domain values.
def discrete_norm(μ, σ, dom):
    div = (dom[1] - dom[0]) / 2.0
    unnormed_dist = jnp.nan_to_num(
        normalize(
            jax.vmap(
                lambda i: tfd.Normal(loc=μ, scale=σ).cdf(i + div)
                - tfd.Normal(loc=μ, scale=σ).cdf(i - div)
            )(dom)
        )
    )
    # this will either sum to 1 or 0.
    dist = (
        ((μ < dom[0]) * (1 - jnp.sum(unnormed_dist)))
        * jnp.hstack((jnp.array([1]), jnp.zeros(len(dom) - 1)))
        + ((μ > dom[-1]) * (1 - jnp.sum(unnormed_dist)))
        * jnp.hstack((jnp.zeros(len(dom) - 1), jnp.array([1])))
        + unnormed_dist
    )
    return dist


def disc_gauss_unnorm(μ, σ, dom):
    div = (dom[1] - dom[0]) / 2.0
    unnormed_dist = jnp.nan_to_num(
        jax.vmap(
            lambda i: tfd.Normal(loc=μ, scale=σ).cdf(i + div)
            - tfd.Normal(loc=μ, scale=σ).cdf(i - div)
        )(dom)
    )
    # this will either sum to 1 or 0.
    return unnormed_dist


def truncate(μ, dom, winsize, normvals):
    return (jnp.abs(dom - μ) <= winsize) * normvals


def discrete_truncnorm(μ, σ, winsize, dom):
    return normalize(truncate(μ, dom, winsize, discrete_norm(μ, σ, dom)))


def onehot(x, dom):
    return discrete_truncnorm(x, 1, 0, dom)


# enter a negative winsize for going the other direction.
def upweight_zone(winstart, dom, winsize, density_in_win):
    weight_in_win = density_in_win / jnp.abs(winsize)
    weight_outside_win = (1 - density_in_win) / (len(dom) - jnp.abs(winsize))
    upweighted_window = (((dom - winstart) * jnp.sign(winsize)) >= 0) & (
        ((dom - winstart) * jnp.sign(winsize)) < jnp.abs(winsize)
    )
    return normalize(
        weight_in_win * upweighted_window + weight_outside_win * ~upweighted_window
    )
