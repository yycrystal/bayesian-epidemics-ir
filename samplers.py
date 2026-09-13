#Changes from GEMTests/samplers.py
#Replaced SingleSiteMH and the hand-written MetropolisHastings with gemlib's MwgStep
#Adaptive random walk from gemlib.mcmc.adaptive_rwmh()
#joint_pdf = log_density, returns a function instead of scalar/vector
#Kernel: one MwgStep per block of parameters, sampling algorithm chosen by node type, with tuning options and optional multi-site blocks
#Every proposal now rescores the whole graph, not just the moved node's dependants
#Same: dfs_nodes, resolve

from __future__ import annotations

import operator
from collections import namedtuple
from functools import reduce
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

import gemlib

from gemlib.mcmc.sampling_algorithm import SamplingAlgorithm

from distributions import (
  Discrete,
  Distribution,
  Exponential,
  Gamma,
  Logical,
  LogNormal,
  resolve,
)

tfb = tfp.bijectors

DTYPE = np.float64

Trace = dict[Distribution, Any] #Current values of the parameters 
Tuning = dict[str, dict[str, Any] | SamplingAlgorithm] #Per-parameter tuning
Groups = list[list[str]] #Parameters to propose together, rather than one by one

#DFS dictionary of nodes, ignoring Logical nodes
def dfs_nodes(
  node: Distribution, order: dict[Distribution, None] | None = None
) -> dict[Distribution, None]:
  if order is None:
    order = {}
  if node in order:
    return order
  for parent in node.parents:
    dfs_nodes(parent, order)
  if not isinstance(node, Logical):
    order[node] = None
  return order

#Log density of a single node, given the current trace
def node_log_prob(node: Distribution, values: Trace):
  parameters = [jnp.asarray(resolve(p, values), DTYPE) for p in node.parameters]
  return jnp.sum(node.distribution(*parameters).log_prob(values[node]))

#Joint log density of the DAG
def log_density(leaves: Distribution, parameters: dict[str, Distribution], observed: Trace):
  order = list(dfs_nodes(leaves))
  observed = {node: jnp.asarray(value, DTYPE) for node, value in observed.items()}

  missing = set(order) - set(parameters.values()) - set(observed)
  if missing:
    raise ValueError(f"no value for {missing}")

  def target_log_density_fn(**position: Any):
    values = {node: position[name] for name, node in parameters.items()} | observed
    return sum(node_log_prob(node, values) for node in order)

  return target_log_density_fn


#Choose a bijector for node
def bijector_for(node: Distribution):
  if isinstance(node, Gamma | Exponential | LogNormal): #bounded (0, inf)
    return tfb.Exp()
  return tfb.Identity() #Normal, else already on the real line


#Sampling algorithm suited to a block of one or more nodes
#tuning contains keyword arguments for whichever kernel the node types select
def sampler_for(names: list[str], nodes: list[Distribution], **tuning: Any):
  #No sampler for discrete nodes yet
  if any(isinstance(node, Discrete) for node in nodes):
    raise TypeError(f"{names}: no sampler for discrete nodes")

  #One bijector per member, in the order the block was given
  bijectors = [bijector_for(node) for node in nodes]
  walk = gemlib.mcmc.adaptive_rwmh(**tuning)
  if all(isinstance(b, tfb.Identity) for b in bijectors):
    return walk
  return gemlib.mcmc.transform_sampling_algorithm(bijectors, walk)


#Blocks of parameters proposed together
#A name in no group is its own block and the order of parameters decides the order of the sweep
def blocks(parameters: dict[str, Distribution], groups: Groups | None):
  grouped: dict[str, tuple[str, ...]] = {}
  for group in groups or []:
    for name in group:
      if name not in parameters:
        raise ValueError(f"group {group} names unknown parameter {name!r}")
      if name in grouped:
        raise ValueError(f"{name!r} is in more than one group")
      grouped[name] = tuple(group)

  ordered: list[tuple[str, ...]] = []
  for name in parameters:
    if (block := grouped.get(name, (name,))) not in ordered:
      ordered.append(block)
  return ordered


#One Metropolis-within-Gibbs step per block of parameters
def kernel(
  parameters: dict[str, Distribution],
  tuning: Tuning | None = None,
  groups: Groups | None = None,
):
  tuning = tuning or {}
  steps = []
  for names in blocks(parameters, groups):
    #Tuning is per parameter, group not supported yet
    options = tuning.get(names[0], {}) if len(names) == 1 else {}
    algorithm = (
      options
      if isinstance(options, SamplingAlgorithm)
      else sampler_for(list(names), [parameters[n] for n in names], **options)
    )
    steps.append(gemlib.mcmc.MwgStep(algorithm, list(names)))
  return reduce(operator.rshift, steps) #alpha_step >> beta_step >> sigma2_step


def run(
  leaves: Distribution,
  parameters: dict[str, Distribution],
  observed: Trace,
  initial: dict[str, float],
  num_samples: int = 10_000,
  tuning: Tuning | None = None,
  groups: Groups | None = None,
  seed=None,
):
  #Sample the DAG, compiles to a single jax jit-able function
  position = namedtuple("Position", list(parameters))(
    **{name: jnp.asarray(value, DTYPE) for name, value in initial.items()}
  ) #Position(alpha=0.0, beta=0.0, sigma2=1.0)
  sweep = kernel(parameters, tuning, groups)
  target = log_density(leaves, parameters, observed)

  def chain(seed):
    return gemlib.mcmc.mcmc(num_samples, sweep, target, position, seed)

  return jax.jit(chain)(jax.random.key(0) if seed is None else seed)
