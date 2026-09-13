from __future__ import annotations

import operator
from typing import Any, Callable, Union

import numpy as np
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

Param = Union["Distribution", float, int, np.ndarray]

class Distribution:
  #tfd distribution instead of scipy density
  def __init__(self, parameters: list[Any]) -> None:
    self.parameters = parameters

  @property
  def parents(self) -> list[Distribution]:
    return [p for p in self.parameters if isinstance(p, Distribution)]

  def distribution(self, *parameters: Any) -> tfd.Distribution:
    raise NotImplementedError

  def __repr__(self) -> str:
    args = ", ".join(map(str, self.parameters))
    return f"{self.__class__.__name__}({args})"


class Continuous(Distribution):
  pass


class Discrete(Distribution):
  pass


class Normal(Continuous):
  def __init__(self, mu: Param, sigma: Param) -> None:
    self.parameters = [mu, sigma]

  def distribution(self, mu: Any, sigma: Any) -> tfd.Distribution:
    return tfd.Normal(loc=mu, scale=sigma)


class Gamma(Continuous):
  def __init__(self, shape: Param, rate: Param) -> None:
    self.parameters = [shape, rate]

  def distribution(self, shape: Any, rate: Any) -> tfd.Distribution:
    return tfd.Gamma(concentration=shape, rate=rate)


class Exponential(Continuous):
  def __init__(self, rate: Param) -> None:
    self.parameters = [rate]

  def distribution(self, rate: Any) -> tfd.Distribution:
    return tfd.Exponential(rate=rate)


class LogNormal(Continuous):
  def __init__(self, mu: Param, sigma: Param) -> None:
    self.parameters = [mu, sigma]

  def distribution(self, mu: Any, sigma: Any) -> tfd.Distribution:
    return tfd.LogNormal(loc=mu, scale=sigma)


class Logical(Distribution):
  def __init__(
    self, op: Callable[..., Any], operands: list[Any], symbol: str | None = None
  ) -> None:
    self.op = op
    self.symbol = symbol
    self.parameters = operands

  def __repr__(self) -> str:
    left, right = self.parameters
    return f"({left} {self.symbol} {right})"


def resolve(param: Any, values: dict[Distribution, Any]) -> Any:
  if isinstance(param, Logical):
    return param.op(*(resolve(p, values) for p in param.parameters))
  if isinstance(param, Distribution):
    return values[param]
  return param


def _wrap(op: Callable[..., Any], symbol: str):
  def fwd(self: Distribution, other: Any) -> Logical:
    return Logical(op, [self, other], symbol)

  def rev(self: Distribution, other: Any) -> Logical:
    return Logical(op, [other, self], symbol)

  return fwd, rev


Distribution.__add__, Distribution.__radd__ = _wrap(operator.add, "+")
Distribution.__sub__, Distribution.__rsub__ = _wrap(operator.sub, "-")
Distribution.__mul__, Distribution.__rmul__ = _wrap(operator.mul, "*")
Distribution.__truediv__, Distribution.__rtruediv__ = _wrap(operator.truediv, "/")
