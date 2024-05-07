import abc
from functools import partial
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from jax import Array, lax, nn, random, tree_map, vmap

MNIST_SIZE = 784
SEED = 0


class Model(eqx.Module):
    start: eqx.nn.Linear
    middle: list[eqx.nn.Linear]
    end: eqx.nn.Linear
    activation: Callable[[Array], Array] = nn.selu

    def __init__(self, input_size: int, latent_size: int, output_size: int, n_layers: int, key: Array):
        start_key, *middle_keys, end_key = random.split(key, n_layers + 2)
        self.start = eqx.nn.Linear(input_size, latent_size, key=start_key)
        self.middle = [eqx.nn.Linear(latent_size, latent_size, key=middle_key) for middle_key in middle_keys]
        self.end = eqx.nn.Linear(latent_size, output_size, key=end_key)

    def select_layer(self, t: float, x: Array) -> Array:
        return lax.switch(jnp.floor(t * len(self.middle)).astype(int), self.middle, x)

    @partial(eqx.filter_vmap, in_axes=(None, 0, None))
    def __call__(self, x: Array, t: float):
        return nn.selu(self.select_layer(t, x))


class Solver(eqx.Module, abc.ABC):
    model: Model
    dt: float = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)

    @abc.abstractmethod
    def step(self, x: Array, t: float, key: Array) -> tuple[Array, float, Array]:
        pass

    def __call__(self, x0: Array, key: Array):
        x = vmap(self.model.start)(x0)

        def scan_step(carry: tuple[Array, float, Array], _) -> tuple[tuple[Array, float, Array], None]:
            return self.step(*carry), _

        (x, *_), _ = lax.scan(scan_step, (x, self.t0, key), jnp.arange(self.t0, self.t1, self.dt))

        return vmap(self.model.end)(x)


class EulerSolver(Solver):
    def step(self, x: Array, t: float, key: Array) -> tuple[Array, float, Array]:
        return x + self.dt * self.model(x, t), t + self.dt, key


def mse_loss(y_pred: Array, y_true: Array) -> Array:
    return jnp.mean(jnp.square(y_pred - y_true))


def sample_data(x: Array, y: Array, batch_size: int, key: Array) -> tuple[Array, Array]:
    random_indices = random.choice(key, len(x), shape=(batch_size,), replace=False)
    return x[random_indices], y[random_indices]


def step(
    solver: Solver, x: Array, y: Array, optim: optax.GradientTransformation, opt_state: optax.OptState, key: Array
) -> tuple[Solver, optax.OptState, Array]:
    def loss_fn(solver: Solver, x: Array, y: Array) -> Array:
        return mse_loss(solver(x, key), y)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(solver, x, y)
    updates, new_opt_state = optim.update(grads, opt_state)
    new_solver = eqx.apply_updates(solver, updates)
    return new_solver, new_opt_state, loss


def train(
    init_solver: Solver,
    x: Array,
    y: Array,
    optim: optax.GradientTransformation,
    batch_size: int,
    num_steps: int,
    key: Array,
) -> tuple[Solver, Array]:
    init_params, static = eqx.partition(init_solver, eqx.is_array)

    def scan_fn(carry: tuple[Solver, optax.OptState], key: Array) -> tuple[tuple[Solver, optax.OptState], Array]:
        params, opt_state = carry
        solver = eqx.combine(params, static)

        x_batch, y_batch = sample_data(x, y, batch_size, key)
        new_solver, new_opt_state, loss = step(solver, x_batch, y_batch, optim, opt_state, key)

        return (eqx.filter(new_solver, eqx.is_array), new_opt_state), loss

    (params, _), loss = lax.scan(scan_fn, (init_params, optim.init(init_params)), random.split(key, num_steps))
    return eqx.combine(params, static), loss


def run_experiment():
    key = random.key(SEED)

    mnist = tfds.load("mnist", split="train", as_supervised=True).batch(60000)  # type: ignore
    mnist = next(iter(mnist))
    x, y = tree_map(jnp.array, mnist)
    x = jnp.reshape(x / 255, (60000, MNIST_SIZE))

    model = Model(MNIST_SIZE, 64, 1, 10, key)
    solver = EulerSolver(model=model, dt=0.01, t0=0.0, t1=1.0)

    optim = optax.adam(1e-4)

    solver, losses = train(solver, x, y, optim, 100, 1000, key)
    print(losses)


run_experiment()
