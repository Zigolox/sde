import abc
from functools import partial
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from jax import Array, lax, nn, random, vmap

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
        x_latent = vmap(self.model.start)(x0)

        def scan_step(carry: tuple[Array, float, Array], _) -> tuple[tuple[Array, float, Array], None]:
            return self.step(*carry), _

        (x, *_), _ = lax.scan(scan_step, (x_latent, self.t0, key), jnp.arange(self.t0, self.t1, self.dt))

        return vmap(self.model.end)(x)


class EulerSolver(Solver):
    def step(self, x: Array, t: float, key: Array) -> tuple[Array, float, Array]:
        return x + self.dt * self.model(x, t), t + self.dt, key


class EulerMaruyamaSolver(Solver):
    std: float = eqx.field(static=True)

    def step(self, x: Array, t: float, key: Array) -> tuple[Array, float, Array]:
        return x + self.dt * self.model(x, t) + self.std * random.normal(key, x.shape), t + self.dt, key


def mse_loss(y_pred: Array, y_true: Array) -> Array:
    return jnp.mean(jnp.square(y_pred - y_true))


def cross_entropy_loss(y_pred: Array, y_true: Array) -> Array:
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true))


def sample_data(x: Array, y: Array, batch_size: int, key: Array) -> tuple[Array, Array]:
    random_indices = random.choice(key, len(x), shape=(batch_size,), replace=False)
    return x[random_indices], y[random_indices]


def train_step(
    solver: Solver,
    x: Array,
    y: Array,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: Callable[[Array, Array], Array],
    key: Array,
) -> tuple[Solver, optax.OptState, Array]:
    def loss_wrapper(solver: Solver, x: Array, y: Array) -> Array:
        return loss_fn(solver(x, key), y)

    loss, grads = eqx.filter_value_and_grad(loss_wrapper)(solver, x, y)
    updates, new_opt_state = optim.update(grads, opt_state)
    new_solver = eqx.apply_updates(solver, updates)
    return new_solver, new_opt_state, loss


@eqx.filter_jit
def train(
    init_solver: Solver,
    x: Array,
    y: Array,
    optim: optax.GradientTransformation,
    loss_fn: Callable[[Array, Array], Array],
    batch_size: int,
    num_steps: int,
    key: Array,
) -> tuple[Solver, Array]:
    init_params, static = eqx.partition(init_solver, eqx.is_array_like)

    def scan_fn(carry: tuple[Solver, optax.OptState], key: Array) -> tuple[tuple[Solver, optax.OptState], Array]:
        params, opt_state = carry
        solver = eqx.combine(params, static)

        x_batch, y_batch = sample_data(x, y, batch_size, key)
        new_solver, new_opt_state, loss = train_step(solver, x_batch, y_batch, optim, opt_state, loss_fn, key)

        return (eqx.filter(new_solver, eqx.is_array_like), new_opt_state), loss

    (params, _), loss = lax.scan(scan_fn, (init_params, optim.init(init_params)), random.split(key, num_steps))
    return eqx.combine(params, static), loss


def load_data(split_name: str) -> tuple[Array, Array]:
    mnist = tfds.load("mnist", split=split_name, as_supervised=True)
    mnist_complete = next(iter(mnist.batch(len(mnist))))  # type: ignore
    x, y = mnist_complete
    # Flatten and normailize the image
    x = jnp.reshape(jnp.asarray(x) / 255, (len(mnist), MNIST_SIZE))
    return jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.int32)


@eqx.filter_jit
def test_solver(solver: Solver, x: Array, y: Array, key: Array, loss_fn: Callable[[Array, Array], Array]) -> Array:
    y_pred = solver(x, key)
    # round each value to the closest integer to allow continuous predictions
    if loss_fn == cross_entropy_loss:
        y_pred = jnp.argmax(y_pred, axis=-1)
    y_pred = jnp.round(y_pred)
    return jnp.mean(y_pred == y)


def run_experiment():
    key = random.key(SEED)

    x, y = load_data("train")
    x_test, y_test = load_data("test")

    model = Model(MNIST_SIZE, latent_size=128, output_size=10, n_layers=10, key=key)
    solver = EulerMaruyamaSolver(model=model, dt=0.01, t0=0.0, t1=1, std=0)
    loss_fn = cross_entropy_loss
    optim = optax.adam(1e-4)

    accuracy = test_solver(solver, x_test, y_test, key, loss_fn)
    print("Initial accuracy:", accuracy)

    fitted_solver, losses = train(solver, x, y, optim, loss_fn, 64, 3000, key)

    accuracy = test_solver(fitted_solver, x_test, y_test, key, loss_fn)
    print(f"Final loss: {losses[-1]}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    run_experiment()
