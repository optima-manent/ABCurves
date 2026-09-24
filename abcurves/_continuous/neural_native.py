"""Fixed-shape execution of the unchanged float32 motor weights.

All 70 history tokens are evaluated. The input-to-hidden GRU projection is
batched before the sequential recurrence; the recurrence uses preallocated
buffers. The public native backend selects the qualified Padé9 motor and exact
small networks. Other variants are retained for reproducing the experiments;
isolated coefficient checks alone do not qualify a complete policy backend.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(inline="always", error_model="numpy")
def _tanh(value, activation):
    if activation == 0:
        return np.tanh(value)
    # Bounded Pade approximants. Coefficients below describe activation
    # functions, not learned parameters. Inputs are clamped before squaring.
    bound = np.float32(5.0 if activation == 1 else 4.0 if activation == 2 else 3.0)
    x = min(max(value, -bound), bound)
    x2 = x * x
    if activation == 1:
        numerator = x * (np.float32(135135.0) + x2 * (
            np.float32(17325.0) + x2 * (np.float32(378.0) + x2)))
        denominator = np.float32(135135.0) + x2 * (
            np.float32(62370.0) + x2 * (np.float32(3150.0) + np.float32(28.0) * x2))
    elif activation == 2:
        numerator = x * (np.float32(945.0) + x2 * (np.float32(105.0) + x2))
        denominator = np.float32(945.0) + x2 * (np.float32(420.0) + np.float32(15.0) * x2)
    else:
        numerator = x * (np.float32(27.0) + x2)
        denominator = np.float32(27.0) + np.float32(9.0) * x2
    return min(max(numerator / denominator, np.float32(-1.0)), np.float32(1.0))


@njit(inline="always", error_model="numpy")
def _sigmoid(value, activation):
    if activation == 0:
        # Keep the exponential finite even on very negative selector inputs.
        # Fast reciprocal code must never receive an intermediate infinity.
        z = np.exp(-abs(value))
        denominator = np.float32(1.0) + z
        return np.float32(1.0) / denominator if value >= np.float32(0.0) else z / denominator
    return np.float32(0.5) + np.float32(0.5) * _tanh(np.float32(0.5) * value, activation)


def _make_kernel(activation, recurrent_method, fastmath):
    @njit(cache=True, fastmath=fastmath, error_model="numpy")
    def run(coarse, fine, dynamics, weights, workspace):
        input_weight, input_bias, input_gates, input_gate_bias, recurrent_weight, recurrent_transpose, recurrent_bias, trunk_weight, trunk_bias, output_weight, output_bias = weights
        tokens, projected, hidden, recurrent, combined, encoded, coefficients = workspace
        np.dot(coarse, input_weight, tokens)
        for row in range(70):
            for column in range(96):
                value = tokens[row, column] + input_bias[column]
                tokens[row, column] = value * _sigmoid(value, 0 if activation == 4 else activation)
        np.dot(tokens, input_gates, projected)
        for row in range(70):
            for column in range(288):
                projected[row, column] += input_gate_bias[column]
        # Capture fastmath in the function closure too: Numba's disk-cache key
        # must distinguish factory instances with different compiler flags.
        if fastmath:
            hidden.fill(np.float32(0.0))
        else:
            hidden[:] = np.float32(0.0)
        for row in range(70):
            if recurrent_method == 0:
                np.dot(recurrent_weight, hidden, recurrent)
                for column in range(288):
                    recurrent[column] += recurrent_bias[column]
            else:
                # Contiguous output columns allow the compiler to vectorize
                # independent accumulators; no serial reduction is required.
                recurrent[:] = recurrent_bias
                for incoming in range(96):
                    value = hidden[incoming]
                    for column in range(288):
                        recurrent[column] += value * recurrent_transpose[incoming, column]
            for column in range(96):
                reset = _sigmoid(projected[row, column] + recurrent[column], activation)
                update = _sigmoid(projected[row, 96 + column] + recurrent[96 + column], activation)
                candidate = _tanh(projected[row, 192 + column] + reset * recurrent[192 + column], activation)
                hidden[column] = candidate + (hidden[column] - candidate) * update
        combined[:96] = hidden
        combined[96:150] = fine
        combined[150:] = dynamics
        np.dot(trunk_weight, combined, encoded)
        for column in range(96):
            value = encoded[column] + trunk_bias[column]
            encoded[column] = value * _sigmoid(value, 0 if activation == 4 else activation)
        np.dot(output_weight, encoded, coefficients)
        for column in range(672):
            coefficients[column] += output_bias[column]

    return run


VARIANTS = {
    "exact_blas": (0, 0, False),
    "exact_fastmath_blas": (0, 0, True),
    "pade7_blas": (1, 0, True),
    "pade5_blas": (2, 0, True),
    "pade3_blas": (3, 0, True),
    "exact_fastmath_vector": (0, 1, True),
    "pade7_vector": (1, 1, True),
    "pade3_vector": (3, 1, True),
    "pade3_gru_blas": (4, 0, True),
    "pade3_gru_vector": (4, 1, True),
}
_KERNELS = {}


class NativeMotor:
    """Single-stream experimental motor with borrowed output buffers.

    ``motor`` accepts the same fixed ``(1,70,9)``, ``(1,54)``, ``(1,8)`` float32
    inputs as the exported ONNX model, and returns encoded ``(1,96)`` and
    coefficient ``(16,21,2)`` arrays. Returned views are overwritten on the next
    call. Construct a separate instance for every concurrent stream.
    """

    def __init__(self, assets, variant="exact_blas"):
        if variant not in VARIANTS:
            raise ValueError("Unknown experimental neural variant: " + str(variant))
        arrays = assets.arrays if hasattr(assets, "arrays") else assets

        def weight(name, transpose=False):
            value = arrays["motor." + name]
            if value.dtype != np.float32:
                raise ValueError("Learned motor weights must remain float32")
            return np.ascontiguousarray(value.T if transpose else value)

        self.variant = variant
        self.weights = (
            weight("input.0.weight", True), weight("input.0.bias"),
            weight("encoder.weight_ih_l0", True), weight("encoder.bias_ih_l0"),
            weight("encoder.weight_hh_l0"), weight("encoder.weight_hh_l0", True), weight("encoder.bias_hh_l0"),
            weight("trunk.0.weight"), weight("trunk.0.bias"),
            weight("output.weight"), weight("output.bias"),
        )
        self.workspace = (
            np.empty((70, 96), np.float32), np.empty((70, 288), np.float32),
            np.empty(96, np.float32), np.empty(288, np.float32),
            np.empty(158, np.float32), np.empty(96, np.float32), np.empty(672, np.float32),
        )
        if variant not in _KERNELS:
            _KERNELS[variant] = _make_kernel(*VARIANTS[variant])
        self.run = _KERNELS[variant]
        self.encoded = self.workspace[-2].reshape(1, 96)
        self.coefficients = self.workspace[-1].reshape(16, 21, 2)

    def motor(self, coarse, fine, dynamics):
        self.run(coarse[0], fine[0], dynamics[0], self.weights, self.workspace)
        return self.encoded, self.coefficients


def _make_heads_kernels(activation):
    @njit(cache=True, fastmath=True, error_model="numpy")
    def choice(encoded, geometry, pairs, previous_valid, weights, workspace):
        unary_latent, unary_geometry, unary_bias, unary_output, unary_output_bias, context_weight, context_bias, pair_context, pair_geometry, pair_bias, pair_output, pair_output_bias = weights
        shared, unary, context, shared_pair, pair, logits, transition = workspace
        np.dot(unary_latent, encoded, shared)
        for column in range(64):
            shared[column] += unary_bias[column]
        np.dot(geometry, unary_geometry, unary)
        for row in range(16):
            for column in range(64):
                value = shared[column] + unary[row, column]
                unary[row, column] = value * _sigmoid(value, activation)
        np.dot(unary, unary_output, logits)
        for row in range(16):
            logits[row] += unary_output_bias[0]
        if previous_valid:
            np.dot(context_weight, encoded, context)
            for column in range(16):
                value = context[column] + context_bias[column]
                context[column] = value * _sigmoid(value, activation)
            np.dot(pair_context, context, shared_pair)
            for column in range(32):
                shared_pair[column] += pair_bias[column]
            np.dot(pairs, pair_geometry, pair)
            for row in range(16):
                for column in range(32):
                    value = shared_pair[column] + pair[row, column]
                    pair[row, column] = value * _sigmoid(value, activation)
            np.dot(pair, pair_output, transition)
            for row in range(16):
                logits[row] += transition[row] + pair_output_bias[0]

    @njit(cache=True, fastmath=True, error_model="numpy")
    def hazard(context, weights, workspace):
        first_weight, first_bias, second_weight, second_bias, final_weight, final_bias = weights
        first, second, output = workspace
        np.dot(first_weight, context, first)
        for column in range(64):
            value = first[column] + first_bias[column]
            first[column] = value * _sigmoid(value, activation)
        np.dot(second_weight, first, second)
        for column in range(32):
            value = second[column] + second_bias[column]
            second[column] = value * _sigmoid(value, activation)
        np.dot(final_weight, second, output)
        for column in range(2):
            output[column] += final_bias[column]

    @njit(cache=True, fastmath=True, error_model="numpy")
    def brake(context, weights, workspace):
        first_weight, first_bias, second_weight, second_bias, brake_weight, brake_bias, frequency_weight, frequency_bias = weights
        first, encoded, parameters, duration, coefficients, frequency = workspace
        np.dot(first_weight, context, first)
        for column in range(96):
            value = first[column] + first_bias[column]
            first[column] = value * _sigmoid(value, activation)
        np.dot(second_weight, first, encoded)
        for column in range(96):
            value = encoded[column] + second_bias[column]
            encoded[column] = value * _sigmoid(value, activation)
        np.dot(brake_weight, encoded, parameters)
        for head in range(16):
            # The float32 clamp/exp is exactly the learned brake's output
            # transform. It is never replaced by an activation approximation.
            log_duration = parameters[11 * head] + brake_bias[11 * head]
            log_duration = min(max(log_duration, np.float32(1.3862943611198906)), np.float32(5.2574953720277815))
            duration[head] = np.exp(log_duration)
            for column in range(10):
                coefficients[10 * head + column] = np.float32(64.0) * (
                    parameters[11 * head + column + 1] + brake_bias[11 * head + column + 1])
        np.dot(frequency_weight, encoded, frequency)
        for head in range(16):
            frequency[head] += frequency_bias[head]

    return choice, hazard, brake


_HEAD_KERNELS = {}


class NativeHeads:
    """Experimental same-weight selector, hazards, and finite-brake networks.

    Public method shapes match ``OnnxBackend``. Learned tensors remain float32.
    ``activation='exact'`` uses float32 exponential SiLU; ``'pade7'`` uses the
    same bounded rational sigmoid as the experimental motor. Outputs are
    borrowed views, overwritten by the next call to the corresponding method.
    Neither this class nor ``NativeMotor`` changes process BLAS thread settings.
    """

    def __init__(self, assets, *, activation="exact"):
        if activation not in ("exact", "pade7"):
            raise ValueError("Unknown small-network activation: " + str(activation))
        arrays = assets.arrays if hasattr(assets, "arrays") else assets

        def weight(name, transpose=False):
            value = arrays[name]
            if value.dtype != np.float32:
                raise ValueError("Learned neural weights must remain float32")
            return np.ascontiguousarray(value.T if transpose else value)

        unary = weight("motor.selection.unary.0.weight")
        pair = weight("motor.selection.pair.0.weight")
        self.choice_weights = (
            np.ascontiguousarray(unary[:, :96]), np.ascontiguousarray(unary[:, 96:].T),
            weight("motor.selection.unary.0.bias"), weight("motor.selection.unary.2.weight").reshape(64),
            weight("motor.selection.unary.2.bias"), weight("motor.selection.context.0.weight"),
            weight("motor.selection.context.0.bias"), np.ascontiguousarray(pair[:, 20:]),
            np.ascontiguousarray(pair[:, :20].T), weight("motor.selection.pair.0.bias"),
            weight("motor.selection.pair.2.weight").reshape(32), weight("motor.selection.pair.2.bias"),
        )
        self.hazard_weights = tuple(weight("events.hazard." + name) for name in (
            "0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"))
        self.brake_weights = tuple(weight("events." + name) for name in (
            "encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias",
            "brake.weight", "brake.bias", "frequency.weight", "frequency.bias"))
        self.choice_workspace = tuple(np.empty(shape, np.float32) for shape in (
            64, (16, 64), 16, 32, (16, 32), 16, 16))
        self.hazard_workspace = tuple(np.empty(shape, np.float32) for shape in (64, 32, 2))
        self.brake_workspace = tuple(np.empty(shape, np.float32) for shape in (96, 96, 176, 16, 160, 16))
        self.duration = self.brake_workspace[-3]
        self.coefficients = self.brake_workspace[-2].reshape(16, 5, 2)
        self.frequency = self.brake_workspace[-1]
        self.hazard_logits = self.hazard_workspace[-1]
        self.choice_logits = self.choice_workspace[-2]
        self.activation = activation
        if activation not in _HEAD_KERNELS:
            _HEAD_KERNELS[activation] = _make_heads_kernels(0 if activation == "exact" else 1)
        self.run_choice, self.run_hazard, self.run_brake = _HEAD_KERNELS[activation]

    def choice(self, encoded, unary_geometry, pairs, previous_valid):
        self.run_choice(encoded[0], unary_geometry[0], pairs[0], bool(previous_valid), self.choice_weights, self.choice_workspace)
        return self.choice_logits

    def hazard(self, context):
        self.run_hazard(context[0], self.hazard_weights, self.hazard_workspace)
        return self.hazard_logits

    def brake(self, context):
        self.run_brake(context[0], self.brake_weights, self.brake_workspace)
        return self.duration, self.coefficients, self.frequency

    def events(self, context):
        duration, coefficients, frequency = self.brake(context)
        return duration, coefficients, frequency, self.hazard(context)


# Separate accuracy experiment: the already qualified Padé7 and exact-head
# execution above is unchanged. The odd continued fraction
# x / (1 + x² / (3 + x² / (... + x² / 17))) expands to this 9/8 rational.
@njit(inline="always", error_model="numpy")
def _tanh_pade9(value):
    x = min(max(value, np.float32(-7.0)), np.float32(7.0))
    z = x * x
    numerator = x * (np.float32(34459425.0) + z * (np.float32(4729725.0) + z * (
        np.float32(135135.0) + z * (np.float32(990.0) + z))))
    denominator = np.float32(34459425.0) + z * (np.float32(16216200.0) + z * (
        np.float32(945945.0) + z * (np.float32(13860.0) + np.float32(45.0) * z)))
    return min(max(numerator / denominator, np.float32(-1.0)), np.float32(1.0))


@njit(inline="always", error_model="numpy")
def _sigmoid_pade9(value):
    return np.float32(0.5) + np.float32(0.5) * _tanh_pade9(np.float32(0.5) * value)


@njit(cache=True, fastmath=True, error_model="numpy")
def _run_pade9(coarse, fine, dynamics, weights, workspace):
    """Same vector recurrence/layout as Padé7, with only activation replaced."""
    input_weight, input_bias, input_gates, input_gate_bias, recurrent_weight, recurrent_transpose, recurrent_bias, trunk_weight, trunk_bias, output_weight, output_bias = weights
    tokens, projected, hidden, recurrent, combined, encoded, coefficients = workspace
    np.dot(coarse, input_weight, tokens)
    for row in range(70):
        for column in range(96):
            value = tokens[row, column] + input_bias[column]
            tokens[row, column] = value * _sigmoid_pade9(value)
    np.dot(tokens, input_gates, projected)
    for row in range(70):
        for column in range(288):
            projected[row, column] += input_gate_bias[column]
    hidden.fill(np.float32(0.0))
    for row in range(70):
        recurrent[:] = recurrent_bias
        for incoming in range(96):
            value = hidden[incoming]
            for column in range(288):
                recurrent[column] += value * recurrent_transpose[incoming, column]
        for column in range(96):
            reset = _sigmoid_pade9(projected[row, column] + recurrent[column])
            update = _sigmoid_pade9(projected[row, 96 + column] + recurrent[96 + column])
            candidate = _tanh_pade9(projected[row, 192 + column] + reset * recurrent[192 + column])
            hidden[column] = candidate + (hidden[column] - candidate) * update
    combined[:96] = hidden
    combined[96:150] = fine
    combined[150:] = dynamics
    np.dot(trunk_weight, combined, encoded)
    for column in range(96):
        value = encoded[column] + trunk_bias[column]
        encoded[column] = value * _sigmoid_pade9(value)
    np.dot(output_weight, encoded, coefficients)
    for column in range(672):
        coefficients[column] += output_bias[column]


VARIANTS["pade9_vector"] = (5, 1, True)
_KERNELS["pade9_vector"] = _run_pade9
