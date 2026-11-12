//! Small allocation-free tensor operations used during inference.

const std = @import("std");

pub const TensorError = error{
    EmptyInput,
    LengthMismatch,
    DimensionOverflow,
    InvalidDimension,
};

pub fn dot(left: []const f32, right: []const f32) TensorError!f32 {
    if (left.len == 0) {
        return error.EmptyInput;
    }

    if (left.len != right.len) {
        return error.LengthMismatch;
    }

    var sum: f32 = 0;

    for (left, right) |left_value, right_value| {
        sum += left_value * right_value;
    }

    return sum;
}

pub fn argmax(values: []const f32) TensorError!usize {
    if (values.len == 0) {
        return error.EmptyInput;
    }

    var best_index: usize = 0;
    var best_value = values[0];

    for (values[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }

    return best_index;
}

pub fn softmax(values: []f32) TensorError!void {
    const max_index = try argmax(values);
    const max_value = values[max_index];

    var sum: f32 = 0;

    for (values) |*value| {
        value.* = @exp(value.* - max_value);
        sum += value.*;
    }

    for (values) |*value| {
        value.* /= sum;
    }
}

/// Computes output = matrix × vector.
///
/// `matrix` has shape [output.len][vector.len] in row-major order.
/// `output` must not overlap `matrix` or `vector`.
pub fn matVec(
    output: []f32,
    matrix: []const f32,
    vector: []const f32,
) TensorError!void {
    if (output.len == 0 or matrix.len == 0 or vector.len == 0) {
        return error.EmptyInput;
    }

    const expected_matrix_len = std.math.mul(
        usize,
        output.len,
        vector.len,
    ) catch return error.DimensionOverflow;

    if (matrix.len != expected_matrix_len) {
        return error.LengthMismatch;
    }

    for (output, 0..) |*output_value, row_index| {
        const row_start = row_index * vector.len;
        const row = matrix[row_start..][0..vector.len];

        output_value.* = try dot(row, vector);
    }
}

/// Normalizes `input` by its root mean square, then applies learned weights.
///
/// `output`, `input`, and `weight` must have equal lengths.
/// `output` may be the same slice as `input`.
pub fn rmsNorm(
    output: []f32,
    input: []const f32,
    weight: []const f32,
) TensorError!void {
    if (output.len == 0 or input.len == 0 or weight.len == 0) {
        return error.EmptyInput;
    }

    if (output.len != input.len or input.len != weight.len) {
        return error.LengthMismatch;
    }

    var sum_of_squares: f32 = 0;

    for (input) |value| {
        sum_of_squares += value * value;
    }

    const mean_square =
        sum_of_squares / @as(f32, @floatFromInt(input.len));
    const scale = 1.0 / @sqrt(mean_square + 1e-5);

    for (output, input, weight) |*result, value, learned_weight| {
        result.* = learned_weight * value * scale;
    }
}

/// Applies SwiGLU element by element.
///
/// On return, each `gate` value becomes:
/// silu(gate) * up
pub fn swiGlu(
    gate: []f32,
    up: []const f32,
) TensorError!void {
    if (gate.len == 0 or up.len == 0) {
        return error.EmptyInput;
    }

    if (gate.len != up.len) {
        return error.LengthMismatch;
    }

    for (gate, up) |*gate_value, up_value| {
        const value = gate_value.*;
        const silu = value / (1.0 + @exp(-value));

        gate_value.* = silu * up_value;
    }
}

/// Applies rotary position encoding in place.
///
/// `values` may contain multiple attention heads.
/// `cosines` and `sines` contain the frequencies for one token position.
pub fn applyRope(
    values: []f32,
    cosines: []const f32,
    sines: []const f32,
    head_size: usize,
) TensorError!void {
    if (values.len == 0 or cosines.len == 0 or sines.len == 0) {
        return error.EmptyInput;
    }

    if (head_size == 0 or head_size % 2 != 0) {
        return error.InvalidDimension;
    }

    if (values.len % head_size != 0) {
        return error.LengthMismatch;
    }

    const frequencies_per_head = head_size / 2;

    if (cosines.len != frequencies_per_head or
        sines.len != frequencies_per_head)
    {
        return error.LengthMismatch;
    }

    var index: usize = 0;

    while (index < values.len) : (index += 2) {
        const frequency_index = (index % head_size) / 2;
        const cosine = cosines[frequency_index];
        const sine = sines[frequency_index];

        const first = values[index];
        const second = values[index + 1];

        values[index] = first * cosine - second * sine;
        values[index + 1] = first * sine + second * cosine;
    }
}

test "dot multiplies corresponding values and sums the products" {
    const left = [_]f32{ 1, 2, 3 };
    const right = [_]f32{ 4, 5, 6 };

    try std.testing.expectEqual(@as(f32, 32), try dot(&left, &right));
}

test "dot supports negative values" {
    const left = [_]f32{ -1, 2, -3 };
    const right = [_]f32{ 4, -5, 6 };

    try std.testing.expectEqual(@as(f32, -32), try dot(&left, &right));
}

test "dot rejects empty input" {
    const empty = [_]f32{};

    try std.testing.expectError(error.EmptyInput, dot(&empty, &empty));
}

test "dot rejects different input lengths" {
    const left = [_]f32{ 1, 2 };
    const right = [_]f32{1};

    try std.testing.expectError(error.LengthMismatch, dot(&left, &right));
}

test "argmax returns the index of the largest value" {
    const values = [_]f32{ -4, 8, 3 };

    try std.testing.expectEqual(@as(usize, 1), try argmax(&values));
}

test "argmax handles all-negative values" {
    const values = [_]f32{ -9, -2, -7 };

    try std.testing.expectEqual(@as(usize, 1), try argmax(&values));
}

test "argmax keeps the first index when values tie" {
    const values = [_]f32{ 7, 7, 2 };

    try std.testing.expectEqual(@as(usize, 0), try argmax(&values));
}

test "argmax rejects empty input" {
    const empty = [_]f32{};

    try std.testing.expectError(error.EmptyInput, argmax(&empty));
}

test "softmax matches NumPy reference values and sums to one" {
    var values = [_]f32{ 1, 2, 3 };
    const expected = [_]f32{ 0.09003057, 0.24472848, 0.66524094 };

    try softmax(&values);

    var sum: f32 = 0;
    for (values, expected) |actual, expected_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.000001);
        sum += actual;
    }

    try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 0.000001);
}

test "softmax remains stable for large inputs" {
    var values = [_]f32{ 1000, 1001, 1002 };
    const expected = [_]f32{ 0.09003057, 0.24472848, 0.66524094 };

    try softmax(&values);

    for (values, expected) |actual, expected_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.000001);
    }
}

test "softmax rejects empty input" {
    var empty = [_]f32{};

    try std.testing.expectError(error.EmptyInput, softmax(&empty));
}

test "matVec matches NumPy for a row-major matrix" {
    const matrix = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const vector = [_]f32{ 0.5, -1, 2 };
    var output: [2]f32 = undefined;

    try matVec(&output, &matrix, &vector);

    try std.testing.expectEqualSlices(f32, &.{ 4.5, 9 }, &output);
}

test "matVec rejects a matrix with the wrong shape" {
    const matrix = [_]f32{ 1, 2, 3, 4, 5 };
    const vector = [_]f32{ 1, 2, 3 };
    var output: [2]f32 = undefined;

    try std.testing.expectError(
        error.LengthMismatch,
        matVec(&output, &matrix, &vector),
    );
}

test "matVec rejects empty inputs" {
    var empty_output = [_]f32{};
    const empty_values = [_]f32{};
    var output: [1]f32 = undefined;
    const one = [_]f32{1};

    try std.testing.expectError(
        error.EmptyInput,
        matVec(&empty_output, &one, &one),
    );
    try std.testing.expectError(
        error.EmptyInput,
        matVec(&output, &empty_values, &one),
    );
    try std.testing.expectError(
        error.EmptyInput,
        matVec(&output, &one, &empty_values),
    );
}

test "rmsNorm matches NumPy reference values" {
    const input = [_]f32{ 1, -2, 3 };
    const weight = [_]f32{ 0.5, 1.5, -2 };
    const expected = [_]f32{
        0.23145477,
        -1.3887286,
        -2.7774572,
    };
    var output: [3]f32 = undefined;

    try rmsNorm(&output, &input, &weight);

    for (output, expected) |actual, expected_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.000001);
    }
}

test "rmsNorm supports using the input as output" {
    const input = [_]f32{ 1, -2, 3 };
    const weight = [_]f32{ 0.5, 1.5, -2 };
    var expected: [3]f32 = undefined;
    var in_place = input;

    try rmsNorm(&expected, &input, &weight);
    try rmsNorm(&in_place, &in_place, &weight);

    try std.testing.expectEqualSlices(f32, &expected, &in_place);
}

test "rmsNorm keeps zero input finite" {
    const input = [_]f32{ 0, 0, 0 };
    const weight = [_]f32{ 1, 2, 3 };
    var output: [3]f32 = undefined;

    try rmsNorm(&output, &input, &weight);

    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, &output);
}

test "rmsNorm rejects empty and mismatched inputs" {
    var empty_output = [_]f32{};
    const empty_values = [_]f32{};
    var output: [2]f32 = undefined;
    const input = [_]f32{ 1, 2 };
    const short_weight = [_]f32{1};

    try std.testing.expectError(
        error.EmptyInput,
        rmsNorm(&empty_output, &empty_values, &empty_values),
    );
    try std.testing.expectError(
        error.LengthMismatch,
        rmsNorm(&output, &input, &short_weight),
    );
}

test "swiGlu matches NumPy reference values" {
    var gate = [_]f32{ -2, 0, 2 };
    const up = [_]f32{ 3, -4, 0.5 };
    const expected = [_]f32{
        -0.71521753,
        0,
        0.88079703,
    };

    try swiGlu(&gate, &up);

    for (gate, expected) |actual, expected_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.000001);
    }
}

test "swiGlu rejects empty and mismatched inputs" {
    var empty_gate = [_]f32{};
    const empty_up = [_]f32{};
    var gate = [_]f32{ 1, 2 };
    const short_up = [_]f32{1};

    try std.testing.expectError(
        error.EmptyInput,
        swiGlu(&empty_gate, &empty_up),
    );
    try std.testing.expectError(
        error.LengthMismatch,
        swiGlu(&gate, &short_up),
    );
}

test "applyRope rotates one pair by ninety degrees" {
    var values = [_]f32{ 1, 0 };
    const cosines = [_]f32{0};
    const sines = [_]f32{1};

    try applyRope(&values, &cosines, &sines, 2);

    try std.testing.expectEqualSlices(f32, &.{ 0, 1 }, &values);
}

test "applyRope matches NumPy and resets frequencies for each head" {
    var values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const cosines = [_]f32{ 0.9689124, 0.73168886 };
    const sines = [_]f32{ 0.24740396, 0.6816388 };
    const expected = [_]f32{
        0.4741045,
        2.1852288,
        -0.53148866,
        4.971672,
        3.3601384,
        7.050494,
        -0.33128834,
        10.624983,
    };

    try applyRope(&values, &cosines, &sines, 4);

    for (values, expected) |actual, expected_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.000001);
    }
}

test "applyRope preserves each pair's squared magnitude" {
    const original = [_]f32{ 1, 2, 3, 4 };
    var values = original;
    const cosines = [_]f32{ 0.9689124, 0.73168886 };
    const sines = [_]f32{ 0.24740396, 0.6816388 };

    try applyRope(&values, &cosines, &sines, 4);

    var index: usize = 0;
    while (index < values.len) : (index += 2) {
        const before = original[index] * original[index] +
            original[index + 1] * original[index + 1];
        const after = values[index] * values[index] +
            values[index + 1] * values[index + 1];

        try std.testing.expectApproxEqAbs(before, after, 0.00001);
    }
}

test "applyRope rejects invalid dimensions and shapes" {
    var empty_values = [_]f32{};
    const empty_frequencies = [_]f32{};
    var values = [_]f32{ 1, 2, 3, 4 };
    var three_values = [_]f32{ 1, 2, 3 };
    const one_frequency = [_]f32{1};

    try std.testing.expectError(
        error.EmptyInput,
        applyRope(&empty_values, &empty_frequencies, &empty_frequencies, 2),
    );
    try std.testing.expectError(
        error.InvalidDimension,
        applyRope(&values, &one_frequency, &one_frequency, 0),
    );
    try std.testing.expectError(
        error.InvalidDimension,
        applyRope(&values, &one_frequency, &one_frequency, 3),
    );
    try std.testing.expectError(
        error.LengthMismatch,
        applyRope(&three_values, &one_frequency, &one_frequency, 2),
    );
    try std.testing.expectError(
        error.LengthMismatch,
        applyRope(&values, &one_frequency, &one_frequency, 4),
    );
}
