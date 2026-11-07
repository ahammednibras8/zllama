//! Borrowed views into legacy llama2.c checkpoint weights.
//!
//! These structures do not own or allocate weight memory. Every slice borrows
//! from the checkpoint mapping supplied by its owner. All slices become invalid
//! when that mapping is destroyed.

const std = @import("std");

pub const LayoutError = error{
    WeightCountOverflow,
    TruncatedWeights,
    TrailingWeights,
};

const Cursor = struct {
    values: []const f32,
    offset: usize = 0,

    fn take(self: *Cursor, count: u128) LayoutError![]const f32 {
        const length = std.math.cast(usize, count) orelse
            return error.WeightCountOverflow;

        const end = std.math.add(usize, self.offset, length) catch
            return error.WeightCountOverflow;

        if (end > self.values.len) {
            return error.TruncatedWeights;
        }

        const result = self.values[self.offset..end];
        self.offset = end;
        return result;
    }

    fn finish(self: Cursor) LayoutError!void {
        if (self.offset != self.values.len) {
            return error.TrailingWeights;
        }
    }
};

pub const AttentionWeights = struct {
    /// Shape: [layers][dimension]
    rms_norm: []const f32,

    /// Shape: [layers][dimension][dimension]
    query: []const f32,

    /// Shape: [layers][dimension][kv_dimension]
    key: []const f32,

    /// Shape: [layers][dimension][kv_dimension]
    value: []const f32,

    /// Shape: [layers][dimension][dimension]
    output: []const f32,
};

pub const FeedForwardWeights = struct {
    /// Shape: [layers][dimension]
    rms_norm: []const f32,

    /// llama2.c w1. Shape: [layers][hidden_dimension][dimension]
    gate: []const f32,

    /// llama2.c w2. Shape: [layers][dimension][hidden_dimension]
    down: []const f32,

    /// llama2.c w3. Shape: [layers][hidden_dimension][dimension]
    up: []const f32,
};

pub const RopeFrequencies = struct {
    /// Shape: [sequence_length][head_size / 2]
    real: []const f32,

    /// Shape: [sequence_length][head_size / 2]
    imaginary: []const f32,
};

pub const Weights = struct {
    /// Shape: [vocabulary_size][dimension]
    token_embeddings: []const f32,

    attention: AttentionWeights,
    feed_forward: FeedForwardWeights,

    /// Shape: [dimension]
    final_norm: []const f32,

    rope: RopeFrequencies,

    /// Shape: [vocabulary_size][dimension].
    /// May alias token_embeddings when classifier weights are shared.
    classifier: []const f32,
};

test "cursor returns consecutive borrowed slices" {
    const values = [_]f32{ 10, 20, 30, 40, 50 };
    var cursor = Cursor{ .values = &values };

    const first = try cursor.take(2);
    const second = try cursor.take(3);

    try std.testing.expectEqualSlices(f32, &.{ 10, 20 }, first);
    try std.testing.expectEqualSlices(f32, &.{ 30, 40, 50 }, second);
    try std.testing.expectEqual(&values[0], &first[0]);
    try cursor.finish();
}

test "cursor rejects truncated weights without advancing" {
    const values = [_]f32{ 10, 20 };
    var cursor = Cursor{ .values = &values };

    try std.testing.expectError(error.TruncatedWeights, cursor.take(3));
    try std.testing.expectEqual(@as(usize, 0), cursor.offset);
}

test "cursor rejects trailing weights" {
    const values = [_]f32{ 10, 20, 30 };
    var cursor = Cursor{ .values = &values };

    _ = try cursor.take(2);

    try std.testing.expectError(error.TrailingWeights, cursor.finish());
}

test "cursor rejects counts that do not fit the address space" {
    const values = [_]f32{};
    var cursor = Cursor{ .values = &values };
    const too_many = @as(u128, std.math.maxInt(usize)) + 1;

    try std.testing.expectError(error.WeightCountOverflow, cursor.take(too_many));
}
