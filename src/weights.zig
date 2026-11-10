//! Borrowed views into legacy llama2.c checkpoint weights.
//!
//! These structures do not own or allocate weight memory. Every slice borrows
//! from the checkpoint mapping supplied by its owner. All slices become invalid
//! when that mapping is destroyed.

const std = @import("std");
const checkpoint = @import("checkpoint.zig");

pub const LayoutError = error{
    WeightCountOverflow,
    TruncatedWeights,
    TrailingWeights,
};

pub const ViewError = LayoutError || checkpoint.ValidationError;

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

pub fn fromSlice(
    config: checkpoint.Config,
    values: []const f32,
) ViewError!Weights {
    const counts = try config.weightCounts();
    var cursor = Cursor{ .values = values };

    const token_embeddings = try cursor.take(counts.token_embeddings);

    const attention_norm = try cursor.take(counts.attention_norm);
    const query = try cursor.take(counts.query);
    const key = try cursor.take(counts.key);
    const value = try cursor.take(counts.value);
    const attention_output = try cursor.take(counts.attention_output);

    const feed_forward_norm = try cursor.take(counts.feed_forward_norm);
    const feed_forward_gate = try cursor.take(counts.feed_forward_gate);
    const feed_forward_down = try cursor.take(counts.feed_forward_down);
    const feed_forward_up = try cursor.take(counts.feed_forward_up);

    const final_norm = try cursor.take(counts.final_norm);
    const rope_real = try cursor.take(counts.rope_real);
    const rope_imaginary = try cursor.take(counts.rope_imaginary);

    const classifier = if (config.shared_classifier)
        token_embeddings
    else
        try cursor.take(counts.classifier);

    try cursor.finish();

    return .{
        .token_embeddings = token_embeddings,
        .attention = .{
            .rms_norm = attention_norm,
            .query = query,
            .key = key,
            .value = value,
            .output = attention_output,
        },
        .feed_forward = .{
            .rms_norm = feed_forward_norm,
            .gate = feed_forward_gate,
            .down = feed_forward_down,
            .up = feed_forward_up,
        },
        .final_norm = final_norm,
        .rope = .{
            .real = rope_real,
            .imaginary = rope_imaginary,
        },
        .classifier = classifier,
    };
}

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

const view_test_config = checkpoint.Config{
    .dim = 2,
    .hidden_dim = 1,
    .n_layers = 1,
    .n_heads = 1,
    .n_kv_heads = 1,
    .vocab_size = 2,
    .seq_len = 2,
    .shared_classifier = true,
};

fn expectBorrowedSlice(
    backing: []const f32,
    actual: []const f32,
    expected_start: usize,
    expected_len: usize,
) !void {
    try std.testing.expectEqual(expected_len, actual.len);
    try std.testing.expectEqual(&backing[expected_start], &actual[0]);
    try std.testing.expectEqual(backing[expected_start], actual[0]);
}

test "construct every shared-classifier tensor view in checkpoint order" {
    var values: [36]f32 = undefined;
    for (&values, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }

    const model = try fromSlice(view_test_config, &values);

    try expectBorrowedSlice(&values, model.token_embeddings, 0, 4);
    try expectBorrowedSlice(&values, model.attention.rms_norm, 4, 2);
    try expectBorrowedSlice(&values, model.attention.query, 6, 4);
    try expectBorrowedSlice(&values, model.attention.key, 10, 4);
    try expectBorrowedSlice(&values, model.attention.value, 14, 4);
    try expectBorrowedSlice(&values, model.attention.output, 18, 4);
    try expectBorrowedSlice(&values, model.feed_forward.rms_norm, 22, 2);
    try expectBorrowedSlice(&values, model.feed_forward.gate, 24, 2);
    try expectBorrowedSlice(&values, model.feed_forward.down, 26, 2);
    try expectBorrowedSlice(&values, model.feed_forward.up, 28, 2);
    try expectBorrowedSlice(&values, model.final_norm, 30, 2);
    try expectBorrowedSlice(&values, model.rope.real, 32, 2);
    try expectBorrowedSlice(&values, model.rope.imaginary, 34, 2);
    try expectBorrowedSlice(&values, model.classifier, 0, 4);
}

test "construct separate classifier view after all shared tensors" {
    var config = view_test_config;
    config.shared_classifier = false;

    var values: [40]f32 = undefined;
    for (&values, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }

    const model = try fromSlice(config, &values);

    try expectBorrowedSlice(&values, model.token_embeddings, 0, 4);
    try expectBorrowedSlice(&values, model.classifier, 36, 4);
}

test "tensor view rejects truncated and trailing weights" {
    const truncated = [_]f32{0} ** 35;
    try std.testing.expectError(
        error.TruncatedWeights,
        fromSlice(view_test_config, &truncated),
    );

    const trailing = [_]f32{0} ** 37;
    try std.testing.expectError(
        error.TrailingWeights,
        fromSlice(view_test_config, &trailing),
    );
}
