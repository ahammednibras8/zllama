//! Reading and validating legacy llama2.c checkpoints.

const std = @import("std");

pub const header_size: u64 = 7 * @sizeOf(i32);
pub const Header = [@as(usize, @intCast(header_size))]u8;

pub const ValidationError = error{
    InvalidDimension,
    InvalidHiddenDimension,
    InvalidLayerCount,
    InvalidAttentionHeadCount,
    InvalidKvHeadCount,
    InvalidVocabularySize,
    InvalidSequenceLength,
    IncompatibleAttentionHeads,
    IncompatibleKvHeads,
};

pub const FileSizeError = ValidationError || error{
    CheckpointTooLarge,
};

pub const WeightCounts = struct {
    token_embeddings: u128,

    attention_norm: u128,
    query: u128,
    key: u128,
    value: u128,
    attention_output: u128,

    feed_forward_norm: u128,
    feed_forward_gate: u128,
    feed_forward_down: u128,
    feed_forward_up: u128,

    final_norm: u128,
    rope_real: u128,
    rope_imaginary: u128,
    classifier: u128,

    pub fn total(self: WeightCounts) u128 {
        return self.token_embeddings +
            self.attention_norm +
            self.query +
            self.key +
            self.value +
            self.attention_output +
            self.feed_forward_norm +
            self.feed_forward_gate +
            self.feed_forward_down +
            self.feed_forward_up +
            self.final_norm +
            self.rope_real +
            self.rope_imaginary +
            self.classifier;
    }
};

pub const Config = struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    seq_len: u32,
    shared_classifier: bool,

    pub fn validate(self: Config) ValidationError!void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.hidden_dim == 0) return error.InvalidHiddenDimension;
        if (self.n_layers == 0) return error.InvalidLayerCount;
        if (self.n_heads == 0) return error.InvalidAttentionHeadCount;
        if (self.n_kv_heads == 0) return error.InvalidKvHeadCount;
        if (self.vocab_size == 0) return error.InvalidVocabularySize;
        if (self.seq_len == 0) return error.InvalidSequenceLength;

        if (self.dim % self.n_heads != 0) {
            return error.IncompatibleAttentionHeads;
        }

        if (self.n_heads % self.n_kv_heads != 0) {
            return error.IncompatibleKvHeads;
        }
    }

    pub fn expectedFileSize(self: Config) FileSizeError!u64 {
        const counts = try self.weightCounts();

        const total_bytes =
            @as(u128, header_size) + counts.total() * @sizeOf(f32);

        return std.math.cast(u64, total_bytes) orelse
            error.CheckpointTooLarge;
    }

    pub fn weightCounts(self: Config) ValidationError!WeightCounts {
        try self.validate();

        const dim: u128 = self.dim;
        const hidden_dim: u128 = self.hidden_dim;
        const n_layers: u128 = self.n_layers;
        const n_heads: u128 = self.n_heads;
        const n_kv_heads: u128 = self.n_kv_heads;
        const vocab_size: u128 = self.vocab_size;
        const seq_len: u128 = self.seq_len;

        const head_size = dim / n_heads;
        const kv_dim = head_size * n_kv_heads;
        const rope_component_width = head_size / 2;

        return .{
            .token_embeddings = vocab_size * dim,

            .attention_norm = n_layers * dim,
            .query = n_layers * dim * dim,
            .key = n_layers * dim * kv_dim,
            .value = n_layers * dim * kv_dim,
            .attention_output = n_layers * dim * dim,

            .feed_forward_norm = n_layers * dim,
            .feed_forward_gate = n_layers * hidden_dim * dim,
            .feed_forward_down = n_layers * dim * hidden_dim,
            .feed_forward_up = n_layers * hidden_dim * dim,

            .final_norm = dim,
            .rope_real = seq_len * rope_component_width,
            .rope_imaginary = seq_len * rope_component_width,
            .classifier = if (self.shared_classifier)
                0
            else
                vocab_size * dim,
        };
    }
};

pub fn decodeHeader(bytes: *const Header) ValidationError!Config {
    const raw_dimension = std.mem.readInt(i32, bytes[0..4], .little);
    const raw_hidden_dimension = std.mem.readInt(i32, bytes[4..8], .little);
    const raw_layer_count = std.mem.readInt(i32, bytes[8..12], .little);
    const raw_attention_head_count = std.mem.readInt(i32, bytes[12..16], .little);
    const raw_kv_head_count = std.mem.readInt(i32, bytes[16..20], .little);
    const raw_vocabulary_size = std.mem.readInt(i32, bytes[20..24], .little);
    const raw_sequence_length = std.mem.readInt(i32, bytes[24..28], .little);

    const config = Config{
        .dim = std.math.cast(u32, raw_dimension) orelse
            return error.InvalidDimension,
        .hidden_dim = std.math.cast(u32, raw_hidden_dimension) orelse
            return error.InvalidHiddenDimension,
        .n_layers = std.math.cast(u32, raw_layer_count) orelse
            return error.InvalidLayerCount,
        .n_heads = std.math.cast(u32, raw_attention_head_count) orelse
            return error.InvalidAttentionHeadCount,
        .n_kv_heads = std.math.cast(u32, raw_kv_head_count) orelse
            return error.InvalidKvHeadCount,
        .vocab_size = @abs(raw_vocabulary_size),
        .seq_len = std.math.cast(u32, raw_sequence_length) orelse
            return error.InvalidSequenceLength,
        .shared_classifier = raw_vocabulary_size > 0,
    };

    try config.validate();
    return config;
}

test "legacy checkpoint header contains seven 32-bit integers" {
    try std.testing.expectEqual(@as(u64, 28), header_size);
}

test "decode legacy checkpoint header" {
    const header = Header{
        0x20, 0x01, 0x00, 0x00,
        0x00, 0x03, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x00, 0x7d, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00,
    };

    const config = try decodeHeader(&header);

    try std.testing.expectEqual(@as(u32, 288), config.dim);
    try std.testing.expectEqual(@as(u32, 768), config.hidden_dim);
    try std.testing.expectEqual(@as(u32, 6), config.n_layers);
    try std.testing.expectEqual(@as(u32, 6), config.n_heads);
    try std.testing.expectEqual(@as(u32, 6), config.n_kv_heads);
    try std.testing.expectEqual(@as(u32, 32_000), config.vocab_size);
    try std.testing.expectEqual(@as(u32, 256), config.seq_len);
    try std.testing.expect(config.shared_classifier);
}

test "negative vocabulary size means classifier weights are not shared" {
    const header = Header{
        0x20, 0x01, 0x00, 0x00,
        0x00, 0x03, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x00, 0x83, 0xff, 0xff,
        0x00, 0x01, 0x00, 0x00,
    };

    const config = try decodeHeader(&header);

    try std.testing.expectEqual(@as(u32, 32_000), config.vocab_size);
    try std.testing.expect(!config.shared_classifier);
}

test "decode rejects a negative model dimension" {
    const header = Header{
        0xff, 0xff, 0xff, 0xff,
        0x00, 0x03, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x00, 0x7d, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00,
    };

    try std.testing.expectError(error.InvalidDimension, decodeHeader(&header));
}

test "decode rejects a negative attention head count" {
    var header = Header{
        0x20, 0x01, 0x00, 0x00,
        0x00, 0x03, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00,
        0x00, 0x7d, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00,
    };
    std.mem.writeInt(i32, header[12..16], -1, .little);

    try std.testing.expectError(error.InvalidAttentionHeadCount, decodeHeader(&header));
}

test "valid model configuration passes validation" {
    const config = Config{
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 32_000,
        .seq_len = 256,
        .shared_classifier = true,
    };

    try config.validate();
}

test "calculate legacy stories15M checkpoint size" {
    const config = Config{
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 32_000,
        .seq_len = 256,
        .shared_classifier = true,
    };

    try std.testing.expectEqual(
        @as(u64, 60_816_028),
        try config.expectedFileSize(),
    );
}

test "calculate every legacy stories15M tensor count" {
    const config = Config{
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 32_000,
        .seq_len = 256,
        .shared_classifier = true,
    };

    const counts = try config.weightCounts();

    try std.testing.expectEqual(@as(u128, 9_216_000), counts.token_embeddings);
    try std.testing.expectEqual(@as(u128, 1_728), counts.attention_norm);
    try std.testing.expectEqual(@as(u128, 497_664), counts.query);
    try std.testing.expectEqual(@as(u128, 497_664), counts.key);
    try std.testing.expectEqual(@as(u128, 497_664), counts.value);
    try std.testing.expectEqual(@as(u128, 497_664), counts.attention_output);
    try std.testing.expectEqual(@as(u128, 1_728), counts.feed_forward_norm);
    try std.testing.expectEqual(@as(u128, 1_327_104), counts.feed_forward_gate);
    try std.testing.expectEqual(@as(u128, 1_327_104), counts.feed_forward_down);
    try std.testing.expectEqual(@as(u128, 1_327_104), counts.feed_forward_up);
    try std.testing.expectEqual(@as(u128, 288), counts.final_norm);
    try std.testing.expectEqual(@as(u128, 6_144), counts.rope_real);
    try std.testing.expectEqual(@as(u128, 6_144), counts.rope_imaginary);
    try std.testing.expectEqual(@as(u128, 0), counts.classifier);
    try std.testing.expectEqual(@as(u128, 15_204_000), counts.total());
}

test "separate classifier weights increase checkpoint size" {
    const config = Config{
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 32_000,
        .seq_len = 256,
        .shared_classifier = false,
    };

    try std.testing.expectEqual(
        @as(u64, 97_680_028),
        try config.expectedFileSize(),
    );
}

test "configuration rejects zero dimensions and head counts" {
    const zero_dimension = Config{
        .dim = 0,
        .hidden_dim = 1,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_heads = 1,
        .vocab_size = 1,
        .seq_len = 1,
        .shared_classifier = true,
    };
    try std.testing.expectError(error.InvalidDimension, zero_dimension.validate());

    const zero_heads = Config{
        .dim = 1,
        .hidden_dim = 1,
        .n_layers = 1,
        .n_heads = 0,
        .n_kv_heads = 1,
        .vocab_size = 1,
        .seq_len = 1,
        .shared_classifier = true,
    };
    try std.testing.expectError(error.InvalidAttentionHeadCount, zero_heads.validate());
}

test "attention heads must divide evenly across key value heads" {
    const config = Config{
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 4,
        .vocab_size = 32_000,
        .seq_len = 256,
        .shared_classifier = true,
    };

    try std.testing.expectError(error.IncompatibleKvHeads, config.validate());
}

test "checkpoint size rejects values larger than u64" {
    const maximum = std.math.maxInt(u32);
    const config = Config{
        .dim = maximum,
        .hidden_dim = maximum,
        .n_layers = maximum,
        .n_heads = 1,
        .n_kv_heads = 1,
        .vocab_size = maximum,
        .seq_len = maximum,
        .shared_classifier = false,
    };

    try std.testing.expectError(error.CheckpointTooLarge, config.expectedFileSize());
}

test "model dimension must divide evenly across attention heads" {
    const config = Config{
        .dim = 289,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 32_000,
        .seq_len = 256,
        .shared_classifier = true,
    };

    try std.testing.expectError(error.IncompatibleAttentionHeads, config.validate());
}
