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
