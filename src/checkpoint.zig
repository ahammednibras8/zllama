//! Reading and validating legacy llama2.c checkpoints.

const std = @import("std");

pub const header_size: u64 = 7 * @sizeOf(i32);

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

test "legacy checkpoint header contains seven 32-bit integers" {
    try std.testing.expectEqual(@as(u64, 28), header_size);
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
