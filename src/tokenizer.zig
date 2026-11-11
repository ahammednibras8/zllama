//! Loading and using legacy llama2.c tokenizers.

const std = @import("std");

pub const ParseError = error{
    TruncatedTokenizer,
    TokenizerSizeOverflow,
    InvalidMaxTokenLength,
    EmptyVocabulary,
    ZeroLengthToken,
    TokenLengthExceedsMaximum,
    InvalidTokenScore,
    VocabularyTooLarge,
};

const Cursor = struct {
    bytes: []const u8,
    offset: usize = 0,

    fn take(self: *Cursor, count: usize) ParseError![]const u8 {
        const end = std.math.add(usize, self.offset, count) catch
            return error.TokenizerSizeOverflow;

        if (end > self.bytes.len) {
            return error.TruncatedTokenizer;
        }

        const result = self.bytes[self.offset..end];
        self.offset = end;
        return result;
    }

    fn readU32(self: *Cursor) ParseError!u32 {
        const raw = try self.take(@sizeOf(u32));
        return std.mem.readInt(u32, raw[0..4], .little);
    }

    fn readF32(self: *Cursor) ParseError!f32 {
        const bits = try self.readU32();
        return @bitCast(bits);
    }

    fn isFinished(self: Cursor) bool {
        return self.offset == self.bytes.len;
    }
};

const ParsedVocabulary = struct {
    entries: []VocabularyEntry,
    max_token_length: u32,
};

const ParsedVocabularyError = ParseError || std.mem.Allocator.Error;

pub const max_tokenizer_file_size: usize = 64 * 1024 * 1024;

pub const LoadError =
    std.Io.Dir.ReadFileAllocError ||
    ParsedVocabularyError;

fn parseVocabulary(
    allocator: std.mem.Allocator,
    storage: []const u8,
) ParsedVocabularyError!ParsedVocabulary {
    var cursor = Cursor{ .bytes = storage };

    const max_token_length = try cursor.readU32();

    if (max_token_length == 0) {
        return error.InvalidMaxTokenLength;
    }

    var entries: std.ArrayList(VocabularyEntry) = .empty;
    defer entries.deinit(allocator);

    var largest_piece_length: u32 = 0;

    while (!cursor.isFinished()) {
        const score = try cursor.readF32();

        if (!std.math.isFinite(score)) {
            return error.InvalidTokenScore;
        }

        const piece_length = try cursor.readU32();

        if (piece_length == 0) {
            return error.ZeroLengthToken;
        }

        if (piece_length > max_token_length) {
            return error.TokenLengthExceedsMaximum;
        }

        const length = std.math.cast(usize, piece_length) orelse
            return error.TokenizerSizeOverflow;

        const piece = try cursor.take(length);

        try entries.append(allocator, .{
            .piece = piece,
            .score = score,
        });

        largest_piece_length = @max(largest_piece_length, piece_length);
    }

    if (entries.items.len == 0) {
        return error.EmptyVocabulary;
    }

    if (largest_piece_length != max_token_length) {
        return error.InvalidMaxTokenLength;
    }

    return .{
        .entries = try entries.toOwnedSlice(allocator),
        .max_token_length = max_token_length,
    };
}

pub const TokenId = u32;

pub const LookupError = error{
    TokenIdOutOfRange,
};

pub const bos_token_id: TokenId = 1;
pub const eos_token_id: TokenId = 2;
const byte_fallback_offset: TokenId = 3;

pub const EncodeError =
    std.mem.Allocator.Error ||
    LookupError ||
    error{
        MissingDummyPrefixToken,
        MissingByteFallbackToken,
        TokenLengthOverflow,
    };

pub const EncodeOptions = struct {
    add_bos: bool,
    add_eos: bool,
};

/// One vocabulary record.
///
/// `piece` does not own its bytes. It points into the owning
/// `Tokenizer.storage` allocation.
pub const VocabularyEntry = struct {
    piece: []const u8,
    score: f32,
};

const TokenIndex = struct {
    piece: []const u8,
    id: TokenId,
};

fn tokenIndexLessThan(
    _: void,
    left: TokenIndex,
    right: TokenIndex,
) bool {
    return std.mem.order(u8, left.piece, right.piece) == .lt;
}

fn compareTokenIndex(
    piece: []const u8,
    item: TokenIndex,
) std.math.Order {
    return std.mem.order(u8, piece, item.piece);
}

fn buildSortedVocabulary(
    allocator: std.mem.Allocator,
    vocabulary: []const VocabularyEntry,
) ParsedVocabularyError![]TokenIndex {
    const sorted = try allocator.alloc(TokenIndex, vocabulary.len);
    errdefer allocator.free(sorted);

    for (vocabulary, 0..) |entry, index| {
        const id = std.math.cast(TokenId, index) orelse
            return error.VocabularyTooLarge;

        sorted[index] = .{
            .piece = entry.piece,
            .id = id,
        };
    }

    std.mem.sort(
        TokenIndex,
        sorted,
        {},
        tokenIndexLessThan,
    );

    return sorted;
}

/// Owns a loaded tokenizer and every vocabulary piece.
///
/// Each entry in `vocabulary` borrows its `piece` bytes from `storage`.
/// All entries and pieces become invalid immediately after `deinit`.
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    storage: []const u8,
    vocabulary: []const VocabularyEntry,
    sorted_vocabulary: []const TokenIndex,
    max_token_length: u32,

    pub fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir: std.Io.Dir,
        path: []const u8,
    ) LoadError!Tokenizer {
        const storage = try dir.readFileAlloc(
            io,
            path,
            allocator,
            .limited(max_tokenizer_file_size),
        );
        errdefer allocator.free(storage);

        const parsed = try parseVocabulary(allocator, storage);

        errdefer allocator.free(parsed.entries);

        const sorted_vocabulary = try buildSortedVocabulary(
            allocator,
            parsed.entries,
        );

        return .{
            .allocator = allocator,
            .storage = storage,
            .vocabulary = parsed.entries,
            .sorted_vocabulary = sorted_vocabulary,
            .max_token_length = parsed.max_token_length,
        };
    }

    pub fn idForPiece(
        self: *const Tokenizer,
        piece: []const u8,
    ) ?TokenId {
        const index = std.sort.binarySearch(
            TokenIndex,
            self.sorted_vocabulary,
            piece,
            compareTokenIndex,
        ) orelse return null;

        return self.sorted_vocabulary[index].id;
    }

    fn appendInitialTokens(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        text: []const u8,
        add_bos: bool,
        tokens: *std.ArrayList(TokenId),
    ) EncodeError!void {
        if (add_bos) {
            _ = try self.pieceForId(bos_token_id);
            try tokens.append(allocator, bos_token_id);
        }

        if (text.len != 0) {
            const space_id = self.idForPiece(" ") orelse
                return error.MissingDummyPrefixToken;

            try tokens.append(allocator, space_id);
        }

        var offset: usize = 0;

        while (offset < text.len) {
            var width: usize =
                std.unicode.utf8ByteSequenceLength(text[offset]) catch 1;

            if (width > text.len - offset or
                !std.unicode.utf8ValidateSlice(text[offset .. offset + width]))
            {
                width = 1;
            }

            const piece = text[offset .. offset + width];

            if (self.idForPiece(piece)) |id| {
                try tokens.append(allocator, id);
            } else {
                for (piece) |byte| {
                    const fallback_id =
                        @as(TokenId, byte) + byte_fallback_offset;

                    const fallback_index =
                        std.math.cast(usize, fallback_id) orelse
                        return error.MissingByteFallbackToken;

                    if (fallback_index >= self.vocabulary.len) {
                        return error.MissingByteFallbackToken;
                    }

                    try tokens.append(allocator, fallback_id);
                }
            }

            offset += width;
        }
    }

    fn mergeTokens(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        tokens: *std.ArrayList(TokenId),
    ) EncodeError!void {
        const max_length = std.math.cast(
            usize,
            self.max_token_length,
        ) orelse return error.TokenLengthOverflow;

        const buffer_length = std.math.mul(
            usize,
            max_length,
            2,
        ) catch return error.TokenLengthOverflow;

        const buffer = try allocator.alloc(u8, buffer_length);
        defer allocator.free(buffer);

        while (tokens.items.len >= 2) {
            var best_index: ?usize = null;
            var best_id: TokenId = undefined;
            var best_score = -std.math.inf(f32);

            for (0..tokens.items.len - 1) |index| {
                const left = try self.pieceForId(tokens.items[index]);
                const right = try self.pieceForId(tokens.items[index + 1]);

                const combined_length = std.math.add(
                    usize,
                    left.len,
                    right.len,
                ) catch return error.TokenLengthOverflow;

                if (combined_length > buffer.len) {
                    return error.TokenLengthOverflow;
                }

                @memcpy(buffer[0..left.len], left);
                @memcpy(
                    buffer[left.len..combined_length],
                    right,
                );

                const merged_id = self.idForPiece(
                    buffer[0..combined_length],
                ) orelse continue;

                const merged_index = std.math.cast(
                    usize,
                    merged_id,
                ) orelse return error.TokenIdOutOfRange;

                const score = self.vocabulary[merged_index].score;

                if (score > best_score) {
                    best_score = score;
                    best_id = merged_id;
                    best_index = index;
                }
            }

            const index = best_index orelse break;

            tokens.items[index] = best_id;
            _ = tokens.orderedRemove(index + 1);
        }
    }

    pub fn encode(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        text: []const u8,
        options: EncodeOptions,
    ) EncodeError![]TokenId {
        var tokens: std.ArrayList(TokenId) = .empty;
        errdefer tokens.deinit(allocator);

        try self.appendInitialTokens(
            allocator,
            text,
            options.add_bos,
            &tokens,
        );

        try self.mergeTokens(allocator, &tokens);

        if (options.add_eos) {
            _ = try self.pieceForId(eos_token_id);
            try tokens.append(allocator, eos_token_id);
        }

        return tokens.toOwnedSlice(allocator);
    }

    pub fn pieceForId(
        self: *const Tokenizer,
        token_id: TokenId,
    ) LookupError![]const u8 {
        const index = std.math.cast(usize, token_id) orelse
            return error.TokenIdOutOfRange;

        if (index >= self.vocabulary.len) {
            return error.TokenIdOutOfRange;
        }

        return self.vocabulary[index].piece;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.allocator.free(self.sorted_vocabulary);
        self.allocator.free(self.vocabulary);
        self.allocator.free(self.storage);
        self.* = undefined;
    }
};

test "cursor reads little-endian integers and float bit patterns" {
    const bytes = [_]u8{
        0x78, 0x56, 0x34, 0x12,
        0x00, 0x00, 0xc0, 0x3f,
    };
    var cursor = Cursor{ .bytes = &bytes };

    try std.testing.expectEqual(@as(u32, 0x12345678), try cursor.readU32());
    try std.testing.expectEqual(@as(f32, 1.5), try cursor.readF32());
    try std.testing.expect(cursor.isFinished());
}

test "cursor rejects truncated fields without advancing" {
    const bytes = [_]u8{ 0x01, 0x02, 0x03 };
    var cursor = Cursor{ .bytes = &bytes };

    try std.testing.expectError(error.TruncatedTokenizer, cursor.readU32());
    try std.testing.expectEqual(@as(usize, 0), cursor.offset);
}

test "cursor rejects offset arithmetic overflow" {
    const bytes = [_]u8{};
    var cursor = Cursor{
        .bytes = &bytes,
        .offset = std.math.maxInt(usize),
    };

    try std.testing.expectError(error.TokenizerSizeOverflow, cursor.take(1));
}

test "parse vocabulary preserves file order scores and borrowed pieces" {
    const bytes = [_]u8{
        0x03, 0x00, 0x00, 0x00,

        0x00, 0x00, 0x80, 0xbf,
        0x01, 0x00, 0x00, 0x00,
        ' ',  0x00, 0x00, 0xc0,
        0x3f, 0x03, 0x00, 0x00,
        0x00, 't',  'h',  'e',

        0x00, 0x00, 0x80, 0x3e,
        0x01, 0x00, 0x00, 0x00,
        'x',
    };

    const parsed = try parseVocabulary(std.testing.allocator, &bytes);
    defer std.testing.allocator.free(parsed.entries);

    try std.testing.expectEqual(@as(u32, 3), parsed.max_token_length);
    try std.testing.expectEqual(@as(usize, 3), parsed.entries.len);
    try std.testing.expectEqualStrings(" ", parsed.entries[0].piece);
    try std.testing.expectEqual(@as(f32, -1.0), parsed.entries[0].score);
    try std.testing.expectEqualStrings("the", parsed.entries[1].piece);
    try std.testing.expectEqual(@as(f32, 1.5), parsed.entries[1].score);
    try std.testing.expectEqualStrings("x", parsed.entries[2].piece);
    try std.testing.expectEqual(@as(f32, 0.25), parsed.entries[2].score);
    try std.testing.expectEqual(
        @intFromPtr(&bytes[12]),
        @intFromPtr(parsed.entries[0].piece.ptr),
    );
}

test "parse vocabulary rejects a partial final record" {
    const bytes = [_]u8{
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x3f,
    };

    try std.testing.expectError(
        error.TruncatedTokenizer,
        parseVocabulary(std.testing.allocator, &bytes),
    );
}

test "parse vocabulary rejects a token longer than the declared maximum" {
    const bytes = [_]u8{
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x3f,
        0x02, 0x00, 0x00, 0x00,
        'n',  'o',
    };

    try std.testing.expectError(
        error.TokenLengthExceedsMaximum,
        parseVocabulary(std.testing.allocator, &bytes),
    );
}

test "parse vocabulary rejects a non-finite score" {
    const bytes = [_]u8{
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xc0, 0x7f,
    };

    try std.testing.expectError(
        error.InvalidTokenScore,
        parseVocabulary(std.testing.allocator, &bytes),
    );
}

test "load owns file storage and vocabulary metadata" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const bytes = [_]u8{
        0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0xbf,
        0x01, 0x00, 0x00, 0x00,
        ' ',  0x00, 0x00, 0xc0,
        0x3f, 0x03, 0x00, 0x00,
        0x00, 't',  'h',  'e',
    };
    try tmp.dir.writeFile(io, .{
        .sub_path = "tokenizer.bin",
        .data = &bytes,
    });

    var tokenizer = try Tokenizer.load(
        std.testing.allocator,
        io,
        tmp.dir,
        "tokenizer.bin",
    );
    defer tokenizer.deinit();

    try std.testing.expectEqual(@as(usize, 2), tokenizer.vocabulary.len);
    try std.testing.expectEqualStrings("the", tokenizer.vocabulary[1].piece);
    try std.testing.expectEqual(
        @intFromPtr(tokenizer.storage.ptr) + 21,
        @intFromPtr(tokenizer.vocabulary[1].piece.ptr),
    );
}

test "load releases file storage when parsing fails" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const truncated = [_]u8{
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x3f,
    };
    try tmp.dir.writeFile(io, .{
        .sub_path = "truncated.bin",
        .data = &truncated,
    });

    try std.testing.expectError(
        error.TruncatedTokenizer,
        Tokenizer.load(
            std.testing.allocator,
            io,
            tmp.dir,
            "truncated.bin",
        ),
    );
}

test "piece lookup uses token id as vocabulary index" {
    const vocabulary = [_]VocabularyEntry{
        .{ .piece = "zero", .score = 0 },
        .{ .piece = "one", .score = 0 },
        .{ .piece = "two", .score = 0 },
    };
    const tokenizer = Tokenizer{
        .allocator = std.testing.allocator,
        .storage = "",
        .vocabulary = &vocabulary,
        .sorted_vocabulary = &.{},
        .max_token_length = 4,
    };

    try std.testing.expectEqualStrings("zero", try tokenizer.pieceForId(0));
    try std.testing.expectEqualStrings("two", try tokenizer.pieceForId(2));
}

test "piece lookup rejects token id outside vocabulary" {
    const vocabulary = [_]VocabularyEntry{
        .{ .piece = "only", .score = 0 },
    };
    const tokenizer = Tokenizer{
        .allocator = std.testing.allocator,
        .storage = "",
        .vocabulary = &vocabulary,
        .sorted_vocabulary = &.{},
        .max_token_length = 4,
    };

    try std.testing.expectError(error.TokenIdOutOfRange, tokenizer.pieceForId(1));
    try std.testing.expectError(
        error.TokenIdOutOfRange,
        tokenizer.pieceForId(std.math.maxInt(TokenId)),
    );
}

test "sorted vocabulary keeps original token ids" {
    const vocabulary = [_]VocabularyEntry{
        .{ .piece = "dog", .score = 0 },
        .{ .piece = " a", .score = 0 },
        .{ .piece = "cat", .score = 0 },
    };

    const sorted = try buildSortedVocabulary(
        std.testing.allocator,
        &vocabulary,
    );
    defer std.testing.allocator.free(sorted);

    try std.testing.expectEqualStrings(" a", sorted[0].piece);
    try std.testing.expectEqual(@as(TokenId, 1), sorted[0].id);
    try std.testing.expectEqualStrings("cat", sorted[1].piece);
    try std.testing.expectEqual(@as(TokenId, 2), sorted[1].id);
    try std.testing.expectEqualStrings("dog", sorted[2].piece);
    try std.testing.expectEqual(@as(TokenId, 0), sorted[2].id);

    const tokenizer = Tokenizer{
        .allocator = std.testing.allocator,
        .storage = "",
        .vocabulary = &vocabulary,
        .sorted_vocabulary = sorted,
        .max_token_length = 3,
    };

    try std.testing.expectEqual(@as(TokenId, 1), tokenizer.idForPiece(" a"));
    try std.testing.expectEqual(@as(TokenId, 0), tokenizer.idForPiece("dog"));
    try std.testing.expectEqual(null, tokenizer.idForPiece("missing"));
}

test "initial encoding adds BOS prefix and byte fallback" {
    var vocabulary: [261]VocabularyEntry = undefined;
    for (&vocabulary) |*entry| {
        entry.* = .{ .piece = "unused", .score = 0 };
    }
    vocabulary[259] = .{ .piece = " ", .score = 0 };
    vocabulary[260] = .{ .piece = "A", .score = 0 };

    const sorted = try buildSortedVocabulary(
        std.testing.allocator,
        &vocabulary,
    );
    defer std.testing.allocator.free(sorted);

    const tokenizer = Tokenizer{
        .allocator = std.testing.allocator,
        .storage = "",
        .vocabulary = &vocabulary,
        .sorted_vocabulary = sorted,
        .max_token_length = 6,
    };

    var tokens: std.ArrayList(TokenId) = .empty;
    defer tokens.deinit(std.testing.allocator);

    try tokenizer.appendInitialTokens(
        std.testing.allocator,
        "Aé",
        true,
        &tokens,
    );

    try std.testing.expectEqualSlices(
        TokenId,
        &.{ 1, 259, 260, 198, 172 },
        tokens.items,
    );
}

test "initial encoding of empty prompt only adds requested BOS" {
    const tokenizer = Tokenizer{
        .allocator = std.testing.allocator,
        .storage = "",
        .vocabulary = &.{},
        .sorted_vocabulary = &.{},
        .max_token_length = 1,
    };

    var tokens: std.ArrayList(TokenId) = .empty;
    defer tokens.deinit(std.testing.allocator);

    try tokenizer.appendInitialTokens(
        std.testing.allocator,
        "",
        true,
        &tokens,
    );

    try std.testing.expectEqualSlices(TokenId, &.{1}, tokens.items);
}

test "encode repeatedly merges the highest-scoring adjacent pair" {
    const vocabulary = [_]VocabularyEntry{
        .{ .piece = "unknown", .score = 0 },
        .{ .piece = "BOS", .score = 0 },
        .{ .piece = "EOS", .score = 0 },
        .{ .piece = "unused", .score = 0 },
        .{ .piece = " ", .score = 0 },
        .{ .piece = "a", .score = 0 },
        .{ .piece = "b", .score = 0 },
        .{ .piece = "c", .score = 0 },
        .{ .piece = "ab", .score = 1 },
        .{ .piece = "bc", .score = 2 },
    };

    const sorted = try buildSortedVocabulary(
        std.testing.allocator,
        &vocabulary,
    );
    defer std.testing.allocator.free(sorted);

    const tokenizer = Tokenizer{
        .allocator = std.testing.allocator,
        .storage = "",
        .vocabulary = &vocabulary,
        .sorted_vocabulary = sorted,
        .max_token_length = 7,
    };

    const tokens = try tokenizer.encode(
        std.testing.allocator,
        "abc",
        .{ .add_bos = true, .add_eos = true },
    );
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqualSlices(
        TokenId,
        &.{ 1, 4, 5, 9, 2 },
        tokens,
    );
}
