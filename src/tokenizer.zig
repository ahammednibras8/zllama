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

/// One vocabulary record.
///
/// `piece` does not own its bytes. It points into the owning
/// `Tokenizer.storage` allocation.
pub const VocabularyEntry = struct {
    piece: []const u8,
    score: f32,
};

/// Owns a loaded tokenizer and every vocabulary piece.
///
/// Each entry in `vocabulary` borrows its `piece` bytes from `storage`.
/// All entries and pieces become invalid immediately after `deinit`.
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    storage: []const u8,
    vocabulary: []const VocabularyEntry,
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

        return .{
            .allocator = allocator,
            .storage = storage,
            .vocabulary = parsed.entries,
            .max_token_length = parsed.max_token_length,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
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
