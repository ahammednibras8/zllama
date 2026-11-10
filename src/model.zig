//! Ownership for a memory-mapped legacy llama2.c model.

const std = @import("std");
const builtin = @import("builtin");
const checkpoint = @import("checkpoint.zig");
const weight_views = @import("weights.zig");

pub const DecodeError =
    checkpoint.FileSizeError ||
    weight_views.ViewError ||
    error{
        UnsupportedEndianness,
        TruncatedHeader,
        TruncatedCheckpoint,
        TrailingCheckpointData,
    };

pub const OpenError =
    std.Io.File.OpenError ||
    std.Io.File.LengthError ||
    std.Io.File.MemoryMap.CreateError ||
    DecodeError;

const ParsedMapping = struct {
    config: checkpoint.Config,
    weights: weight_views.Weights,
};

fn parseMappedBytes(
    bytes: []align(@alignOf(f32)) const u8,
) DecodeError!ParsedMapping {
    if (builtin.cpu.arch.endian() != .little) {
        return error.UnsupportedEndianness;
    }

    const header_len: usize = @intCast(checkpoint.header_size);

    if (bytes.len < header_len) {
        return error.TruncatedHeader;
    }

    const header: *const checkpoint.Header = @ptrCast(bytes.ptr);
    const config = try checkpoint.decodeHeader(header);

    const expected_len = std.math.cast(
        usize,
        try config.expectedFileSize(),
    ) orelse return error.CheckpointTooLarge;

    if (bytes.len < expected_len) {
        return error.TruncatedCheckpoint;
    }

    if (bytes.len > expected_len) {
        return error.TrailingCheckpointData;
    }

    const weight_bytes: []align(@alignOf(f32)) const u8 = @alignCast(bytes[header_len..]);

    const values = std.mem.bytesAsSlice(f32, weight_bytes);

    return .{
        .config = config,
        .weights = try weight_views.fromSlice(config, values),
    };
}

/// Owns the checkpoint file and its read-only memory mapping.
///
/// Every slice in `weights` points into `mapping.memory`. Those slices become
/// invalid immediately after `deinit` is called.
///
/// The I/O implementation referenced by `io` is borrowed from the caller and
/// must outlive this model.
pub const MappedModel = struct {
    io: std.Io,
    file: std.Io.File,
    mapping: std.Io.File.MemoryMap,
    config: checkpoint.Config,
    weights: weight_views.Weights,

    pub fn deinit(self: *MappedModel) void {
        self.mapping.destroy(self.io);
        self.file.close(self.io);
        self.* = undefined;
    }

    /// Opens and memory-maps a legacy llama2.c FP32 checkpoint.
    ///
    /// Header integers and weight floats are stored in little-endian byte
    /// order. Big-endian targets return `error.UnsupportedEndianness` because
    /// the zero-copy weight views do not perform byte swapping.
    ///
    /// The returned model owns both the file and its mapping. The caller must
    /// call `deinit` before the borrowed `weights` slices go out of use.
    pub fn open(
        io: std.Io,
        dir: std.Io.Dir,
        path: []const u8,
    ) OpenError!MappedModel {
        const file = try dir.openFile(io, path, .{
            .mode = .read_only,
        });
        errdefer file.close(io);

        const file_size = try file.length(io);

        if (file_size < checkpoint.header_size) {
            return error.TruncatedHeader;
        }

        const mapping_len = std.math.cast(usize, file_size) orelse
            return error.CheckpointTooLarge;

        var mapping = try file.createMemoryMap(io, .{
            .len = mapping_len,
            .protection = .{ .read = true },
        });
        errdefer mapping.destroy(io);

        const parsed = try parseMappedBytes(mapping.memory);

        return .{
            .io = io,
            .file = file,
            .mapping = mapping,
            .config = parsed.config,
            .weights = parsed.weights,
        };
    }
};

const minimal_header = checkpoint.Header{
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
};

const minimal_weights = [_]f32{
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
};

fn minimalCheckpointBytes() [72]u8 {
    var bytes: [72]u8 = undefined;
    @memcpy(bytes[0..checkpoint.header_size], &minimal_header);
    @memcpy(
        bytes[checkpoint.header_size..],
        std.mem.sliceAsBytes(&minimal_weights),
    );
    return bytes;
}

test "parse mapped bytes creates zero-copy typed weight views" {
    const bytes align(@alignOf(f32)) = minimalCheckpointBytes();
    const parsed = try parseMappedBytes(&bytes);

    try std.testing.expectEqual(@as(u32, 1), parsed.config.dim);
    try std.testing.expectEqualSlices(
        f32,
        &.{10},
        parsed.weights.token_embeddings,
    );
    try std.testing.expectEqualSlices(
        f32,
        &.{110},
        parsed.weights.final_norm,
    );

    try std.testing.expectEqual(
        @intFromPtr(&bytes[checkpoint.header_size]),
        @intFromPtr(parsed.weights.token_embeddings.ptr),
    );
}

test "parse mapped bytes rejects a truncated header" {
    const bytes align(@alignOf(f32)) = [_]u8{0} ** 27;

    try std.testing.expectError(
        error.TruncatedHeader,
        parseMappedBytes(&bytes),
    );
}

test "parse mapped bytes rejects truncated weights" {
    const bytes align(@alignOf(f32)) = minimalCheckpointBytes();

    try std.testing.expectError(
        error.TruncatedCheckpoint,
        parseMappedBytes(bytes[0 .. bytes.len - 1]),
    );
}

test "parse mapped bytes rejects trailing data" {
    var bytes: [73]u8 align(@alignOf(f32)) = undefined;
    const valid = minimalCheckpointBytes();
    @memcpy(bytes[0..valid.len], &valid);
    bytes[valid.len] = 0;

    try std.testing.expectError(
        error.TrailingCheckpointData,
        parseMappedBytes(&bytes),
    );
}

test "open creates weight views backed by the file mapping" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const bytes = minimalCheckpointBytes();
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.bin",
        .data = &bytes,
    });

    var model = try MappedModel.open(io, tmp.dir, "model.bin");
    defer model.deinit();

    try std.testing.expectEqual(bytes.len, model.mapping.memory.len);
    try std.testing.expectEqualSlices(
        f32,
        &minimal_weights,
        std.mem.bytesAsSlice(
            f32,
            @as([]align(@alignOf(f32)) const u8, @alignCast(
                model.mapping.memory[checkpoint.header_size..],
            )),
        ),
    );
    try std.testing.expectEqual(
        @intFromPtr(model.mapping.memory.ptr) + checkpoint.header_size,
        @intFromPtr(model.weights.token_embeddings.ptr),
    );
}

test "open releases its file and mapping when decoding fails" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(io, .{
        .sub_path = "truncated.bin",
        .data = &minimal_header,
    });

    try std.testing.expectError(
        error.TruncatedCheckpoint,
        MappedModel.open(io, tmp.dir, "truncated.bin"),
    );

    try tmp.dir.deleteFile(io, "truncated.bin");
}
