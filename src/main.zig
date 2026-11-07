const std = @import("std");
const Io = std.Io;
const zllama = @import("zllama");

pub fn main(init: std.process.Init) !void {
    const exit_code = try run(init);

    if (exit_code != 0) {
        std.process.exit(exit_code);
    }
}

fn run(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    if (args.len != 3) {
        std.debug.print("Usage: zllama inspect <checkpoint>\n", .{});
        return 1;
    }

    if (!std.mem.eql(u8, args[1], "inspect")) {
        std.debug.print("Unknown command: {s}\n", .{args[1]});
        std.debug.print("Usage: zllama inspect <checkpoint>\n", .{});
        return 1;
    }

    const checkpoint_path = args[2];

    if (checkpoint_path.len == 0) {
        std.debug.print("Checkpoint path cannot be empty\n", .{});
        return 1;
    }

    const io = init.io;

    const file = Io.Dir.cwd().openFile(io, checkpoint_path, .{
        .mode = .read_only,
    }) catch |err| {
        std.debug.print(
            "Unable to open checkpoint '{s}': {s}\n",
            .{ checkpoint_path, @errorName(err) },
        );
        return 1;
    };
    defer file.close(io);

    const file_size = file.length(io) catch |err| {
        std.debug.print(
            "Unable to read checkpoint size for '{s}': {s}\n",
            .{ checkpoint_path, @errorName(err) },
        );
        return 1;
    };

    if (file_size < zllama.checkpoint.header_size) {
        std.debug.print(
            "Checkpoint '{s}' is too small: expected at least {d} bytes, found {d}\n",
            .{ checkpoint_path, zllama.checkpoint.header_size, file_size },
        );
        return 1;
    }

    var header_bytes: zllama.checkpoint.Header = undefined;

    var file_reader = file.reader(io, &.{});
    file_reader.interface.readSliceAll(&header_bytes) catch |err| {
        std.debug.print(
            "Unable to read checkpoint header from '{s}': {s}\n",
            .{ checkpoint_path, @errorName(err) },
        );
        return 1;
    };

    const config = zllama.checkpoint.decodeHeader(&header_bytes) catch |err| {
        std.debug.print(
            "Invalid checkpoint configuration in '{s}': {s}\n",
            .{ checkpoint_path, @errorName(err) },
        );
        return 1;
    };

    std.debug.print("Dimension: {d}\n", .{config.dim});
    std.debug.print("Hidden dimension: {d}\n", .{config.hidden_dim});
    std.debug.print("Layers: {d}\n", .{config.n_layers});
    std.debug.print("Attention heads: {d}\n", .{config.n_heads});
    std.debug.print("KV heads: {d}\n", .{config.n_kv_heads});
    std.debug.print("Vocabulary size: {d}\n", .{config.vocab_size});
    std.debug.print("Sequence length: {d}\n", .{config.seq_len});

    return 0;
}
