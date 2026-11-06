const std = @import("std");
const Io = std.Io;

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

    std.debug.print(
        "Opened checkpoint: {s} ({d} bytes)\n",
        .{ checkpoint_path, file_size },
    );
    return 0;
}
