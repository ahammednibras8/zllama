const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    if (args.len != 3) {
        std.debug.print("Usage: zllama inspect <checkpoint>\n", .{});
        std.process.exit(1);
    }

    if (!std.mem.eql(u8, args[1], "inspect")) {
        std.debug.print("Unknown command: {s}\n", .{args[1]});
        std.debug.print("Usage: zllama inspect <checkpoint>\n", .{});
        std.process.exit(1);
    }

    std.debug.print("command: {s}\n", .{args[1]});
    std.debug.print("checkpoint: {s}\n", .{args[2]});
}
