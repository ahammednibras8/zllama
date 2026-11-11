const std = @import("std");
const Io = std.Io;
const zllama = @import("zllama");

const usage =
    \\Usage: zllama inspect <checkpoint>
    \\       zllama tokenize <prompt>
    \\
;

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
        std.debug.print("{s}", .{usage});
        return 1;
    }

    if (std.mem.eql(u8, args[1], "tokenize")) {
        return tokenizePrompt(init, args[2]);
    }

    if (!std.mem.eql(u8, args[1], "inspect")) {
        std.debug.print("Unknown command: {s}\n", .{args[1]});
        std.debug.print("{s}", .{usage});
        return 1;
    }

    const checkpoint_path = args[2];

    if (checkpoint_path.len == 0) {
        std.debug.print("Checkpoint path cannot be empty\n", .{});
        return 1;
    }

    const io = init.io;

    var model = zllama.model.MappedModel.open(
        io,
        Io.Dir.cwd(),
        checkpoint_path,
    ) catch |err| {
        std.debug.print(
            "Unable to load checkpoint '{s}': {s}\n",
            .{ checkpoint_path, @errorName(err) },
        );
        return 1;
    };
    defer model.deinit();

    const config = model.config;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;

    try stdout.print("Dimension: {d}\n", .{config.dim});
    try stdout.print("Hidden dimension: {d}\n", .{config.hidden_dim});
    try stdout.print("Layers: {d}\n", .{config.n_layers});
    try stdout.print("Attention heads: {d}\n", .{config.n_heads});
    try stdout.print("KV heads: {d}\n", .{config.n_kv_heads});
    try stdout.print("Vocabulary size: {d}\n", .{config.vocab_size});
    try stdout.print("Sequence length: {d}\n", .{config.seq_len});
    try stdout.flush();

    return 0;
}

fn tokenizePrompt(
    init: std.process.Init,
    prompt: []const u8,
) !u8 {
    const allocator = init.arena.allocator();
    const io = init.io;

    var tokenizer = zllama.tokenizer.Tokenizer.load(
        allocator,
        io,
        Io.Dir.cwd(),
        "tokenizer.bin",
    ) catch |err| {
        std.debug.print(
            "Unable to load tokenizer.bin: {s}\n",
            .{@errorName(err)},
        );
        return 1;
    };
    defer tokenizer.deinit();

    const tokens = tokenizer.encode(
        allocator,
        prompt,
        .{ .add_bos = true, .add_eos = false },
    ) catch |err| {
        std.debug.print(
            "Unable to encode prompt: {s}\n",
            .{@errorName(err)},
        );
        return 1;
    };
    defer allocator.free(tokens);

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(
        .stdout(),
        io,
        &stdout_buffer,
    );
    const stdout = &stdout_file_writer.interface;

    try stdout.writeByte('[');

    for (tokens, 0..) |token_id, index| {
        if (index != 0) {
            try stdout.writeAll(", ");
        }

        try stdout.print("{d}", .{token_id});
    }

    try stdout.writeAll("]\n");
    try stdout.flush();

    return 0;
}
