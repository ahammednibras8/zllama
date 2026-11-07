const std = @import("std");
const builtin = @import("builtin");

const required_zig_version = std.SemanticVersion{
    .major = 0,
    .minor = 16,
    .patch = 0,
};

comptime {
    const current = builtin.zig_version;
    const matches =
        current.major == required_zig_version.major and
        current.minor == required_zig_version.minor and
        current.patch == required_zig_version.patch;

    if (!matches) {
        @compileError(std.fmt.comptimePrint(
            "zllama requires Zig {}, but the current compiler is {}",
            .{ required_zig_version, current },
        ));
    }
}

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("zllama", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
    });

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "zllama",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "zllama" is the name you will use in your source code to
                // import this module (e.g. `@import("zllama")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "zllama", .module = mod },
            },
        }),
    });

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_files = b.addWriteFiles();
    const valid_checkpoint_header = test_files.add(
        "stories15M-header.bin",
        "\x20\x01\x00\x00" ++
            "\x00\x03\x00\x00" ++
            "\x06\x00\x00\x00" ++
            "\x06\x00\x00\x00" ++
            "\x06\x00\x00\x00" ++
            "\x00\x7d\x00\x00" ++
            "\x00\x01\x00\x00",
    );

    const run_cli_valid = b.addRunArtifact(exe);
    run_cli_valid.setName("test CLI reads complete checkpoint header");
    run_cli_valid.addArg("inspect");
    run_cli_valid.addFileArg(valid_checkpoint_header);
    run_cli_valid.expectStdErrEqual(
        \\Dimension: 288
        \\Hidden dimension: 768
        \\Layers: 6
        \\Attention heads: 6
        \\KV heads: 6
        \\Vocabulary size: 32000
        \\Sequence length: 256
        \\
    );
    run_cli_valid.expectExitCode(0);

    const run_cli_too_few = b.addRunArtifact(exe);
    run_cli_too_few.setName("test CLI missing arguments");
    run_cli_too_few.expectStdErrEqual("Usage: zllama inspect <checkpoint>\n");
    run_cli_too_few.expectExitCode(1);

    const run_cli_too_many = b.addRunArtifact(exe);
    run_cli_too_many.setName("test CLI excess arguments");
    run_cli_too_many.addArgs(&.{ "inspect", "stories15M.bin", "extra" });
    run_cli_too_many.expectStdErrEqual("Usage: zllama inspect <checkpoint>\n");
    run_cli_too_many.expectExitCode(1);

    const run_cli_unknown_command = b.addRunArtifact(exe);
    run_cli_unknown_command.setName("test CLI unknown command");
    run_cli_unknown_command.addArgs(&.{ "banana", "stories15M.bin" });
    run_cli_unknown_command.expectStdErrEqual(
        \\Unknown command: banana
        \\Usage: zllama inspect <checkpoint>
        \\
    );
    run_cli_unknown_command.expectExitCode(1);

    const run_cli_empty_path = b.addRunArtifact(exe);
    run_cli_empty_path.setName("test CLI empty checkpoint path");
    run_cli_empty_path.addArgs(&.{ "inspect", "" });
    run_cli_empty_path.expectStdErrEqual("Checkpoint path cannot be empty\n");
    run_cli_empty_path.expectExitCode(1);

    const run_cli_missing_file = b.addRunArtifact(exe);
    run_cli_missing_file.setName("test CLI missing checkpoint file");
    run_cli_missing_file.addArgs(&.{ "inspect", "does-not-exist.bin" });
    run_cli_missing_file.expectStdErrEqual(
        "Unable to open checkpoint 'does-not-exist.bin': FileNotFound\n",
    );
    run_cli_missing_file.expectExitCode(1);

    const run_cli_short_file = b.addRunArtifact(exe);
    run_cli_short_file.setName("test CLI truncated checkpoint header");
    run_cli_short_file.addArgs(&.{ "inspect", ".gitattributes" });
    run_cli_short_file.expectStdErrEqual(
        "Checkpoint '.gitattributes' is too small: expected at least 28 bytes, found 19\n",
    );
    run_cli_short_file.expectExitCode(1);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_cli_valid.step);
    test_step.dependOn(&run_cli_too_few.step);
    test_step.dependOn(&run_cli_too_many.step);
    test_step.dependOn(&run_cli_unknown_command.step);
    test_step.dependOn(&run_cli_empty_path.step);
    test_step.dependOn(&run_cli_missing_file.step);
    test_step.dependOn(&run_cli_short_file.step);

    const fmt_check = b.addFmt(.{
        .paths = &.{ "build.zig", "src" },
        .check = true,
    });

    const check_step = b.step("check", "Check formatting, compilation, and tests");
    check_step.dependOn(&fmt_check.step);
    check_step.dependOn(&exe.step);
    check_step.dependOn(&run_mod_tests.step);
    check_step.dependOn(&run_exe_tests.step);
    check_step.dependOn(&run_cli_valid.step);
    check_step.dependOn(&run_cli_too_few.step);
    check_step.dependOn(&run_cli_too_many.step);
    check_step.dependOn(&run_cli_unknown_command.step);
    check_step.dependOn(&run_cli_empty_path.step);
    check_step.dependOn(&run_cli_missing_file.step);
    check_step.dependOn(&run_cli_short_file.step);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}
