# zllama

An educational, dependency-free Zig implementation of
[karpathy/llama2.c](https://github.com/karpathy/llama2.c).

## Status

Under development. The current executable is only the initial Zig project
template and does not perform model inference yet.

## Initial target

- Run FP32 TinyStories checkpoints on a CPU.
- Implement the inference path directly in Zig.
- Keep the implementation small enough to study from end to end.
- Preserve a clear separation between reusable inference code and the CLI.
- Validate behavior with automated tests and reference outputs.

## Non-goals

- Training models.
- GPU acceleration.
- Quantized inference.
- Compatibility with every Llama model format.
- Copying or consulting existing Zig ports before the first working version.

## Development

Requires Zig 0.16.0.

```sh
zig build
zig build run
zig build test
zig build check

```

`zig build check` verifies formatting, compiles the executable, and runs all
automated tests.

## Reference

This project independently reimplements the inference behavior described by
[`karpathy/llama2.c`](https://github.com/karpathy/llama2.c). The original
project is licensed under the MIT License.

## License

zllama is licensed under the [MIT License](LICENSE).
