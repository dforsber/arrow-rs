//! Benchmark comparing SIMD varint decoder vs scalar VLQ decoder
//!
//! Run with: cargo bench --features jit --bench simd_varint_benchmark

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "jit")]
use arrow_avro::reader::simd_varint::{simd_decode_varints_i64, simd_decode_varints_i32};

/// Encode varint
fn encode_varint(mut n: u64, dst: &mut Vec<u8>) {
    while n >= 0x80 {
        dst.push(0x80 | (n as u8));
        n >>= 7;
    }
    dst.push(n as u8);
}

/// Zigzag encode
fn zigzag_encode(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Scalar decode varints (baseline)
fn scalar_decode_varints_i64(input: &[u8], output: &mut [i64]) -> (usize, usize) {
    let mut cursor = 0;
    let mut count = 0;

    while cursor < input.len() && count < output.len() {
        let first = input[cursor];
        if first < 0x80 {
            // Single byte
            let val = first as u64;
            output[count] = ((val >> 1) as i64) ^ (-((val & 1) as i64));
            count += 1;
            cursor += 1;
            continue;
        }

        // Multi-byte
        let mut value = (first & 0x7F) as u64;
        let mut shift = 7;
        let mut pos = 1;

        while pos < (input.len() - cursor).min(10) {
            let byte = input[cursor + pos];
            value |= ((byte & 0x7F) as u64) << shift;
            pos += 1;
            if byte < 0x80 {
                output[count] = ((value >> 1) as i64) ^ (-((value & 1) as i64));
                count += 1;
                cursor += pos;
                break;
            }
            shift += 7;
        }
    }

    (count, cursor)
}

/// Benchmark group for SIMD varint decoding
#[cfg(feature = "jit")]
fn bench_simd_varints(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_decode");

    // Test different batch sizes
    for &size in &[100, 1000, 10000, 100000] {
        // Generate test data: mostly single-byte varints (common case)
        let mut input_small = Vec::new();
        for i in 0..size {
            encode_varint(zigzag_encode((i % 64) as i64), &mut input_small);
        }

        // Generate mixed size varints
        let mut input_mixed = Vec::new();
        for i in 0..size {
            let val = match i % 5 {
                0 => i as i64,           // Small (1 byte)
                1 => 128 + i as i64,     // Medium (2 bytes)
                2 => 16384 + i as i64,   // Large (3 bytes)
                3 => -(i as i64),        // Negative (zigzag)
                _ => 1000000 + i as i64, // Very large
            };
            encode_varint(zigzag_encode(val), &mut input_mixed);
        }

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark SIMD decoder with small varints
        group.bench_with_input(
            BenchmarkId::new("simd_small", size),
            &input_small,
            |b, input| {
                let mut output = vec![0i64; size];
                b.iter(|| {
                    unsafe {
                        simd_decode_varints_i64(
                            black_box(input.as_ptr()),
                            black_box(input.len()),
                            black_box(output.as_mut_ptr()),
                            black_box(size),
                        )
                    }
                });
            },
        );

        // Benchmark scalar decoder with small varints
        group.bench_with_input(
            BenchmarkId::new("scalar_small", size),
            &input_small,
            |b, input| {
                let mut output = vec![0i64; size];
                b.iter(|| {
                    scalar_decode_varints_i64(black_box(input), black_box(&mut output))
                });
            },
        );

        // Benchmark SIMD decoder with mixed varints
        group.bench_with_input(
            BenchmarkId::new("simd_mixed", size),
            &input_mixed,
            |b, input| {
                let mut output = vec![0i64; size];
                b.iter(|| {
                    unsafe {
                        simd_decode_varints_i64(
                            black_box(input.as_ptr()),
                            black_box(input.len()),
                            black_box(output.as_mut_ptr()),
                            black_box(size),
                        )
                    }
                });
            },
        );

        // Benchmark scalar decoder with mixed varints
        group.bench_with_input(
            BenchmarkId::new("scalar_mixed", size),
            &input_mixed,
            |b, input| {
                let mut output = vec![0i64; size];
                b.iter(|| {
                    scalar_decode_varints_i64(black_box(input), black_box(&mut output))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark i32 varint decoding (array lengths, enum indices)
#[cfg(feature = "jit")]
fn bench_i32_varints(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_i32_decode");

    for &size in &[1000, 10000, 100000] {
        // All single-byte (0-127) - common for array lengths
        let input: Vec<u8> = (0..size).map(|i| (i % 128) as u8).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &input,
            |b, input| {
                let mut output = vec![0i32; size];
                b.iter(|| {
                    unsafe {
                        simd_decode_varints_i32(
                            black_box(input.as_ptr()),
                            black_box(input.len()),
                            black_box(output.as_mut_ptr()),
                            black_box(size),
                        )
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &input,
            |b, input| {
                let mut output = vec![0i32; size];
                b.iter(|| {
                    // Simple scalar for comparison
                    for (i, &byte) in input.iter().enumerate().take(size) {
                        output[i] = byte as i32;
                    }
                    black_box(size)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "jit")]
criterion_group!(benches, bench_simd_varints, bench_i32_varints);

#[cfg(not(feature = "jit"))]
fn main() {
    println!("This benchmark requires the 'jit' feature");
}

#[cfg(feature = "jit")]
criterion_main!(benches);
