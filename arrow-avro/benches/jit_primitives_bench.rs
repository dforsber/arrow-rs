// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! JIT Decoder Benchmark: Primitive Types
//!
//! Tests: Int32, Int64, Float32, Float64
//! - Single column benchmarks for each type
//! - Multi-column combined benchmarks
//! - Various row counts (1K, 10K, 100K)
//!
//! Run with: cargo bench --features jit --bench jit_primitives_bench

use arrow_avro::reader::jit_decoder::{JitDecoder, OutputBuffer};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;

/// Encode a varint (zig-zag + variable length)
fn encode_varint(buf: &mut Vec<u8>, value: i64) {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    let mut v = zigzag;
    loop {
        if v < 0x80 {
            buf.push(v as u8);
            break;
        }
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
}

// Generate int32 test data
fn gen_int32_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut data, (i as i32) as i64);
    }
    data
}

// Generate int64 test data
fn gen_int64_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut data, i as i64);
    }
    data
}

// Generate float32 test data
fn gen_float32_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }
    data
}

// Generate float64 test data
fn gen_float64_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        data.extend_from_slice(&(i as f64).to_le_bytes());
    }
    data
}

// Generate combined primitives data
fn gen_combined_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut data, (i as i32) as i64);
        encode_varint(&mut data, i as i64);
        data.extend_from_slice(&(i as f32).to_le_bytes());
        data.extend_from_slice(&(i as f64).to_le_bytes());
    }
    data
}

fn bench_int32(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_int32");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let data = gen_int32_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", row_count),
            &row_count,
            |b, &n| {
                b.iter(|| {
                    let mut output = vec![0u8; n * 4];
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    let result =
                        unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_int64(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_int64");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let data = gen_int64_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", row_count),
            &row_count,
            |b, &n| {
                b.iter(|| {
                    let mut output = vec![0u8; n * 8];
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    let result =
                        unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_float32(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_float32");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, false)]));
        let data = gen_float32_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", row_count),
            &row_count,
            |b, &n| {
                b.iter(|| {
                    let mut output = vec![0u8; n * 4];
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    let result =
                        unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_float64(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_float64");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
        let data = gen_float64_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", row_count),
            &row_count,
            |b, &n| {
                b.iter(|| {
                    let mut output = vec![0u8; n * 8];
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    let result =
                        unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_combined_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_combined_primitives");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("int32_col", DataType::Int32, false),
            Field::new("int64_col", DataType::Int64, false),
            Field::new("float32_col", DataType::Float32, false),
            Field::new("float64_col", DataType::Float64, false),
        ]));
        let data = gen_combined_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", row_count),
            &row_count,
            |b, &n| {
                b.iter(|| {
                    let mut out0 = vec![0u8; n * 4];
                    let mut out1 = vec![0u8; n * 8];
                    let mut out2 = vec![0u8; n * 4];
                    let mut out3 = vec![0u8; n * 8];
                    let buffers = vec![
                        OutputBuffer::new(out0.as_mut_ptr(), out0.len()),
                        OutputBuffer::new(out1.as_mut_ptr(), out1.len()),
                        OutputBuffer::new(out2.as_mut_ptr(), out2.len()),
                        OutputBuffer::new(out3.as_mut_ptr(), out3.len()),
                    ];
                    let result =
                        unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_int32,
    bench_int64,
    bench_float32,
    bench_float64,
    bench_combined_primitives,
);
criterion_main!(benches);
