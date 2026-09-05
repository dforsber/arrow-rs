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

//! JIT Decoder Benchmark: Binary Types
//!
//! Tests: Boolean, Bytes (Binary), Fixed, Enum (Dictionary)
//! - Single column benchmarks for each type
//! - Various row counts (1K, 10K, 100K)
//!
//! Run with: cargo bench --features jit --bench jit_binary_bench

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

/// Encode Avro bytes (length + data)
fn encode_avro_bytes(buf: &mut Vec<u8>, data: &[u8]) {
    encode_varint(buf, data.len() as i64);
    buf.extend_from_slice(data);
}

// Generate boolean test data
fn gen_boolean_data(row_count: usize) -> Vec<u8> {
    (0..row_count)
        .map(|i| if i % 2 == 0 { 1u8 } else { 0u8 })
        .collect()
}

// Generate bytes test data with varying lengths
fn gen_bytes_data(row_count: usize, avg_len: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        let len = avg_len + (i % 10); // Vary length slightly
        let payload: Vec<u8> = (0..len).map(|j| ((i + j) % 256) as u8).collect();
        encode_avro_bytes(&mut data, &payload);
    }
    data
}

// Generate fixed test data
fn gen_fixed_data(row_count: usize, size: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        for j in 0..size {
            data.push(((i + j) % 256) as u8);
        }
    }
    data
}

// Generate enum test data
fn gen_enum_data(row_count: usize, num_variants: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut data, (i % num_variants) as i64);
    }
    data
}

fn bench_boolean(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_boolean");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Boolean, false)]));
        let data = gen_boolean_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", row_count),
            &row_count,
            |b, &n| {
                b.iter(|| {
                    // JIT decoder stores booleans as 1 byte per value (not bit-packed)
                    let mut output = vec![0u8; n];
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

fn bench_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_bytes");

    for (avg_len, row_count) in [(10, 10_000), (50, 10_000), (100, 10_000)] {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Binary, false)]));
        let data = gen_bytes_data(row_count, avg_len);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        let id = format!("decode_len{}", avg_len);
        group.bench_with_input(BenchmarkId::new(&id, row_count), &row_count, |b, &n| {
            b.iter(|| {
                let mut output = vec![0u8; n * (avg_len + 20)]; // Extra space for varying lengths
                let mut offsets = vec![0u8; (n + 1) * 4];
                let buffers = vec![
                    OutputBuffer::new(offsets.as_mut_ptr(), offsets.len()),
                    OutputBuffer::new(output.as_mut_ptr(), output.len()),
                ];
                let result = unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_fixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_fixed");

    for fixed_size in [4, 16, 32, 128] {
        let row_count = 10_000;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeBinary(fixed_size),
            false,
        )]));
        let data = gen_fixed_data(row_count, fixed_size as usize);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        let id = format!("decode_size{}", fixed_size);
        group.bench_with_input(BenchmarkId::new(&id, row_count), &row_count, |b, &n| {
            b.iter(|| {
                let mut output = vec![0u8; n * fixed_size as usize];
                let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                let result = unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_enum(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_enum");

    for num_variants in [4, 16, 64, 256] {
        let row_count = 10_000;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        )]));
        let data = gen_enum_data(row_count, num_variants);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        let id = format!("decode_variants{}", num_variants);
        group.bench_with_input(BenchmarkId::new(&id, row_count), &row_count, |b, &n| {
            b.iter(|| {
                let mut output = vec![0u8; n * 4];
                let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                let result = unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_boolean, bench_bytes, bench_fixed, bench_enum,);
criterion_main!(benches);
