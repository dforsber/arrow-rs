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

//! JIT Decoder Benchmark: Complex Types
//!
//! Tests: supported List and Struct layouts
//! - Single column benchmarks for each type
//! - Various configurations (list lengths, struct fields, union variants)
//!
//! Run with: cargo bench --features jit --bench jit_complex_bench

use arrow_avro::reader::jit_decoder::{JitDecoder, OutputBuffer};
use arrow_schema::{DataType, Field, Fields, Schema};
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

/// Encode unsigned varint (for union type index)
fn encode_block_count(buf: &mut Vec<u8>, count: i64) {
    encode_varint(buf, count);
}

/// Encode Avro bytes (length + data)
fn gen_list_data(row_count: usize, avg_len: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        let list_len = (avg_len + (i % 3)) as i64; // Vary length: avg, avg+1, avg+2
        encode_block_count(&mut data, list_len);
        for j in 0..list_len as usize {
            encode_varint(&mut data, (i * 10 + j) as i64);
        }
        encode_block_count(&mut data, 0); // End of list
    }
    data
}

// Generate struct test data (Int64, Float64)
fn gen_struct_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut data, i as i64);
        data.extend_from_slice(&(i as f64 * 1.5).to_le_bytes());
    }
    data
}

// Generate union test data (Int64 | Float64)
fn bench_list(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_list");

    // Test with different average list lengths
    for avg_len in [1, 3, 10] {
        let row_count = 10_000;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        )]));
        let data = gen_list_data(row_count, avg_len);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        let id = format!("avg_len_{}", avg_len);
        group.bench_with_input(BenchmarkId::new("decode", id), &row_count, |b, &n| {
            b.iter(|| {
                // List requires 2 buffers: offsets (row_count+1 i32s) + data
                let offsets_size = (n + 1) * 4;
                let mut offsets = vec![0u8; offsets_size];
                let data_size = n * (avg_len + 10) * 16;
                let mut data_buf = vec![0u8; data_size];
                let buffers = vec![
                    OutputBuffer::new(offsets.as_mut_ptr(), offsets.len()),
                    OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
                ];
                let result = unsafe { decoder.decode_checked(&data, &buffers, n) }.expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_struct(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_struct");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Float64, false),
            ])),
            false,
        )]));
        let data = gen_struct_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", row_count),
            &row_count,
            |b, &n| {
                b.iter(|| {
                    let mut output = vec![0u8; n * 16];
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

criterion_group!(benches, bench_list, bench_struct,);
criterion_main!(benches);
