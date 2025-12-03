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

//! Benchmark JIT decoder for each supported type
//!
//! This benchmark tests all simple Avro types using the JIT-compiled decoder:
//! - Primitives: int32, int64, float32, float64
//! - Boolean, Fixed, Enum (Dictionary)
//! - Decimal128, Duration, Interval
//!
//! Due to Cranelift JIT module memory limits, all types are combined into a single
//! multi-column schema. The benchmark produces valid results but may crash during
//! cleanup due to a known Cranelift JITModule cleanup issue (doesn't affect results).
//!
//! Run with: cargo bench --features jit --bench jit_per_type_benchmark

use arrow_avro::reader::jit_decoder::{JitDecoder, OutputBuffer};
use arrow_schema::{DataType, Field, IntervalUnit, Schema, TimeUnit};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;

const ROW_COUNT: usize = 10000;

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

// Benchmark all primitive and simple types in a single combined decoder
fn bench_all_simple_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_types");
    group.throughput(Throughput::Elements(ROW_COUNT as u64));

    // Combined schema for primitives + boolean + fixed + enum + decimal + duration + interval
    let schema = Arc::new(Schema::new(vec![
        // Primitives
        Field::new("int32_col", DataType::Int32, false),
        Field::new("int64_col", DataType::Int64, false),
        Field::new("float32_col", DataType::Float32, false),
        Field::new("float64_col", DataType::Float64, false),
        // Boolean
        Field::new("bool_col", DataType::Boolean, false),
        // Fixed
        Field::new("fixed_col", DataType::FixedSizeBinary(16), false),
        // Enum (Dictionary)
        Field::new("enum_col", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), false),
        // Decimal128
        Field::new("decimal_col", DataType::Decimal128(38, 10), false),
        // Duration
        Field::new("duration_col", DataType::Duration(TimeUnit::Microsecond), false),
        // Interval
        Field::new("interval_col", DataType::Interval(IntervalUnit::YearMonth), false),
    ]));

    // Generate combined data for all types
    let mut data = Vec::new();
    for i in 0..ROW_COUNT {
        // int32
        encode_varint(&mut data, (i as i32) as i64);
        // int64
        encode_varint(&mut data, i as i64);
        // float32
        data.extend_from_slice(&(i as f32).to_le_bytes());
        // float64
        data.extend_from_slice(&(i as f64).to_le_bytes());
        // boolean
        data.push(if i % 2 == 0 { 1 } else { 0 });
        // fixed (16 bytes)
        data.extend(std::iter::repeat(0xABu8).take(16));
        // enum (varint index)
        encode_varint(&mut data, (i % 4) as i64);
        // decimal128 (16 bytes fixed)
        data.extend(std::iter::repeat(0u8).take(16));
        // duration (varint)
        encode_varint(&mut data, i as i64 * 1000);
        // interval year_month (varint)
        encode_varint(&mut data, (i % 120) as i64);
    }

    let decoder = JitDecoder::compile(&schema).expect("compile");

    group.bench_function("all_simple_types", |b| {
        b.iter(|| {
            let mut out0 = vec![0u8; ROW_COUNT * 4];     // int32
            let mut out1 = vec![0u8; ROW_COUNT * 8];     // int64
            let mut out2 = vec![0u8; ROW_COUNT * 4];     // float32
            let mut out3 = vec![0u8; ROW_COUNT * 8];     // float64
            let mut out4 = vec![0u8; ROW_COUNT]; // boolean (1 byte per row)
            let mut out5 = vec![0u8; ROW_COUNT * 16];    // fixed
            let mut out6 = vec![0u8; ROW_COUNT * 4];     // enum
            let mut out7 = vec![0u8; ROW_COUNT * 16];    // decimal128
            let mut out8 = vec![0u8; ROW_COUNT * 8];     // duration
            let mut out9 = vec![0u8; ROW_COUNT * 4];     // interval

            let buffers = vec![
                OutputBuffer::new(out0.as_mut_ptr(), out0.len()),
                OutputBuffer::new(out1.as_mut_ptr(), out1.len()),
                OutputBuffer::new(out2.as_mut_ptr(), out2.len()),
                OutputBuffer::new(out3.as_mut_ptr(), out3.len()),
                OutputBuffer::new(out4.as_mut_ptr(), out4.len()),
                OutputBuffer::new(out5.as_mut_ptr(), out5.len()),
                OutputBuffer::new(out6.as_mut_ptr(), out6.len()),
                OutputBuffer::new(out7.as_mut_ptr(), out7.len()),
                OutputBuffer::new(out8.as_mut_ptr(), out8.len()),
                OutputBuffer::new(out9.as_mut_ptr(), out9.len()),
            ];
            let result = decoder.decode_checked(&data, &buffers, ROW_COUNT).expect("decode");
            black_box(result);
        });
    });

    drop(decoder);
    group.finish();
}

criterion_group!(
    benches,
    bench_all_simple_types,
);
criterion_main!(benches);
