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

//! JIT Decoder Benchmark: Numeric/Special Types
//!
//! Tests: Decimal128, Duration, Interval
//! - Single column benchmarks for each type
//! - Various row counts (1K, 10K, 100K)
//!
//! Run with: cargo bench --features jit --bench jit_numeric_bench

use arrow_avro::reader::jit_decoder::JitDecoder;
use arrow_schema::{DataType, Field, IntervalUnit, Schema, TimeUnit};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
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

// Generate decimal128 test data (16 bytes fixed)
fn gen_decimal128_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        // Create a 16-byte decimal value
        let val = i as i128;
        data.extend_from_slice(&val.to_le_bytes());
    }
    data
}

// Generate duration test data (varint encoded)
fn gen_duration_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut data, (i as i64) * 1000); // Microseconds
    }
    data
}

// Generate interval year_month test data (varint encoded)
fn gen_interval_year_month_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut data, (i % 1200) as i64); // 0-100 years in months
    }
    data
}

// Generate interval day_time test data (8 bytes fixed: 4 bytes days + 4 bytes ms)
fn gen_interval_day_time_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        data.extend_from_slice(&(i as i32).to_le_bytes()); // days
        data.extend_from_slice(&((i * 1000) as i32).to_le_bytes()); // milliseconds
    }
    data
}

// Generate interval month_day_nano test data (16 bytes fixed)
fn gen_interval_month_day_nano_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..row_count {
        data.extend_from_slice(&(i as i32).to_le_bytes()); // months
        data.extend_from_slice(&((i % 31) as i32).to_le_bytes()); // days
        data.extend_from_slice(&((i as i64) * 1_000_000).to_le_bytes()); // nanoseconds
    }
    data
}

fn bench_decimal128(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_decimal128");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Decimal128(38, 10),
            false,
        )]));
        let data = gen_decimal128_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(BenchmarkId::new("decode", row_count), &row_count, |b, &n| {
            b.iter(|| {
                let mut output = vec![0u8; n * 16];
                let mut ptrs = vec![output.as_mut_ptr()];
                let result = decoder.decode(&data, &mut ptrs, n).expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_duration(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_duration");

    // Test different time units
    for (unit, unit_name) in [
        (TimeUnit::Microsecond, "us"),
        (TimeUnit::Nanosecond, "ns"),
        (TimeUnit::Millisecond, "ms"),
    ] {
        let row_count = 10_000;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Duration(unit),
            false,
        )]));
        let data = gen_duration_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(BenchmarkId::new("decode", unit_name), &row_count, |b, &n| {
            b.iter(|| {
                let mut output = vec![0u8; n * 8];
                let mut ptrs = vec![output.as_mut_ptr()];
                let result = decoder.decode(&data, &mut ptrs, n).expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_interval_year_month(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_interval_year_month");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Interval(IntervalUnit::YearMonth),
            false,
        )]));
        let data = gen_interval_year_month_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(BenchmarkId::new("decode", row_count), &row_count, |b, &n| {
            b.iter(|| {
                let mut output = vec![0u8; n * 4];
                let mut ptrs = vec![output.as_mut_ptr()];
                let result = decoder.decode(&data, &mut ptrs, n).expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_interval_day_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_interval_day_time");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Interval(IntervalUnit::DayTime),
            false,
        )]));
        let data = gen_interval_day_time_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(BenchmarkId::new("decode", row_count), &row_count, |b, &n| {
            b.iter(|| {
                let mut output = vec![0u8; n * 8];
                let mut ptrs = vec![output.as_mut_ptr()];
                let result = decoder.decode(&data, &mut ptrs, n).expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_interval_month_day_nano(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_interval_month_day_nano");

    for row_count in [1_000, 10_000, 100_000] {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Interval(IntervalUnit::MonthDayNano),
            false,
        )]));
        let data = gen_interval_month_day_nano_data(row_count);
        let decoder = JitDecoder::compile(&schema).expect("compile");

        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(BenchmarkId::new("decode", row_count), &row_count, |b, &n| {
            b.iter(|| {
                let mut output = vec![0u8; n * 16];
                let mut ptrs = vec![output.as_mut_ptr()];
                let result = decoder.decode(&data, &mut ptrs, n).expect("decode");
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_decimal128,
    bench_duration,
    bench_interval_year_month,
    bench_interval_day_time,
    bench_interval_month_day_nano,
);
criterion_main!(benches);
