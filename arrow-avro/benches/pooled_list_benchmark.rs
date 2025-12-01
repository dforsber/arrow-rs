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

//! Benchmarks comparing Vec-based vs Pre-allocated List decoding in PooledRecordDecoder

use apache_avro::types::Value;
use apache_avro::{to_avro_datum, Schema as ApacheSchema};
use arrow_array::{Array, ListArray};
use arrow_avro::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};
use arrow_avro::reader::pooled_decoder::PooledRecordDecoder;
use arrow_schema::{DataType, Field, Schema};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

const LIST_INT64_SCHEMA: &str = r#"{
    "type": "record",
    "name": "ListRec",
    "fields": [
        {"name": "values", "type": {"type": "array", "items": "long"}}
    ]
}"#;

/// Generate Avro-encoded data for List<Int64> records (raw datum, no prefix)
fn gen_list_int64_data(n: usize, list_size: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(LIST_INT64_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let items: Vec<Value> = (0..list_size)
            .map(|j| Value::Long((i * list_size + j) as i64))
            .collect();
        let record = Value::Record(vec![("values".into(), Value::Array(items))]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Create Arrow schema for List<Int64>
fn create_list_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "values",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
        false,
    )]))
}

/// Calculate buffer size needed for List<Int64>
fn calculate_list_buffer_size(row_count: usize, elements_per_row: usize) -> usize {
    // Offsets: (row_count + 1) * 4 bytes (i32)
    // Child data: row_count * elements_per_row * 8 bytes (i64)
    let offsets_size = (row_count + 1) * 4;
    let child_data_size = row_count * elements_per_row * 8;
    offsets_size + child_data_size + 1024 // Extra padding
}

/// Create a ColumnRegion for List<Int64> WITHOUT child regions (Vec mode fallback)
fn create_list_region_vec_mode(row_count: usize, elements_per_row: usize) -> ColumnRegion {
    let offsets_size = (row_count + 1) * 4;
    let child_data_size = row_count * elements_per_row * 8;

    ColumnRegion {
        data_offset: 0,
        data_capacity: offsets_size + child_data_size + 512,
        offsets_offset: Some(0),
        offsets_capacity: Some(offsets_size),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![], // No child regions = Vec fallback mode
    }
}

/// Create a ColumnRegion for List<Int64> WITH child regions (pre-allocated mode)
fn create_list_region_preallocated(row_count: usize, elements_per_row: usize) -> ColumnRegion {
    let offsets_size = (row_count + 1) * 4;
    let child_data_size = row_count * elements_per_row * 8;
    let child_offset = offsets_size + 64; // Start child region after offsets with padding

    ColumnRegion {
        data_offset: 0,
        data_capacity: offsets_size + child_data_size + 512,
        offsets_offset: Some(0),
        offsets_capacity: Some(offsets_size),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![ColumnRegion {
            data_offset: child_offset,
            data_capacity: child_data_size,
            offsets_offset: None,
            offsets_capacity: None,
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: vec![],
        }],
    }
}

fn bench_vec_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("List_Vec_Mode");
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    for &(rows, list_size) in &[(1000, 10), (10000, 10), (1000, 100), (10000, 100)] {
        let data = gen_list_int64_data(rows, list_size);
        let schema = create_list_schema();
        let buffer_size = calculate_list_buffer_size(rows, list_size);

        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_function(
            BenchmarkId::new("decode", format!("{rows}x{list_size}")),
            |b| {
                b.iter(|| {
                    // Use Vec mode region (no child regions = auto-detects as Vec fallback)
                    let layout = BufferLayout {
                        columns: vec![create_list_region_vec_mode(rows, list_size)],
                        total_size: buffer_size,
                        row_count: rows,
                    };
                    let buffer = vec![0u8; buffer_size];
                    let provider = PooledBufferProvider::new(buffer, layout);
                    let mut decoder =
                        PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

                    // Vec mode - no child regions configured, auto-falls back to Vec
                    decoder.decode(&data, rows).unwrap();
                    let batch = black_box(decoder.flush().unwrap());

                    // Verify
                    let list_col = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap();
                    assert_eq!(list_col.len(), rows);
                    batch
                })
            },
        );
    }
    group.finish();
}

fn bench_preallocated_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("List_Preallocated_Mode");
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    for &(rows, list_size) in &[(1000, 10), (10000, 10), (1000, 100), (10000, 100)] {
        let data = gen_list_int64_data(rows, list_size);
        let schema = create_list_schema();
        let buffer_size = calculate_list_buffer_size(rows, list_size);

        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_function(
            BenchmarkId::new("decode", format!("{rows}x{list_size}")),
            |b| {
                b.iter(|| {
                    // Use pre-allocated region (with child regions = auto-enables pre-allocation)
                    let layout = BufferLayout {
                        columns: vec![create_list_region_preallocated(rows, list_size)],
                        total_size: buffer_size,
                        row_count: rows,
                    };
                    let buffer = vec![0u8; buffer_size];
                    let provider = PooledBufferProvider::new(buffer, layout);
                    let mut decoder =
                        PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

                    // Pre-allocated mode auto-enabled via child region detection

                    // Decode all rows
                    decoder.decode(&data, rows).unwrap();
                    let batch = black_box(decoder.flush().unwrap());

                    // Verify
                    let list_col = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap();
                    assert_eq!(list_col.len(), rows);
                    batch
                })
            },
        );
    }
    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("List_Vec_vs_Preallocated");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(50);

    // Test with 10000 rows, 50 elements per list (realistic workload)
    let rows = 10000;
    let list_size = 50;
    let data = gen_list_int64_data(rows, list_size);
    let schema = create_list_schema();
    let buffer_size = calculate_list_buffer_size(rows, list_size);

    group.throughput(Throughput::Elements(rows as u64));

    // Vec mode (no child regions configured)
    group.bench_function("Vec", |b| {
        b.iter(|| {
            let layout = BufferLayout {
                columns: vec![create_list_region_vec_mode(rows, list_size)],
                total_size: buffer_size,
                row_count: rows,
            };
            let buffer = vec![0u8; buffer_size];
            let provider = PooledBufferProvider::new(buffer, layout);
            let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

            decoder.decode(&data, rows).unwrap();
            black_box(decoder.flush().unwrap())
        })
    });

    // Pre-allocated mode (child regions configured = auto-enabled)
    group.bench_function("Preallocated", |b| {
        b.iter(|| {
            let layout = BufferLayout {
                columns: vec![create_list_region_preallocated(rows, list_size)],
                total_size: buffer_size,
                row_count: rows,
            };
            let buffer = vec![0u8; buffer_size];
            let provider = PooledBufferProvider::new(buffer, layout);
            let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

            decoder.decode(&data, rows).unwrap();
            black_box(decoder.flush().unwrap())
        })
    });

    group.finish();
}

criterion_group! {
    name = pooled_list;
    config = Criterion::default().configure_from_args();
    targets = bench_vec_mode, bench_preallocated_mode, bench_comparison
}
criterion_main!(pooled_list);
