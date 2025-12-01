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

//! Benchmarks comparing Vec-based vs Pre-allocated decoding for variable-length
//! (string) columns.

use apache_avro::types::Value;
use apache_avro::{to_avro_datum, Schema as ApacheSchema};
use arrow_avro::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};
use arrow_avro::reader::pooled_decoder::PooledRecordDecoder;
use arrow_avro::reader::ReaderBuilder;
use arrow_avro::schema::{AvroSchema, Fingerprint, FingerprintAlgorithm, SchemaStore, SINGLE_OBJECT_MAGIC};
use arrow_schema::{DataType, Field, Schema};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

// Schema with 3 string columns
const STRINGS_SCHEMA: &str = r#"{
    "type": "record",
    "name": "StringRecord",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
        {"name": "address", "type": "string"}
    ]
}"#;

// Schema with mixed fixed and variable columns
const MIXED_SCHEMA: &str = r#"{
    "type": "record",
    "name": "MixedRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "value", "type": "double"},
        {"name": "description", "type": "string"},
        {"name": "count", "type": "int"}
    ]
}"#;

/// Create SOE prefix (magic + fingerprint) for a schema
fn make_soe_prefix(schema_json: &str) -> Vec<u8> {
    let avro_schema = AvroSchema::new(schema_json.parse().unwrap());
    let fp = avro_schema.fingerprint(FingerprintAlgorithm::Rabin).unwrap();
    let mut prefix = Vec::with_capacity(10);
    prefix.extend_from_slice(&SINGLE_OBJECT_MAGIC);
    if let Fingerprint::Rabin(val) = fp {
        prefix.extend_from_slice(&val.to_le_bytes());
    }
    prefix
}

/// Generate test strings of varying lengths
fn gen_string(idx: usize, base: &str, avg_len: usize) -> String {
    let variation = (idx % 10) as i32 - 5; // -5 to +4
    let len = (avg_len as i32 + variation).max(1) as usize;
    format!("{}{:0>width$}", base, idx % 10000, width = len.saturating_sub(base.len()))
}

/// Generate Avro-encoded data for 3 string columns (raw, no prefix)
fn gen_strings_data(n: usize, avg_string_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(STRINGS_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("name".into(), Value::String(gen_string(i, "name_", avg_string_len))),
            ("email".into(), Value::String(gen_string(i, "email_", avg_string_len))),
            ("address".into(), Value::String(gen_string(i, "addr_", avg_string_len))),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Generate Avro-encoded data for 3 string columns (with SOE prefix)
fn gen_strings_data_soe(n: usize, avg_string_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(STRINGS_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(STRINGS_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("name".into(), Value::String(gen_string(i, "name_", avg_string_len))),
            ("email".into(), Value::String(gen_string(i, "email_", avg_string_len))),
            ("address".into(), Value::String(gen_string(i, "addr_", avg_string_len))),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Generate Avro-encoded data for mixed schema (raw)
fn gen_mixed_data(n: usize, avg_string_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(MIXED_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("name".into(), Value::String(gen_string(i, "name_", avg_string_len))),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("description".into(), Value::String(gen_string(i, "desc_", avg_string_len * 2))),
            ("count".into(), Value::Int((i % 1000) as i32)),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Generate Avro-encoded data for mixed schema (with SOE prefix)
fn gen_mixed_data_soe(n: usize, avg_string_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(MIXED_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(MIXED_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("name".into(), Value::String(gen_string(i, "name_", avg_string_len))),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("description".into(), Value::String(gen_string(i, "desc_", avg_string_len * 2))),
            ("count".into(), Value::Int((i % 1000) as i32)),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Create Arrow schema for 3 strings
fn create_strings_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("email", DataType::Utf8, false),
        Field::new("address", DataType::Utf8, false),
    ]))
}

/// Create Arrow schema for mixed
fn create_mixed_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("description", DataType::Utf8, false),
        Field::new("count", DataType::Int32, false),
    ]))
}

/// Create BufferLayout for string columns
fn create_strings_layout(row_count: usize, avg_string_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // 3 string columns, each needs:
    // - data region: row_count * avg_string_len (with some buffer)
    // - offsets region: (row_count + 1) * 4 bytes
    for _ in 0..3 {
        let data_capacity = row_count * (avg_string_len + 10); // Extra buffer
        let offsets_capacity = (row_count + 1) * 4;

        columns.push(ColumnRegion {
            data_offset: offset,
            data_capacity,
            offsets_offset: Some(offset + data_capacity),
            offsets_capacity: Some(offsets_capacity),
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: vec![],
        });
        offset += data_capacity + offsets_capacity;
    }

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

/// Create BufferLayout for mixed schema
fn create_mixed_layout(row_count: usize, avg_string_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // Column 0: id (Int64) - 8 bytes per row
    let id_capacity = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: id_capacity,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_capacity;

    // Column 1: name (String)
    let name_data_capacity = row_count * (avg_string_len + 10);
    let name_offsets_capacity = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: name_data_capacity,
        offsets_offset: Some(offset + name_data_capacity),
        offsets_capacity: Some(name_offsets_capacity),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += name_data_capacity + name_offsets_capacity;

    // Column 2: value (Float64) - 8 bytes per row
    let value_capacity = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: value_capacity,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += value_capacity;

    // Column 3: description (String) - longer strings
    let desc_data_capacity = row_count * (avg_string_len * 2 + 10);
    let desc_offsets_capacity = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: desc_data_capacity,
        offsets_offset: Some(offset + desc_data_capacity),
        offsets_capacity: Some(desc_offsets_capacity),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += desc_data_capacity + desc_offsets_capacity;

    // Column 4: count (Int32) - 4 bytes per row
    let count_capacity = row_count * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: count_capacity,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += count_capacity;

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

/// Helper to create a Vec-based decoder with schema store
fn create_vec_decoder(schema_json: &str, batch_size: usize) -> arrow_avro::reader::Decoder {
    let avro_schema = AvroSchema::new(schema_json.parse().unwrap());
    let mut store = SchemaStore::new();
    store.register(avro_schema).unwrap();
    ReaderBuilder::new()
        .with_writer_schema_store(store)
        .with_batch_size(batch_size)
        .build_decoder()
        .unwrap()
}

/// Benchmark Pooled decoder for string columns
fn bench_pooled_strings(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pooled_Strings");
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    let avg_string_len = 20;

    for rows in [1000, 10000, 50000] {
        let data = gen_strings_data(rows, avg_string_len);
        let schema = create_strings_schema();
        let layout = create_strings_layout(rows, avg_string_len);

        // Pre-allocate outside benchmark
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout.clone());
        let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::new("3_strings", rows), |b| {
            b.iter(|| {
                decoder.decode(&data, rows).unwrap();
                black_box(decoder.flush().unwrap())
            })
        });
    }
    group.finish();
}

/// Benchmark Vec decoder for string columns
fn bench_vec_strings(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec_Strings");
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    let avg_string_len = 20;

    for rows in [1000, 10000, 50000] {
        let data = gen_strings_data_soe(rows, avg_string_len);

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::new("3_strings", rows), |b| {
            b.iter(|| {
                let mut decoder = create_vec_decoder(STRINGS_SCHEMA, rows);
                decoder.decode(&data).unwrap();
                black_box(decoder.flush().unwrap().unwrap())
            })
        });
    }
    group.finish();
}

/// Direct comparison benchmark for strings
fn bench_string_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pooled_vs_Vec_Strings");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(50);

    let rows = 50000;
    let avg_string_len = 20;

    let data_raw = gen_strings_data(rows, avg_string_len);
    let data_soe = gen_strings_data_soe(rows, avg_string_len);
    let schema = create_strings_schema();
    let layout = create_strings_layout(rows, avg_string_len);

    group.throughput(Throughput::Elements(rows as u64));

    // Pooled (pre-allocated)
    let buffer = vec![0u8; layout.total_size];
    let provider = PooledBufferProvider::new(buffer, layout.clone());
    let mut pooled_decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

    group.bench_function("Pooled_strings_50k", |b| {
        b.iter(|| {
            pooled_decoder.decode(&data_raw, rows).unwrap();
            black_box(pooled_decoder.flush().unwrap())
        })
    });

    // Vec-based
    group.bench_function("Vec_strings_50k", |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(STRINGS_SCHEMA, rows);
            decoder.decode(&data_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    group.finish();
}

/// Direct comparison benchmark for mixed schema
fn bench_mixed_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pooled_vs_Vec_Mixed");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(50);

    let rows = 50000;
    let avg_string_len = 20;

    let data_raw = gen_mixed_data(rows, avg_string_len);
    let data_soe = gen_mixed_data_soe(rows, avg_string_len);
    let schema = create_mixed_schema();
    let layout = create_mixed_layout(rows, avg_string_len);

    group.throughput(Throughput::Elements(rows as u64));

    // Pooled (pre-allocated)
    let buffer = vec![0u8; layout.total_size];
    let provider = PooledBufferProvider::new(buffer, layout.clone());
    let mut pooled_decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

    group.bench_function("Pooled_mixed_50k", |b| {
        b.iter(|| {
            pooled_decoder.decode(&data_raw, rows).unwrap();
            black_box(pooled_decoder.flush().unwrap())
        })
    });

    // Vec-based
    group.bench_function("Vec_mixed_50k", |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(MIXED_SCHEMA, rows);
            decoder.decode(&data_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    group.finish();
}

criterion_group! {
    name = pooled_strings;
    config = Criterion::default().configure_from_args();
    targets = bench_pooled_strings, bench_vec_strings, bench_string_comparison, bench_mixed_comparison
}
criterion_main!(pooled_strings);
