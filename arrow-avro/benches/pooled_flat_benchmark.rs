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

//! Benchmarks comparing Vec-based vs Pre-allocated decoding for flat schemas
//! with fixed-size (non-variable) fields.

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

// Schema with multiple fixed-size columns (typical IoT/metrics schema)
const METRICS_SCHEMA: &str = r#"{
    "type": "record",
    "name": "Metrics",
    "fields": [
        {"name": "timestamp", "type": "long"},
        {"name": "sensor_id", "type": "int"},
        {"name": "temperature", "type": "double"},
        {"name": "humidity", "type": "float"},
        {"name": "pressure", "type": "double"},
        {"name": "battery_level", "type": "int"},
        {"name": "is_active", "type": "boolean"},
        {"name": "reading_count", "type": "long"}
    ]
}"#;

// Simple schema with just primitives
const PRIMITIVES_SCHEMA: &str = r#"{
    "type": "record",
    "name": "Primitives",
    "fields": [
        {"name": "i32_val", "type": "int"},
        {"name": "i64_val", "type": "long"},
        {"name": "f32_val", "type": "float"},
        {"name": "f64_val", "type": "double"}
    ]
}"#;

// Wide schema (many columns)
const WIDE_SCHEMA: &str = r#"{
    "type": "record",
    "name": "Wide",
    "fields": [
        {"name": "col1", "type": "long"},
        {"name": "col2", "type": "long"},
        {"name": "col3", "type": "long"},
        {"name": "col4", "type": "long"},
        {"name": "col5", "type": "long"},
        {"name": "col6", "type": "long"},
        {"name": "col7", "type": "long"},
        {"name": "col8", "type": "long"},
        {"name": "col9", "type": "long"},
        {"name": "col10", "type": "long"},
        {"name": "col11", "type": "double"},
        {"name": "col12", "type": "double"},
        {"name": "col13", "type": "double"},
        {"name": "col14", "type": "double"},
        {"name": "col15", "type": "double"},
        {"name": "col16", "type": "int"},
        {"name": "col17", "type": "int"},
        {"name": "col18", "type": "int"},
        {"name": "col19", "type": "int"},
        {"name": "col20", "type": "int"}
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

/// Generate Avro-encoded data for Metrics schema (raw, no prefix)
fn gen_metrics_data(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(METRICS_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("timestamp".into(), Value::Long(1700000000000 + i as i64)),
            ("sensor_id".into(), Value::Int((i % 1000) as i32)),
            ("temperature".into(), Value::Double(20.0 + (i % 30) as f64)),
            ("humidity".into(), Value::Float(50.0 + (i % 50) as f32)),
            ("pressure".into(), Value::Double(1013.25 + (i % 100) as f64)),
            ("battery_level".into(), Value::Int((100 - i % 100) as i32)),
            ("is_active".into(), Value::Boolean(i % 2 == 0)),
            ("reading_count".into(), Value::Long(i as i64 * 10)),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Generate Avro-encoded data for Metrics schema (with SOE prefix)
fn gen_metrics_data_soe(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(METRICS_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(METRICS_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("timestamp".into(), Value::Long(1700000000000 + i as i64)),
            ("sensor_id".into(), Value::Int((i % 1000) as i32)),
            ("temperature".into(), Value::Double(20.0 + (i % 30) as f64)),
            ("humidity".into(), Value::Float(50.0 + (i % 50) as f32)),
            ("pressure".into(), Value::Double(1013.25 + (i % 100) as f64)),
            ("battery_level".into(), Value::Int((100 - i % 100) as i32)),
            ("is_active".into(), Value::Boolean(i % 2 == 0)),
            ("reading_count".into(), Value::Long(i as i64 * 10)),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Generate Avro-encoded data for Primitives schema (raw)
fn gen_primitives_data(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(PRIMITIVES_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("i32_val".into(), Value::Int(i as i32)),
            ("i64_val".into(), Value::Long(i as i64)),
            ("f32_val".into(), Value::Float(i as f32 + 0.5)),
            ("f64_val".into(), Value::Double(i as f64 + 0.25)),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Generate Avro-encoded data for Primitives schema (with SOE prefix)
fn gen_primitives_data_soe(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(PRIMITIVES_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(PRIMITIVES_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("i32_val".into(), Value::Int(i as i32)),
            ("i64_val".into(), Value::Long(i as i64)),
            ("f32_val".into(), Value::Float(i as f32 + 0.5)),
            ("f64_val".into(), Value::Double(i as f64 + 0.25)),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Generate Avro-encoded data for Wide schema (raw)
fn gen_wide_data(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(WIDE_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("col1".into(), Value::Long(i as i64)),
            ("col2".into(), Value::Long(i as i64 + 1)),
            ("col3".into(), Value::Long(i as i64 + 2)),
            ("col4".into(), Value::Long(i as i64 + 3)),
            ("col5".into(), Value::Long(i as i64 + 4)),
            ("col6".into(), Value::Long(i as i64 + 5)),
            ("col7".into(), Value::Long(i as i64 + 6)),
            ("col8".into(), Value::Long(i as i64 + 7)),
            ("col9".into(), Value::Long(i as i64 + 8)),
            ("col10".into(), Value::Long(i as i64 + 9)),
            ("col11".into(), Value::Double(i as f64)),
            ("col12".into(), Value::Double(i as f64 + 0.1)),
            ("col13".into(), Value::Double(i as f64 + 0.2)),
            ("col14".into(), Value::Double(i as f64 + 0.3)),
            ("col15".into(), Value::Double(i as f64 + 0.4)),
            ("col16".into(), Value::Int(i as i32)),
            ("col17".into(), Value::Int(i as i32 + 1)),
            ("col18".into(), Value::Int(i as i32 + 2)),
            ("col19".into(), Value::Int(i as i32 + 3)),
            ("col20".into(), Value::Int(i as i32 + 4)),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Generate Avro-encoded data for Wide schema (with SOE prefix)
fn gen_wide_data_soe(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(WIDE_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(WIDE_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("col1".into(), Value::Long(i as i64)),
            ("col2".into(), Value::Long(i as i64 + 1)),
            ("col3".into(), Value::Long(i as i64 + 2)),
            ("col4".into(), Value::Long(i as i64 + 3)),
            ("col5".into(), Value::Long(i as i64 + 4)),
            ("col6".into(), Value::Long(i as i64 + 5)),
            ("col7".into(), Value::Long(i as i64 + 6)),
            ("col8".into(), Value::Long(i as i64 + 7)),
            ("col9".into(), Value::Long(i as i64 + 8)),
            ("col10".into(), Value::Long(i as i64 + 9)),
            ("col11".into(), Value::Double(i as f64)),
            ("col12".into(), Value::Double(i as f64 + 0.1)),
            ("col13".into(), Value::Double(i as f64 + 0.2)),
            ("col14".into(), Value::Double(i as f64 + 0.3)),
            ("col15".into(), Value::Double(i as f64 + 0.4)),
            ("col16".into(), Value::Int(i as i32)),
            ("col17".into(), Value::Int(i as i32 + 1)),
            ("col18".into(), Value::Int(i as i32 + 2)),
            ("col19".into(), Value::Int(i as i32 + 3)),
            ("col20".into(), Value::Int(i as i32 + 4)),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

/// Create Arrow schema for Metrics
fn create_metrics_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("sensor_id", DataType::Int32, false),
        Field::new("temperature", DataType::Float64, false),
        Field::new("humidity", DataType::Float32, false),
        Field::new("pressure", DataType::Float64, false),
        Field::new("battery_level", DataType::Int32, false),
        Field::new("is_active", DataType::Boolean, false),
        Field::new("reading_count", DataType::Int64, false),
    ]))
}

/// Create Arrow schema for Primitives
fn create_primitives_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("i32_val", DataType::Int32, false),
        Field::new("i64_val", DataType::Int64, false),
        Field::new("f32_val", DataType::Float32, false),
        Field::new("f64_val", DataType::Float64, false),
    ]))
}

/// Create Arrow schema for Wide
fn create_wide_schema() -> Arc<Schema> {
    let mut fields = Vec::new();
    for i in 1..=10 {
        fields.push(Field::new(format!("col{}", i), DataType::Int64, false));
    }
    for i in 11..=15 {
        fields.push(Field::new(format!("col{}", i), DataType::Float64, false));
    }
    for i in 16..=20 {
        fields.push(Field::new(format!("col{}", i), DataType::Int32, false));
    }
    Arc::new(Schema::new(fields))
}

/// Create BufferLayout for a flat schema with fixed-size columns
fn create_flat_layout(schema: &Schema, row_count: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    for field in schema.fields() {
        let elem_size = match field.data_type() {
            DataType::Boolean => 1,
            DataType::Int32 | DataType::Float32 => 4,
            DataType::Int64 | DataType::Float64 => 8,
            _ => 8, // Default
        };
        let capacity = row_count * elem_size;
        columns.push(ColumnRegion {
            data_offset: offset,
            data_capacity: capacity,
            offsets_offset: None,
            offsets_capacity: None,
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: vec![],
        });
        offset += capacity;
    }

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

/// Benchmark using PooledRecordDecoder (pre-allocated)
///
/// This tests the REAL use case: buffer is pre-allocated ONCE, decoder is created once,
/// and we measure only decode + flush time (no allocation).
/// Note: flush() automatically resets the decoder for the next batch.
fn bench_pooled_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pooled_Flat_Decode");
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    for rows in [1000, 10000, 100000] {
        // Metrics schema (8 columns)
        let data = gen_metrics_data(rows);
        let schema = create_metrics_schema();
        let layout = create_flat_layout(&schema, rows);

        // PRE-ALLOCATE buffer ONCE - this is not part of benchmark
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout.clone());
        let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::new("metrics", rows), |b| {
            b.iter(|| {
                // decode + flush (flush resets decoder for next iteration)
                decoder.decode(&data, rows).unwrap();
                black_box(decoder.flush().unwrap())
            })
        });

        // Primitives schema (4 columns)
        let data = gen_primitives_data(rows);
        let schema = create_primitives_schema();
        let layout = create_flat_layout(&schema, rows);

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout.clone());
        let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        group.bench_function(BenchmarkId::new("primitives", rows), |b| {
            b.iter(|| {
                decoder.decode(&data, rows).unwrap();
                black_box(decoder.flush().unwrap())
            })
        });

        // Wide schema (20 columns)
        let data = gen_wide_data(rows);
        let schema = create_wide_schema();
        let layout = create_flat_layout(&schema, rows);

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout.clone());
        let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        group.bench_function(BenchmarkId::new("wide_20col", rows), |b| {
            b.iter(|| {
                decoder.decode(&data, rows).unwrap();
                black_box(decoder.flush().unwrap())
            })
        });
    }
    group.finish();
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

/// Benchmark using standard Decoder (Vec-based)
fn bench_vec_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec_Flat_Decode");
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    for rows in [1000, 10000, 100000] {
        // Metrics schema (with SOE prefix for Vec decoder)
        let data = gen_metrics_data_soe(rows);

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::new("metrics", rows), |b| {
            b.iter(|| {
                let mut decoder = create_vec_decoder(METRICS_SCHEMA, rows);
                decoder.decode(&data).unwrap();
                black_box(decoder.flush().unwrap().unwrap())
            })
        });

        // Primitives schema (with SOE prefix)
        let data = gen_primitives_data_soe(rows);

        group.bench_function(BenchmarkId::new("primitives", rows), |b| {
            b.iter(|| {
                let mut decoder = create_vec_decoder(PRIMITIVES_SCHEMA, rows);
                decoder.decode(&data).unwrap();
                black_box(decoder.flush().unwrap().unwrap())
            })
        });

        // Wide schema (with SOE prefix)
        let data = gen_wide_data_soe(rows);

        group.bench_function(BenchmarkId::new("wide_20col", rows), |b| {
            b.iter(|| {
                let mut decoder = create_vec_decoder(WIDE_SCHEMA, rows);
                decoder.decode(&data).unwrap();
                black_box(decoder.flush().unwrap().unwrap())
            })
        });
    }
    group.finish();
}

/// Direct comparison benchmark
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pooled_vs_Vec_Comparison");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(50);

    let rows = 50000;
    let data_raw = gen_metrics_data(rows);
    let data_soe = gen_metrics_data_soe(rows);
    let schema = create_metrics_schema();
    let layout = create_flat_layout(&schema, rows);

    group.throughput(Throughput::Elements(rows as u64));

    // Pooled (pre-allocated) - buffer allocated ONCE outside benchmark
    let buffer = vec![0u8; layout.total_size];
    let provider = PooledBufferProvider::new(buffer, layout.clone());
    let mut pooled_decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

    group.bench_function("Pooled_metrics_50k", |b| {
        b.iter(|| {
            pooled_decoder.decode(&data_raw, rows).unwrap();
            black_box(pooled_decoder.flush().unwrap())
        })
    });

    // Vec-based - decoder created each iteration (this is what Vec approach does)
    group.bench_function("Vec_metrics_50k", |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(METRICS_SCHEMA, rows);
            decoder.decode(&data_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    group.finish();
}

criterion_group! {
    name = pooled_flat;
    config = Criterion::default().configure_from_args();
    targets = bench_pooled_decoder, bench_vec_decoder, bench_comparison
}
criterion_main!(pooled_flat);
