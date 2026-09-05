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

//! Separate microbenchmarks, NOT an end-to-end speedup comparison.
//! `jit_raw_kernel` stops at decoded buffers; `ocf_reader_including_schema`
//! includes header parsing and Arrow assembly. Compare each series against its
//! own baseline only. The equivalent production-pipeline comparison lives in
//! ingestion-agent/benches/jit_avro_ingestion_bench.rs.

use apache_avro::types::Value;
use apache_avro::{Decimal, Schema as ApacheSchema, Writer, to_avro_datum};
use arrow_avro::reader::ReaderBuilder;
use arrow_avro::reader::jit_decoder::{JitDecoder, OutputBuffer};
use arrow_schema::{DataType, Field, Fields, Schema};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;

// === Schema Definitions ===

// Flat schema: all fixed-size primitives
const FLAT_SCHEMA: &str = r#"{
    "type": "record",
    "name": "FlatRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "value", "type": "double"},
        {"name": "count", "type": "int"},
        {"name": "ratio", "type": "float"}
    ]
}"#;

// Boolean schema
const BOOLEAN_SCHEMA: &str = r#"{
    "type": "record",
    "name": "BooleanRecord",
    "fields": [
        {"name": "flag", "type": "boolean"}
    ]
}"#;

// String schema
const STRING_SCHEMA: &str = r#"{
    "type": "record",
    "name": "StringRecord",
    "fields": [
        {"name": "name", "type": "string"}
    ]
}"#;

// Bytes schema
const BYTES_SCHEMA: &str = r#"{
    "type": "record",
    "name": "BytesRecord",
    "fields": [
        {"name": "data", "type": "bytes"}
    ]
}"#;

// Fixed schema
const FIXED_SCHEMA: &str = r#"{
    "type": "record",
    "name": "FixedRecord",
    "fields": [
        {"name": "uuid", "type": {"type": "fixed", "name": "uuid_t", "size": 16}}
    ]
}"#;

// Enum schema
const ENUM_SCHEMA: &str = r#"{
    "type": "record",
    "name": "EnumRecord",
    "fields": [
        {"name": "status", "type": {"type": "enum", "name": "Status", "symbols": ["PENDING", "ACTIVE", "COMPLETED", "FAILED"]}}
    ]
}"#;

// Array (List) schema
const ARRAY_SCHEMA: &str = r#"{
    "type": "record",
    "name": "ArrayRecord",
    "fields": [
        {"name": "values", "type": {"type": "array", "items": "long"}}
    ]
}"#;

// Map schema

// Nested Record (Struct) schema
const STRUCT_SCHEMA: &str = r#"{
    "type": "record",
    "name": "StructRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "nested", "type": {
            "type": "record",
            "name": "NestedRecord",
            "fields": [
                {"name": "x", "type": "int"},
                {"name": "y", "type": "int"}
            ]
        }}
    ]
}"#;

// Union schema (nullable long)

// Multi-int schema for JIT comparison
const MULTI_INT_SCHEMA: &str = r#"{
    "type": "record",
    "name": "MultiIntRecord",
    "fields": [
        {"name": "a", "type": "long"},
        {"name": "b", "type": "long"},
        {"name": "c", "type": "long"},
        {"name": "d", "type": "long"}
    ]
}"#;

// === Arrow Schema Definitions ===

fn flat_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("count", DataType::Int32, false),
        Field::new("ratio", DataType::Float32, false),
    ]))
}

fn boolean_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "flag",
        DataType::Boolean,
        false,
    )]))
}

fn string_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]))
}

fn bytes_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::Binary,
        false,
    )]))
}

fn fixed_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "uuid",
        DataType::FixedSizeBinary(16),
        false,
    )]))
}

fn enum_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "status",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        false,
    )]))
}

fn array_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "values",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
        false,
    )]))
}

fn struct_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "nested",
            DataType::Struct(Fields::from(vec![
                Field::new("x", DataType::Int32, false),
                Field::new("y", DataType::Int32, false),
            ])),
            false,
        ),
    ]))
}

fn multi_int_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Int64, false),
        Field::new("d", DataType::Int64, false),
    ]))
}

// === Data Generation ===

/// Holds both container format (for ReaderBuilder) and raw datum (for JIT)
struct TestData {
    container: Vec<u8>,
    raw_datum: Vec<u8>,
}

fn gen_flat_data(n: usize) -> TestData {
    let schema = ApacheSchema::parse_str(FLAT_SCHEMA).expect("invalid schema");

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("count".into(), Value::Int((i % 1000) as i32)),
            ("ratio".into(), Value::Float((i as f32) / 100.0)),
        ]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("count".into(), Value::Int((i % 1000) as i32)),
            ("ratio".into(), Value::Float((i as f32) / 100.0)),
        ]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

fn gen_boolean_data(n: usize) -> TestData {
    let schema = ApacheSchema::parse_str(BOOLEAN_SCHEMA).expect("invalid schema");

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let record = Value::Record(vec![("flag".into(), Value::Boolean(i % 2 == 0))]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![("flag".into(), Value::Boolean(i % 2 == 0))]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

fn gen_string_data(n: usize, avg_len: usize) -> TestData {
    let schema = ApacheSchema::parse_str(STRING_SCHEMA).expect("invalid schema");

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let s: String = format!("value_{:0width$}", i, width = avg_len);
        let record = Value::Record(vec![("name".into(), Value::String(s))]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let s: String = format!("value_{:0width$}", i, width = avg_len);
        let record = Value::Record(vec![("name".into(), Value::String(s))]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

fn gen_bytes_data(n: usize, avg_len: usize) -> TestData {
    let schema = ApacheSchema::parse_str(BYTES_SCHEMA).expect("invalid schema");

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let data: Vec<u8> = (0..avg_len).map(|j| ((i + j) % 256) as u8).collect();
        let record = Value::Record(vec![("data".into(), Value::Bytes(data))]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let data: Vec<u8> = (0..avg_len).map(|j| ((i + j) % 256) as u8).collect();
        let record = Value::Record(vec![("data".into(), Value::Bytes(data))]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

fn gen_fixed_data(n: usize) -> TestData {
    let schema = ApacheSchema::parse_str(FIXED_SCHEMA).expect("invalid schema");

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let mut uuid = [0u8; 16];
        for (j, b) in uuid.iter_mut().enumerate() {
            *b = ((i + j) % 256) as u8;
        }
        let record = Value::Record(vec![("uuid".into(), Value::Fixed(16, uuid.to_vec()))]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let mut uuid = [0u8; 16];
        for (j, b) in uuid.iter_mut().enumerate() {
            *b = ((i + j) % 256) as u8;
        }
        let record = Value::Record(vec![("uuid".into(), Value::Fixed(16, uuid.to_vec()))]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

fn gen_enum_data(n: usize) -> TestData {
    let schema = ApacheSchema::parse_str(ENUM_SCHEMA).expect("invalid schema");
    let symbols = ["PENDING", "ACTIVE", "COMPLETED", "FAILED"];

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let symbol = symbols[i % 4];
        let record = Value::Record(vec![(
            "status".into(),
            Value::Enum(i as u32 % 4, symbol.to_string()),
        )]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let symbol = symbols[i % 4];
        let record = Value::Record(vec![(
            "status".into(),
            Value::Enum(i as u32 % 4, symbol.to_string()),
        )]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

fn gen_array_data(n: usize, avg_len: usize) -> TestData {
    let schema = ApacheSchema::parse_str(ARRAY_SCHEMA).expect("invalid schema");

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let values: Vec<Value> = (0..avg_len).map(|j| Value::Long((i + j) as i64)).collect();
        let record = Value::Record(vec![("values".into(), Value::Array(values))]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let values: Vec<Value> = (0..avg_len).map(|j| Value::Long((i + j) as i64)).collect();
        let record = Value::Record(vec![("values".into(), Value::Array(values))]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

fn gen_struct_data(n: usize) -> TestData {
    let schema = ApacheSchema::parse_str(STRUCT_SCHEMA).expect("invalid schema");

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let nested = Value::Record(vec![
            ("x".into(), Value::Int(i as i32)),
            ("y".into(), Value::Int((i * 2) as i32)),
        ]);
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("nested".into(), nested),
        ]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let nested = Value::Record(vec![
            ("x".into(), Value::Int(i as i32)),
            ("y".into(), Value::Int((i * 2) as i32)),
        ]);
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("nested".into(), nested),
        ]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

fn gen_multi_int_data(n: usize) -> TestData {
    let schema = ApacheSchema::parse_str(MULTI_INT_SCHEMA).expect("invalid schema");

    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        let record = Value::Record(vec![
            ("a".into(), Value::Long(i as i64)),
            ("b".into(), Value::Long((i * 2) as i64)),
            ("c".into(), Value::Long((i * 3) as i64)),
            ("d".into(), Value::Long((i * 4) as i64)),
        ]);
        writer.append(record).expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");

    let mut raw_datum = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("a".into(), Value::Long(i as i64)),
            ("b".into(), Value::Long((i * 2) as i64)),
            ("c".into(), Value::Long((i * 3) as i64)),
            ("d".into(), Value::Long((i * 4) as i64)),
        ]);
        raw_datum.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }

    TestData {
        container,
        raw_datum,
    }
}

// === Benchmark Helper ===

fn run_apache_benchmark(data: &[u8]) -> usize {
    let reader = ReaderBuilder::new()
        .build(std::io::Cursor::new(data))
        .expect("reader");
    let mut total_rows = 0;
    for batch_result in reader {
        let batch = batch_result.expect("batch");
        total_rows += batch.num_rows();
        black_box(&batch);
    }
    total_rows
}

// === Benchmarks ===

fn bench_flat_schema(c: &mut Criterion) {
    let arrow_schema = flat_arrow_schema();
    let row_counts = [1000, 2000, 10000];

    let mut group = c.benchmark_group("kernel_and_ocf_flat");

    for &row_count in &row_counts {
        let test_data = gen_flat_data(row_count);
        group.throughput(Throughput::Elements(row_count as u64));

        group.bench_with_input(
            BenchmarkId::new("ocf_reader_including_schema", row_count),
            &row_count,
            |b, _| {
                b.iter(|| run_apache_benchmark(&test_data.container));
            },
        );

        let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
        group.bench_with_input(
            BenchmarkId::new("jit_raw_kernel", row_count),
            &row_count,
            |b, _| {
                b.iter(|| {
                    let mut output0 = vec![0u8; row_count * 8];
                    let mut output1 = vec![0u8; row_count * 8];
                    let mut output2 = vec![0u8; row_count * 4];
                    let mut output3 = vec![0u8; row_count * 4];
                    let buffers = vec![
                        OutputBuffer::new(output0.as_mut_ptr(), output0.len()),
                        OutputBuffer::new(output1.as_mut_ptr(), output1.len()),
                        OutputBuffer::new(output2.as_mut_ptr(), output2.len()),
                        OutputBuffer::new(output3.as_mut_ptr(), output3.len()),
                    ];
                    let result = unsafe {
                        jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count)
                    }
                    .expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_boolean(c: &mut Criterion) {
    let arrow_schema = boolean_arrow_schema();
    let row_counts = [1000, 2000, 10000];

    let mut group = c.benchmark_group("kernel_and_ocf_boolean");

    for &row_count in &row_counts {
        let test_data = gen_boolean_data(row_count);
        group.throughput(Throughput::Elements(row_count as u64));

        group.bench_with_input(
            BenchmarkId::new("ocf_reader_including_schema", row_count),
            &row_count,
            |b, _| {
                b.iter(|| run_apache_benchmark(&test_data.container));
            },
        );

        let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
        group.bench_with_input(
            BenchmarkId::new("jit_raw_kernel", row_count),
            &row_count,
            |b, _| {
                b.iter(|| {
                    // JIT stores booleans as 1 byte per value
                    let mut output = vec![0u8; row_count];
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    let result = unsafe {
                        jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count)
                    }
                    .expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_string(c: &mut Criterion) {
    let arrow_schema = string_arrow_schema();
    let row_count = 10000;
    let avg_len = 20;

    let mut group = c.benchmark_group("kernel_and_ocf_string");
    let test_data = gen_string_data(row_count, avg_len);
    group.throughput(Throughput::Elements(row_count as u64));

    group.bench_function("ocf_reader_including_schema", |b| {
        b.iter(|| run_apache_benchmark(&test_data.container));
    });

    let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
    group.bench_function("jit_raw_kernel", |b| {
        b.iter(|| {
            // String needs 2 buffers: offsets (i32) + data
            let mut offsets = vec![0u8; (row_count + 1) * 4];
            let mut data = vec![0u8; row_count * (avg_len + 20)];
            let buffers = vec![
                OutputBuffer::new(offsets.as_mut_ptr(), offsets.len()),
                OutputBuffer::new(data.as_mut_ptr(), data.len()),
            ];
            let result =
                unsafe { jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count) }
                    .expect("decode");
            black_box(result);
        });
    });

    group.finish();
}

fn bench_bytes(c: &mut Criterion) {
    let arrow_schema = bytes_arrow_schema();
    let row_count = 10000;
    let avg_len = 50;

    let mut group = c.benchmark_group("kernel_and_ocf_bytes");
    let test_data = gen_bytes_data(row_count, avg_len);
    group.throughput(Throughput::Elements(row_count as u64));

    group.bench_function("ocf_reader_including_schema", |b| {
        b.iter(|| run_apache_benchmark(&test_data.container));
    });

    let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
    group.bench_function("jit_raw_kernel", |b| {
        b.iter(|| {
            // Bytes needs 2 buffers: offsets (i32) + data
            let mut offsets = vec![0u8; (row_count + 1) * 4];
            let mut data = vec![0u8; row_count * (avg_len + 20)];
            let buffers = vec![
                OutputBuffer::new(offsets.as_mut_ptr(), offsets.len()),
                OutputBuffer::new(data.as_mut_ptr(), data.len()),
            ];
            let result =
                unsafe { jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count) }
                    .expect("decode");
            black_box(result);
        });
    });

    group.finish();
}

fn bench_fixed(c: &mut Criterion) {
    let arrow_schema = fixed_arrow_schema();
    let row_counts = [1000, 2000, 10000];

    let mut group = c.benchmark_group("kernel_and_ocf_fixed");

    for &row_count in &row_counts {
        let test_data = gen_fixed_data(row_count);
        group.throughput(Throughput::Elements(row_count as u64));

        group.bench_with_input(
            BenchmarkId::new("ocf_reader_including_schema", row_count),
            &row_count,
            |b, _| {
                b.iter(|| run_apache_benchmark(&test_data.container));
            },
        );

        let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
        group.bench_with_input(
            BenchmarkId::new("jit_raw_kernel", row_count),
            &row_count,
            |b, _| {
                b.iter(|| {
                    let mut output = vec![0u8; row_count * 16];
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    let result = unsafe {
                        jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count)
                    }
                    .expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_enum(c: &mut Criterion) {
    let arrow_schema = enum_arrow_schema();
    let row_counts = [1000, 2000, 10000];

    let mut group = c.benchmark_group("kernel_and_ocf_enum");

    for &row_count in &row_counts {
        let test_data = gen_enum_data(row_count);
        group.throughput(Throughput::Elements(row_count as u64));

        group.bench_with_input(
            BenchmarkId::new("ocf_reader_including_schema", row_count),
            &row_count,
            |b, _| {
                b.iter(|| run_apache_benchmark(&test_data.container));
            },
        );

        let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
        group.bench_with_input(
            BenchmarkId::new("jit_raw_kernel", row_count),
            &row_count,
            |b, _| {
                b.iter(|| {
                    let mut output = vec![0u8; row_count * 4]; // i32 indices
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    let result = unsafe {
                        jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count)
                    }
                    .expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

const DECIMAL_SCHEMA: &str = r#"{"type":"record","name":"DecRec","fields":[{"name":"d","type":{"type":"bytes","logicalType":"decimal","precision":18,"scale":4}}]}"#;

fn decimal_arrow_schema() -> Arc<Schema> {
    let mut meta = std::collections::HashMap::new();
    meta.insert("avro.decimal.encoding".to_string(), "bytes".to_string());
    Arc::new(Schema::new(vec![
        Field::new("d", DataType::Decimal128(18, 4), false).with_metadata(meta),
    ]))
}

fn gen_decimal_data(n: usize) -> TestData {
    let schema = ApacheSchema::parse_str(DECIMAL_SCHEMA).expect("invalid schema");
    let val = |i: usize| {
        Value::Decimal(Decimal::from(
            ((i as i128) * 100 - 5000).to_be_bytes().to_vec(),
        ))
    };
    let mut writer = Writer::new(&schema, Vec::new());
    for i in 0..n {
        writer
            .append(Value::Record(vec![("d".into(), val(i))]))
            .expect("append failed");
    }
    let container = writer.into_inner().expect("flush failed");
    let mut raw_datum = Vec::new();
    for i in 0..n {
        raw_datum.extend_from_slice(
            &to_avro_datum(&schema, Value::Record(vec![("d".into(), val(i))]))
                .expect("encode failed"),
        );
    }
    TestData {
        container,
        raw_datum,
    }
}

fn bench_decimal(c: &mut Criterion) {
    let arrow_schema = decimal_arrow_schema();
    let row_counts = [1000, 2000, 10000];
    let mut group = c.benchmark_group("kernel_and_ocf_decimal");
    for &row_count in &row_counts {
        let test_data = gen_decimal_data(row_count);
        group.throughput(Throughput::Elements(row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("ocf_reader_including_schema", row_count),
            &row_count,
            |b, _| {
                b.iter(|| run_apache_benchmark(&test_data.container));
            },
        );
        let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
        group.bench_with_input(
            BenchmarkId::new("jit_raw_kernel", row_count),
            &row_count,
            |b, _| {
                b.iter(|| {
                    let mut output = vec![0u8; row_count * 16]; // Decimal128 = 16 bytes
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    let result = unsafe {
                        jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count)
                    }
                    .expect("decode");
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

fn bench_array(c: &mut Criterion) {
    let arrow_schema = array_arrow_schema();
    let row_count = 10000;
    let avg_len = 5;

    let mut group = c.benchmark_group("kernel_and_ocf_array");
    let test_data = gen_array_data(row_count, avg_len);
    group.throughput(Throughput::Elements(row_count as u64));

    group.bench_function("ocf_reader_including_schema", |b| {
        b.iter(|| run_apache_benchmark(&test_data.container));
    });

    let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
    group.bench_function("jit_raw_kernel", |b| {
        b.iter(|| {
            // List<Int64> needs 2 buffers: offsets (i32) + values (i64)
            let mut offsets = vec![0u8; (row_count + 1) * 4]; // i32 offsets
            let mut values = vec![0u8; row_count * avg_len * 8]; // i64 values
            let buffers = vec![
                OutputBuffer::new(offsets.as_mut_ptr(), offsets.len()),
                OutputBuffer::new(values.as_mut_ptr(), values.len()),
            ];
            let result =
                unsafe { jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count) }
                    .expect("decode");
            black_box(result);
        });
    });

    group.finish();
}

fn bench_struct(c: &mut Criterion) {
    let arrow_schema = struct_arrow_schema();
    let row_counts = [1000, 2000, 10000];

    let mut group = c.benchmark_group("kernel_and_ocf_struct");

    for &row_count in &row_counts {
        let test_data = gen_struct_data(row_count);
        group.throughput(Throughput::Elements(row_count as u64));

        group.bench_with_input(
            BenchmarkId::new("ocf_reader_including_schema", row_count),
            &row_count,
            |b, _| {
                b.iter(|| run_apache_benchmark(&test_data.container));
            },
        );

        let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
        group.bench_with_input(
            BenchmarkId::new("jit_raw_kernel", row_count),
            &row_count,
            |b, _| {
                b.iter(|| {
                    // JIT uses top-level field count: id (i64) + nested (struct flattened to single buffer)
                    // nested contains x (i32) + y (i32) = 8 bytes per row
                    let mut output0 = vec![0u8; row_count * 8]; // id
                    let mut output1 = vec![0u8; row_count * 8]; // nested (x + y concatenated)
                    let buffers = vec![
                        OutputBuffer::new(output0.as_mut_ptr(), output0.len()),
                        OutputBuffer::new(output1.as_mut_ptr(), output1.len()),
                    ];
                    let result = unsafe {
                        jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count)
                    }
                    .expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_multi_int(c: &mut Criterion) {
    let arrow_schema = multi_int_arrow_schema();
    let row_counts = [1000, 2000, 10000];

    let mut group = c.benchmark_group("kernel_and_ocf_multi_int");

    for &row_count in &row_counts {
        let test_data = gen_multi_int_data(row_count);
        group.throughput(Throughput::Elements(row_count as u64));

        group.bench_with_input(
            BenchmarkId::new("ocf_reader_including_schema", row_count),
            &row_count,
            |b, _| {
                b.iter(|| run_apache_benchmark(&test_data.container));
            },
        );

        let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
        group.bench_with_input(
            BenchmarkId::new("jit_raw_kernel", row_count),
            &row_count,
            |b, _| {
                b.iter(|| {
                    let mut output0 = vec![0u8; row_count * 8];
                    let mut output1 = vec![0u8; row_count * 8];
                    let mut output2 = vec![0u8; row_count * 8];
                    let mut output3 = vec![0u8; row_count * 8];
                    let buffers = vec![
                        OutputBuffer::new(output0.as_mut_ptr(), output0.len()),
                        OutputBuffer::new(output1.as_mut_ptr(), output1.len()),
                        OutputBuffer::new(output2.as_mut_ptr(), output2.len()),
                        OutputBuffer::new(output3.as_mut_ptr(), output3.len()),
                    ];
                    let result = unsafe {
                        jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count)
                    }
                    .expect("decode");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_throughput_1m(c: &mut Criterion) {
    let arrow_schema = multi_int_arrow_schema();
    let row_count = 1_000_000;

    let test_data = gen_multi_int_data(row_count);

    let mut group = c.benchmark_group("kernel_and_ocf_1M_rows");
    group.throughput(Throughput::Elements(row_count as u64));
    group.sample_size(20);

    group.bench_function("ocf_reader_including_schema", |b| {
        b.iter(|| run_apache_benchmark(&test_data.container));
    });

    let jit_decoder = JitDecoder::compile(&arrow_schema).expect("JIT compile");
    group.bench_function("jit_raw_kernel", |b| {
        b.iter(|| {
            let mut output0 = vec![0u8; row_count * 8];
            let mut output1 = vec![0u8; row_count * 8];
            let mut output2 = vec![0u8; row_count * 8];
            let mut output3 = vec![0u8; row_count * 8];
            let buffers = vec![
                OutputBuffer::new(output0.as_mut_ptr(), output0.len()),
                OutputBuffer::new(output1.as_mut_ptr(), output1.len()),
                OutputBuffer::new(output2.as_mut_ptr(), output2.len()),
                OutputBuffer::new(output3.as_mut_ptr(), output3.len()),
            ];
            let result =
                unsafe { jit_decoder.decode_checked(&test_data.raw_datum, &buffers, row_count) }
                    .expect("decode");
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_flat_schema,
    bench_boolean,
    bench_string,
    bench_bytes,
    bench_fixed,
    bench_enum,
    bench_decimal,
    bench_array,
    bench_struct,
    bench_multi_int,
    bench_throughput_1m
);
criterion_main!(benches);
