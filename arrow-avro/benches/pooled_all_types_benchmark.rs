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

//! Comprehensive benchmark comparing Vec-based vs Pre-allocated decoding
//! for ALL supported Avro types (flat, strings, mixed, dictionary, all-combined)

use apache_avro::types::Value;
use apache_avro::{to_avro_datum, Schema as ApacheSchema};
use arrow_avro::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};
use arrow_avro::reader::pooled_decoder::PooledRecordDecoder;
use arrow_avro::reader::ReaderBuilder;
use arrow_avro::schema::{AvroSchema, Fingerprint, FingerprintAlgorithm, SchemaStore, SINGLE_OBJECT_MAGIC};
use arrow_schema::{DataType, Field, Schema, UnionFields, UnionMode};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

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

// Strings-only schema
const STRINGS_SCHEMA: &str = r#"{
    "type": "record",
    "name": "StringsRecord",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
        {"name": "address", "type": "string"}
    ]
}"#;

// Mixed schema: fixed + variable-length
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

// Dictionary-only schema (Avro enums)
const DICT_SCHEMA: &str = r#"{
    "type": "record",
    "name": "DictRecord",
    "fields": [
        {"name": "status", "type": {"type": "enum", "name": "Status", "symbols": ["ACTIVE", "INACTIVE", "PENDING", "ARCHIVED"]}},
        {"name": "category", "type": {"type": "enum", "name": "Category", "symbols": ["A", "B", "C", "D", "E"]}}
    ]
}"#;

// Boolean and Binary schema
const BOOL_BINARY_SCHEMA: &str = r#"{
    "type": "record",
    "name": "BoolBinaryRecord",
    "fields": [
        {"name": "active", "type": "boolean"},
        {"name": "verified", "type": "boolean"},
        {"name": "data", "type": "bytes"},
        {"name": "checksum", "type": {"type": "fixed", "name": "Checksum", "size": 16}}
    ]
}"#;

// List schema (arrays)
const LIST_SCHEMA: &str = r#"{
    "type": "record",
    "name": "ListRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "tags", "type": {"type": "array", "items": "string"}},
        {"name": "scores", "type": {"type": "array", "items": "int"}}
    ]
}"#;

// Map schema
const MAP_SCHEMA: &str = r#"{
    "type": "record",
    "name": "MapRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "metadata", "type": {"type": "map", "values": "string"}},
        {"name": "counts", "type": {"type": "map", "values": "int"}}
    ]
}"#;

// Union schema (nullable types)
const UNION_SCHEMA: &str = r#"{
    "type": "record",
    "name": "UnionRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "optional_name", "type": ["null", "string"]},
        {"name": "optional_value", "type": ["null", "double"]},
        {"name": "optional_count", "type": ["null", "int"]}
    ]
}"#;

// Nested struct schema
const STRUCT_SCHEMA: &str = r#"{
    "type": "record",
    "name": "StructRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "address", "type": {
            "type": "record",
            "name": "Address",
            "fields": [
                {"name": "street", "type": "string"},
                {"name": "city", "type": "string"},
                {"name": "zip", "type": "int"}
            ]
        }},
        {"name": "name", "type": "string"}
    ]
}"#;

// Comprehensive schema: all types combined
const ALL_TYPES_SCHEMA: &str = r#"{
    "type": "record",
    "name": "AllTypesRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "value", "type": "double"},
        {"name": "status", "type": {"type": "enum", "name": "Status", "symbols": ["ACTIVE", "INACTIVE", "PENDING"]}},
        {"name": "description", "type": "string"},
        {"name": "count", "type": "int"},
        {"name": "category", "type": {"type": "enum", "name": "Category", "symbols": ["A", "B", "C", "D"]}}
    ]
}"#;

// Ultimate comprehensive schema with ALL supported types
const COMPREHENSIVE_SCHEMA: &str = r#"{
    "type": "record",
    "name": "ComprehensiveRecord",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "active", "type": "boolean"},
        {"name": "name", "type": "string"},
        {"name": "value", "type": "double"},
        {"name": "ratio", "type": "float"},
        {"name": "count", "type": "int"},
        {"name": "data", "type": "bytes"},
        {"name": "checksum", "type": {"type": "fixed", "name": "Checksum", "size": 16}},
        {"name": "status", "type": {"type": "enum", "name": "Status", "symbols": ["ACTIVE", "INACTIVE", "PENDING"]}},
        {"name": "tags", "type": {"type": "array", "items": "string"}},
        {"name": "scores", "type": {"type": "array", "items": "int"}},
        {"name": "metadata", "type": {"type": "map", "values": "string"}},
        {"name": "optional_note", "type": ["null", "string"]},
        {"name": "address", "type": {
            "type": "record",
            "name": "Address",
            "fields": [
                {"name": "street", "type": "string"},
                {"name": "city", "type": "string"},
                {"name": "zip", "type": "int"}
            ]
        }}
    ]
}"#;

// === Helper Functions ===

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

fn gen_string(idx: usize, base: &str, avg_len: usize) -> String {
    let variation = (idx % 10) as i32 - 5;
    let len = (avg_len as i32 + variation).max(1) as usize;
    format!("{}{:0>width$}", base, idx % 10000, width = len.saturating_sub(base.len()))
}

// === Data Generation ===

fn gen_flat_data(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(FLAT_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("count".into(), Value::Int((i % 1000) as i32)),
            ("ratio".into(), Value::Float((i as f32) / 100.0)),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_flat_data_soe(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(FLAT_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(FLAT_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("count".into(), Value::Int((i % 1000) as i32)),
            ("ratio".into(), Value::Float((i as f32) / 100.0)),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_strings_data(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(STRINGS_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
            ("email".into(), Value::String(gen_string(i, "email_", avg_len))),
            ("address".into(), Value::String(gen_string(i, "addr_", avg_len))),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_strings_data_soe(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(STRINGS_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(STRINGS_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
            ("email".into(), Value::String(gen_string(i, "email_", avg_len))),
            ("address".into(), Value::String(gen_string(i, "addr_", avg_len))),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_mixed_data(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(MIXED_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("description".into(), Value::String(gen_string(i, "desc_", avg_len * 2))),
            ("count".into(), Value::Int((i % 1000) as i32)),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_mixed_data_soe(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(MIXED_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(MIXED_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("description".into(), Value::String(gen_string(i, "desc_", avg_len * 2))),
            ("count".into(), Value::Int((i % 1000) as i32)),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_dict_data(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(DICT_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("status".into(), Value::Enum((i % 4) as u32, ["ACTIVE", "INACTIVE", "PENDING", "ARCHIVED"][i % 4].to_string())),
            ("category".into(), Value::Enum((i % 5) as u32, ["A", "B", "C", "D", "E"][i % 5].to_string())),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_dict_data_soe(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(DICT_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(DICT_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("status".into(), Value::Enum((i % 4) as u32, ["ACTIVE", "INACTIVE", "PENDING", "ARCHIVED"][i % 4].to_string())),
            ("category".into(), Value::Enum((i % 5) as u32, ["A", "B", "C", "D", "E"][i % 5].to_string())),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_all_types_data(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(ALL_TYPES_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("status".into(), Value::Enum((i % 3) as u32, ["ACTIVE", "INACTIVE", "PENDING"][i % 3].to_string())),
            ("description".into(), Value::String(gen_string(i, "desc_", avg_len * 2))),
            ("count".into(), Value::Int((i % 1000) as i32)),
            ("category".into(), Value::Enum((i % 4) as u32, ["A", "B", "C", "D"][i % 4].to_string())),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_all_types_data_soe(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(ALL_TYPES_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(ALL_TYPES_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("status".into(), Value::Enum((i % 3) as u32, ["ACTIVE", "INACTIVE", "PENDING"][i % 3].to_string())),
            ("description".into(), Value::String(gen_string(i, "desc_", avg_len * 2))),
            ("count".into(), Value::Int((i % 1000) as i32)),
            ("category".into(), Value::Enum((i % 4) as u32, ["A", "B", "C", "D"][i % 4].to_string())),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

// === NEW TYPE DATA GENERATORS ===

fn gen_bool_binary_data(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(BOOL_BINARY_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let data_bytes: Vec<u8> = (0..20).map(|j| ((i + j) % 256) as u8).collect();
        let fixed_bytes: [u8; 16] = std::array::from_fn(|j| ((i * 7 + j) % 256) as u8);
        let record = Value::Record(vec![
            ("active".into(), Value::Boolean(i % 2 == 0)),
            ("verified".into(), Value::Boolean(i % 3 == 0)),
            ("data".into(), Value::Bytes(data_bytes)),
            ("checksum".into(), Value::Fixed(16, fixed_bytes.to_vec())),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_bool_binary_data_soe(n: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(BOOL_BINARY_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(BOOL_BINARY_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let data_bytes: Vec<u8> = (0..20).map(|j| ((i + j) % 256) as u8).collect();
        let fixed_bytes: [u8; 16] = std::array::from_fn(|j| ((i * 7 + j) % 256) as u8);
        let record = Value::Record(vec![
            ("active".into(), Value::Boolean(i % 2 == 0)),
            ("verified".into(), Value::Boolean(i % 3 == 0)),
            ("data".into(), Value::Bytes(data_bytes)),
            ("checksum".into(), Value::Fixed(16, fixed_bytes.to_vec())),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_list_data(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(LIST_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let num_tags = (i % 5) + 1; // 1-5 tags
        let tags: Vec<Value> = (0..num_tags)
            .map(|j| Value::String(gen_string(i * 10 + j, "tag_", avg_len)))
            .collect();
        let num_scores = (i % 4) + 2; // 2-5 scores
        let scores: Vec<Value> = (0..num_scores)
            .map(|j| Value::Int(((i + j) % 100) as i32))
            .collect();
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("tags".into(), Value::Array(tags)),
            ("scores".into(), Value::Array(scores)),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_list_data_soe(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(LIST_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(LIST_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let num_tags = (i % 5) + 1;
        let tags: Vec<Value> = (0..num_tags)
            .map(|j| Value::String(gen_string(i * 10 + j, "tag_", avg_len)))
            .collect();
        let num_scores = (i % 4) + 2;
        let scores: Vec<Value> = (0..num_scores)
            .map(|j| Value::Int(((i + j) % 100) as i32))
            .collect();
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("tags".into(), Value::Array(tags)),
            ("scores".into(), Value::Array(scores)),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_map_data(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(MAP_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let num_entries = (i % 3) + 1; // 1-3 entries
        let metadata: std::collections::HashMap<String, Value> = (0..num_entries)
            .map(|j| (format!("key_{}", j), Value::String(gen_string(i + j, "val_", avg_len))))
            .collect();
        let counts: std::collections::HashMap<String, Value> = (0..num_entries)
            .map(|j| (format!("cnt_{}", j), Value::Int(((i + j) % 1000) as i32)))
            .collect();
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("metadata".into(), Value::Map(metadata)),
            ("counts".into(), Value::Map(counts)),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_map_data_soe(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(MAP_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(MAP_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let num_entries = (i % 3) + 1;
        let metadata: std::collections::HashMap<String, Value> = (0..num_entries)
            .map(|j| (format!("key_{}", j), Value::String(gen_string(i + j, "val_", avg_len))))
            .collect();
        let counts: std::collections::HashMap<String, Value> = (0..num_entries)
            .map(|j| (format!("cnt_{}", j), Value::Int(((i + j) % 1000) as i32)))
            .collect();
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("metadata".into(), Value::Map(metadata)),
            ("counts".into(), Value::Map(counts)),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_union_data(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(UNION_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        // Vary nullability pattern
        let optional_name = if i % 3 == 0 {
            Value::Union(0, Box::new(Value::Null))
        } else {
            Value::Union(1, Box::new(Value::String(gen_string(i, "name_", avg_len))))
        };
        let optional_value = if i % 4 == 0 {
            Value::Union(0, Box::new(Value::Null))
        } else {
            Value::Union(1, Box::new(Value::Double(i as f64 * 0.5)))
        };
        let optional_count = if i % 5 == 0 {
            Value::Union(0, Box::new(Value::Null))
        } else {
            Value::Union(1, Box::new(Value::Int((i % 1000) as i32)))
        };
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("optional_name".into(), optional_name),
            ("optional_value".into(), optional_value),
            ("optional_count".into(), optional_count),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_union_data_soe(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(UNION_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(UNION_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let optional_name = if i % 3 == 0 {
            Value::Union(0, Box::new(Value::Null))
        } else {
            Value::Union(1, Box::new(Value::String(gen_string(i, "name_", avg_len))))
        };
        let optional_value = if i % 4 == 0 {
            Value::Union(0, Box::new(Value::Null))
        } else {
            Value::Union(1, Box::new(Value::Double(i as f64 * 0.5)))
        };
        let optional_count = if i % 5 == 0 {
            Value::Union(0, Box::new(Value::Null))
        } else {
            Value::Union(1, Box::new(Value::Int((i % 1000) as i32)))
        };
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("optional_name".into(), optional_name),
            ("optional_value".into(), optional_value),
            ("optional_count".into(), optional_count),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_struct_data(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(STRUCT_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let address = Value::Record(vec![
            ("street".into(), Value::String(gen_string(i, "street_", avg_len))),
            ("city".into(), Value::String(gen_string(i, "city_", avg_len / 2))),
            ("zip".into(), Value::Int((10000 + (i % 90000)) as i32)),
        ]);
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("address".into(), address),
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_struct_data_soe(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(STRUCT_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(STRUCT_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let address = Value::Record(vec![
            ("street".into(), Value::String(gen_string(i, "street_", avg_len))),
            ("city".into(), Value::String(gen_string(i, "city_", avg_len / 2))),
            ("zip".into(), Value::Int((10000 + (i % 90000)) as i32)),
        ]);
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("address".into(), address),
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_comprehensive_data(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(COMPREHENSIVE_SCHEMA).expect("invalid schema");
    let mut out = Vec::new();
    for i in 0..n {
        let data_bytes: Vec<u8> = (0..20).map(|j| ((i + j) % 256) as u8).collect();
        let fixed_bytes: [u8; 16] = std::array::from_fn(|j| ((i * 7 + j) % 256) as u8);
        let num_tags = (i % 3) + 1;
        let tags: Vec<Value> = (0..num_tags)
            .map(|j| Value::String(gen_string(i * 10 + j, "tag_", avg_len / 2)))
            .collect();
        let num_scores = (i % 3) + 1;
        let scores: Vec<Value> = (0..num_scores)
            .map(|j| Value::Int(((i + j) % 100) as i32))
            .collect();
        let num_meta = (i % 2) + 1;
        let metadata: std::collections::HashMap<String, Value> = (0..num_meta)
            .map(|j| (format!("k{}", j), Value::String(gen_string(i + j, "v_", avg_len / 2))))
            .collect();
        let optional_note = if i % 4 == 0 {
            Value::Union(0, Box::new(Value::Null))
        } else {
            Value::Union(1, Box::new(Value::String(gen_string(i, "note_", avg_len))))
        };
        let address = Value::Record(vec![
            ("street".into(), Value::String(gen_string(i, "st_", avg_len))),
            ("city".into(), Value::String(gen_string(i, "city_", avg_len / 2))),
            ("zip".into(), Value::Int((10000 + (i % 90000)) as i32)),
        ]);
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("active".into(), Value::Boolean(i % 2 == 0)),
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("ratio".into(), Value::Float(i as f32 / 100.0)),
            ("count".into(), Value::Int((i % 1000) as i32)),
            ("data".into(), Value::Bytes(data_bytes)),
            ("checksum".into(), Value::Fixed(16, fixed_bytes.to_vec())),
            ("status".into(), Value::Enum((i % 3) as u32, ["ACTIVE", "INACTIVE", "PENDING"][i % 3].to_string())),
            ("tags".into(), Value::Array(tags)),
            ("scores".into(), Value::Array(scores)),
            ("metadata".into(), Value::Map(metadata)),
            ("optional_note".into(), optional_note),
            ("address".into(), address),
        ]);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

fn gen_comprehensive_data_soe(n: usize, avg_len: usize) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(COMPREHENSIVE_SCHEMA).expect("invalid schema");
    let prefix = make_soe_prefix(COMPREHENSIVE_SCHEMA);
    let mut out = Vec::new();
    for i in 0..n {
        let data_bytes: Vec<u8> = (0..20).map(|j| ((i + j) % 256) as u8).collect();
        let fixed_bytes: [u8; 16] = std::array::from_fn(|j| ((i * 7 + j) % 256) as u8);
        let num_tags = (i % 3) + 1;
        let tags: Vec<Value> = (0..num_tags)
            .map(|j| Value::String(gen_string(i * 10 + j, "tag_", avg_len / 2)))
            .collect();
        let num_scores = (i % 3) + 1;
        let scores: Vec<Value> = (0..num_scores)
            .map(|j| Value::Int(((i + j) % 100) as i32))
            .collect();
        let num_meta = (i % 2) + 1;
        let metadata: std::collections::HashMap<String, Value> = (0..num_meta)
            .map(|j| (format!("k{}", j), Value::String(gen_string(i + j, "v_", avg_len / 2))))
            .collect();
        let optional_note = if i % 4 == 0 {
            Value::Union(0, Box::new(Value::Null))
        } else {
            Value::Union(1, Box::new(Value::String(gen_string(i, "note_", avg_len))))
        };
        let address = Value::Record(vec![
            ("street".into(), Value::String(gen_string(i, "st_", avg_len))),
            ("city".into(), Value::String(gen_string(i, "city_", avg_len / 2))),
            ("zip".into(), Value::Int((10000 + (i % 90000)) as i32)),
        ]);
        let record = Value::Record(vec![
            ("id".into(), Value::Long(i as i64)),
            ("active".into(), Value::Boolean(i % 2 == 0)),
            ("name".into(), Value::String(gen_string(i, "name_", avg_len))),
            ("value".into(), Value::Double(i as f64 * 0.1)),
            ("ratio".into(), Value::Float(i as f32 / 100.0)),
            ("count".into(), Value::Int((i % 1000) as i32)),
            ("data".into(), Value::Bytes(data_bytes)),
            ("checksum".into(), Value::Fixed(16, fixed_bytes.to_vec())),
            ("status".into(), Value::Enum((i % 3) as u32, ["ACTIVE", "INACTIVE", "PENDING"][i % 3].to_string())),
            ("tags".into(), Value::Array(tags)),
            ("scores".into(), Value::Array(scores)),
            ("metadata".into(), Value::Map(metadata)),
            ("optional_note".into(), optional_note),
            ("address".into(), address),
        ]);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&to_avro_datum(&schema, record).expect("encode failed"));
    }
    out
}

// === Arrow Schema Creators ===

fn create_flat_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("count", DataType::Int32, false),
        Field::new("ratio", DataType::Float32, false),
    ]))
}

fn create_strings_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("email", DataType::Utf8, false),
        Field::new("address", DataType::Utf8, false),
    ]))
}

fn create_mixed_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("description", DataType::Utf8, false),
        Field::new("count", DataType::Int32, false),
    ]))
}

fn create_dict_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("status", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), false),
        Field::new("category", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), false),
    ]))
}

fn create_all_types_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("status", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), false),
        Field::new("description", DataType::Utf8, false),
        Field::new("count", DataType::Int32, false),
        Field::new("category", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), false),
    ]))
}

fn create_bool_binary_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("active", DataType::Boolean, false),
        Field::new("verified", DataType::Boolean, false),
        Field::new("data", DataType::Binary, false),
        Field::new("checksum", DataType::FixedSizeBinary(16), false),
    ]))
}

fn create_list_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("tags", DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))), false),
        Field::new("scores", DataType::List(Arc::new(Field::new("item", DataType::Int32, false))), false),
    ]))
}

fn create_map_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("metadata", DataType::Map(
            Arc::new(Field::new("entries", DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Utf8, true),
                ].into()
            ), false)),
            false
        ), false),
        Field::new("counts", DataType::Map(
            Arc::new(Field::new("entries", DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int32, true),
                ].into()
            ), false)),
            false
        ), false),
    ]))
}

fn create_union_schema() -> Arc<Schema> {
    // Avro unions like ["null", "string"] map to Arrow Union with Null + value type
    // Using Dense union mode for efficiency
    let union_string_fields = UnionFields::new(
        vec![0, 1],
        vec![
            Field::new("null", DataType::Null, true),
            Field::new("string", DataType::Utf8, false),
        ],
    );
    let union_double_fields = UnionFields::new(
        vec![0, 1],
        vec![
            Field::new("null", DataType::Null, true),
            Field::new("double", DataType::Float64, false),
        ],
    );
    let union_int_fields = UnionFields::new(
        vec![0, 1],
        vec![
            Field::new("null", DataType::Null, true),
            Field::new("int", DataType::Int32, false),
        ],
    );
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("optional_name", DataType::Union(union_string_fields, UnionMode::Dense), false),
        Field::new("optional_value", DataType::Union(union_double_fields, UnionMode::Dense), false),
        Field::new("optional_count", DataType::Union(union_int_fields, UnionMode::Dense), false),
    ]))
}

fn create_struct_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("address", DataType::Struct(
            vec![
                Field::new("street", DataType::Utf8, false),
                Field::new("city", DataType::Utf8, false),
                Field::new("zip", DataType::Int32, false),
            ].into()
        ), false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

fn create_comprehensive_schema() -> Arc<Schema> {
    // Union for optional_note: ["null", "string"]
    let optional_note_fields = UnionFields::new(
        vec![0, 1],
        vec![
            Field::new("null", DataType::Null, true),
            Field::new("string", DataType::Utf8, false),
        ],
    );
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("active", DataType::Boolean, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("ratio", DataType::Float32, false),
        Field::new("count", DataType::Int32, false),
        Field::new("data", DataType::Binary, false),
        Field::new("checksum", DataType::FixedSizeBinary(16), false),
        Field::new("status", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), false),
        Field::new("tags", DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))), false),
        Field::new("scores", DataType::List(Arc::new(Field::new("item", DataType::Int32, false))), false),
        Field::new("metadata", DataType::Map(
            Arc::new(Field::new("entries", DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Utf8, true),
                ].into()
            ), false)),
            false
        ), false),
        Field::new("optional_note", DataType::Union(optional_note_fields, UnionMode::Dense), false),
        Field::new("address", DataType::Struct(
            vec![
                Field::new("street", DataType::Utf8, false),
                Field::new("city", DataType::Utf8, false),
                Field::new("zip", DataType::Int32, false),
            ].into()
        ), false),
    ]))
}

// === Buffer Layout Creators ===

fn create_flat_layout(row_count: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // id: Int64 - 8 bytes
    let id_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: id_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_cap;

    // value: Float64 - 8 bytes
    let val_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: val_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += val_cap;

    // count: Int32 - 4 bytes
    let cnt_cap = row_count * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: cnt_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += cnt_cap;

    // ratio: Float32 - 4 bytes
    let ratio_cap = row_count * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: ratio_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += ratio_cap;

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_strings_layout(row_count: usize, avg_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    for _ in 0..3 {
        let data_cap = row_count * (avg_len + 10);
        let offsets_cap = (row_count + 1) * 4;
        columns.push(ColumnRegion {
            data_offset: offset,
            data_capacity: data_cap,
            offsets_offset: Some(offset + data_cap),
            offsets_capacity: Some(offsets_cap),
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: vec![],
        });
        offset += data_cap + offsets_cap;
    }

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_mixed_layout(row_count: usize, avg_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // id: Int64
    let id_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: id_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_cap;

    // name: String
    let name_data_cap = row_count * (avg_len + 10);
    let name_off_cap = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: name_data_cap,
        offsets_offset: Some(offset + name_data_cap),
        offsets_capacity: Some(name_off_cap),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += name_data_cap + name_off_cap;

    // value: Float64
    let val_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: val_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += val_cap;

    // description: String
    let desc_data_cap = row_count * (avg_len * 2 + 10);
    let desc_off_cap = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: desc_data_cap,
        offsets_offset: Some(offset + desc_data_cap),
        offsets_capacity: Some(desc_off_cap),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += desc_data_cap + desc_off_cap;

    // count: Int32
    let cnt_cap = row_count * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: cnt_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += cnt_cap;

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_dict_layout(row_count: usize) -> BufferLayout {
    // Dictionary columns just use Vec<i32> internally, no pre-allocated buffer needed
    // But we need minimal layout for initialization
    BufferLayout {
        columns: vec![
            ColumnRegion {
                data_offset: 0,
                data_capacity: 0,
                offsets_offset: None,
                offsets_capacity: None,
                null_bitmap_offset: None,
                null_bitmap_capacity: None,
                child_regions: vec![],
            },
            ColumnRegion {
                data_offset: 0,
                data_capacity: 0,
                offsets_offset: None,
                offsets_capacity: None,
                null_bitmap_offset: None,
                null_bitmap_capacity: None,
                child_regions: vec![],
            },
        ],
        total_size: 0,
        row_count,
    }
}

fn create_all_types_layout(row_count: usize, avg_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // id: Int64
    let id_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: id_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_cap;

    // name: String
    let name_data_cap = row_count * (avg_len + 10);
    let name_off_cap = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: name_data_cap,
        offsets_offset: Some(offset + name_data_cap),
        offsets_capacity: Some(name_off_cap),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += name_data_cap + name_off_cap;

    // value: Float64
    let val_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: val_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += val_cap;

    // status: Dictionary (uses Vec internally)
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });

    // description: String
    let desc_data_cap = row_count * (avg_len * 2 + 10);
    let desc_off_cap = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: desc_data_cap,
        offsets_offset: Some(offset + desc_data_cap),
        offsets_capacity: Some(desc_off_cap),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += desc_data_cap + desc_off_cap;

    // count: Int32
    let cnt_cap = row_count * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: cnt_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += cnt_cap;

    // category: Dictionary (uses Vec internally)
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_bool_binary_layout(row_count: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // active: Boolean - bit-packed (8 values per byte)
    let bool_cap = (row_count + 7) / 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: bool_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += bool_cap;

    // verified: Boolean - bit-packed
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: bool_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += bool_cap;

    // data: Binary (variable length ~20 bytes)
    let data_cap = row_count * 30; // Extra buffer
    let data_off_cap = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: data_cap,
        offsets_offset: Some(offset + data_cap),
        offsets_capacity: Some(data_off_cap),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += data_cap + data_off_cap;

    // checksum: FixedSizeBinary(16)
    let fixed_cap = row_count * 16;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: fixed_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += fixed_cap;

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_list_layout(row_count: usize, avg_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // id: Int64
    let id_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: id_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_cap;

    // tags: List<String> - avg 3 items per row
    let avg_items = 3;
    let total_items = row_count * avg_items;
    let list_off_cap = (row_count + 1) * 4;
    let item_data_cap = total_items * (avg_len + 10);
    let item_off_cap = (total_items + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0, // List itself has no data buffer
        offsets_offset: Some(offset),
        offsets_capacity: Some(list_off_cap),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![ColumnRegion {
            data_offset: offset + list_off_cap,
            data_capacity: item_data_cap,
            offsets_offset: Some(offset + list_off_cap + item_data_cap),
            offsets_capacity: Some(item_off_cap),
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: vec![],
        }],
    });
    offset += list_off_cap + item_data_cap + item_off_cap;

    // scores: List<Int32> - avg 3 items per row
    let score_list_off_cap = (row_count + 1) * 4;
    let score_item_cap = total_items * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0,
        offsets_offset: Some(offset),
        offsets_capacity: Some(score_list_off_cap),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![ColumnRegion {
            data_offset: offset + score_list_off_cap,
            data_capacity: score_item_cap,
            offsets_offset: None,
            offsets_capacity: None,
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: vec![],
        }],
    });
    offset += score_list_off_cap + score_item_cap;

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_map_layout(row_count: usize, _avg_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // id: Int64
    let id_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: id_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_cap;

    // metadata: Map<String, String> - uses Vec mode (no child_regions needed)
    // Map types use Vec-based decoding for better performance
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0, // Map uses Vec mode, no pre-allocated buffer needed
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![], // Map uses Vec mode
    });

    // counts: Map<String, Int32> - uses Vec mode (no child_regions needed)
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![], // Map uses Vec mode
    });

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_union_layout(row_count: usize, _avg_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // id: Int64
    let id_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: id_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_cap;

    // optional_name: Union ["null", "string"] - uses Vec mode (no child_regions needed)
    // Union types use Vec-based decoding for better performance
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0, // Union uses Vec mode
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![], // Union uses Vec mode
    });

    // optional_value: Union ["null", "double"] - uses Vec mode
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![], // Union uses Vec mode
    });

    // optional_count: Union ["null", "int"] - uses Vec mode
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![], // Union uses Vec mode
    });

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_struct_layout(row_count: usize, avg_len: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    // id: Int64
    let id_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: id_cap,
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_cap;

    // address: Struct { street, city, zip }
    // Struct has child regions for each field
    let street_data_cap = row_count * (avg_len + 10);
    let street_off_cap = (row_count + 1) * 4;
    let city_data_cap = row_count * (avg_len / 2 + 10);
    let city_off_cap = (row_count + 1) * 4;
    let zip_cap = row_count * 4;

    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: 0, // Struct itself has no data
        offsets_offset: None,
        offsets_capacity: None,
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![
            // street
            ColumnRegion {
                data_offset: offset,
                data_capacity: street_data_cap,
                offsets_offset: Some(offset + street_data_cap),
                offsets_capacity: Some(street_off_cap),
                null_bitmap_offset: None,
                null_bitmap_capacity: None,
                child_regions: vec![],
            },
            // city
            ColumnRegion {
                data_offset: offset + street_data_cap + street_off_cap,
                data_capacity: city_data_cap,
                offsets_offset: Some(offset + street_data_cap + street_off_cap + city_data_cap),
                offsets_capacity: Some(city_off_cap),
                null_bitmap_offset: None,
                null_bitmap_capacity: None,
                child_regions: vec![],
            },
            // zip
            ColumnRegion {
                data_offset: offset + street_data_cap + street_off_cap + city_data_cap + city_off_cap,
                data_capacity: zip_cap,
                offsets_offset: None,
                offsets_capacity: None,
                null_bitmap_offset: None,
                null_bitmap_capacity: None,
                child_regions: vec![],
            },
        ],
    });
    offset += street_data_cap + street_off_cap + city_data_cap + city_off_cap + zip_cap;

    // name: String
    let name_data_cap = row_count * (avg_len + 10);
    let name_off_cap = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset,
        data_capacity: name_data_cap,
        offsets_offset: Some(offset + name_data_cap),
        offsets_capacity: Some(name_off_cap),
        null_bitmap_offset: None,
        null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += name_data_cap + name_off_cap;

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn create_comprehensive_layout(row_count: usize, avg_len: usize) -> BufferLayout {
    // Comprehensive layout with all types - uses minimal allocation for complex types
    // that use Vec internally (Dictionary, List, Map, Union)
    let mut columns = Vec::new();
    let mut offset = 0;

    // id: Int64
    let id_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: id_cap,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += id_cap;

    // active: Boolean - bit-packed (8 values per byte)
    let bool_cap = (row_count + 7) / 8;
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: bool_cap,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += bool_cap;

    // name: String
    let name_data_cap = row_count * (avg_len + 10);
    let name_off_cap = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: name_data_cap,
        offsets_offset: Some(offset + name_data_cap), offsets_capacity: Some(name_off_cap),
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += name_data_cap + name_off_cap;

    // value: Float64
    let val_cap = row_count * 8;
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: val_cap,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += val_cap;

    // ratio: Float32
    let ratio_cap = row_count * 4;
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: ratio_cap,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += ratio_cap;

    // count: Int32
    let cnt_cap = row_count * 4;
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: cnt_cap,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += cnt_cap;

    // data: Binary
    let bin_data_cap = row_count * 30;
    let bin_off_cap = (row_count + 1) * 4;
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: bin_data_cap,
        offsets_offset: Some(offset + bin_data_cap), offsets_capacity: Some(bin_off_cap),
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += bin_data_cap + bin_off_cap;

    // checksum: FixedSizeBinary(16)
    let fixed_cap = row_count * 16;
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: fixed_cap,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });
    offset += fixed_cap;

    // status: Dictionary - uses Vec internally
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: 0,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });

    // tags: List<String> - uses Vec internally
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: 0,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });

    // scores: List<Int32> - uses Vec internally
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: 0,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });

    // metadata: Map - uses Vec internally
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: 0,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });

    // optional_note: nullable String - uses Vec internally for union
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: 0,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });

    // address: Struct - uses Vec internally
    columns.push(ColumnRegion {
        data_offset: offset, data_capacity: 0,
        offsets_offset: None, offsets_capacity: None,
        null_bitmap_offset: None, null_bitmap_capacity: None,
        child_regions: vec![],
    });

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

// === Vec Decoder Helper ===

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

// === Benchmarks ===

fn bench_all_types_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("AllTypes_Comparison_50k");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(50);

    let rows = 50000;
    let avg_len = 20;

    // === FLAT ===
    let flat_raw = gen_flat_data(rows);
    let flat_soe = gen_flat_data_soe(rows);
    let flat_schema = create_flat_schema();
    let flat_layout = create_flat_layout(rows);

    group.throughput(Throughput::Elements(rows as u64));

    let flat_buf = vec![0u8; flat_layout.total_size];
    let flat_provider = PooledBufferProvider::new(flat_buf, flat_layout.clone());
    let mut flat_pooled = PooledRecordDecoder::try_new(flat_schema.clone(), flat_provider).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "flat"), |b| {
        b.iter(|| {
            flat_pooled.decode(&flat_raw, rows).unwrap();
            black_box(flat_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "flat"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(FLAT_SCHEMA, rows);
            decoder.decode(&flat_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === STRINGS ===
    let str_raw = gen_strings_data(rows, avg_len);
    let str_soe = gen_strings_data_soe(rows, avg_len);
    let str_schema = create_strings_schema();
    let str_layout = create_strings_layout(rows, avg_len);

    let str_buf = vec![0u8; str_layout.total_size];
    let str_provider = PooledBufferProvider::new(str_buf, str_layout.clone());
    let mut str_pooled = PooledRecordDecoder::try_new(str_schema.clone(), str_provider).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "strings"), |b| {
        b.iter(|| {
            str_pooled.decode(&str_raw, rows).unwrap();
            black_box(str_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "strings"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(STRINGS_SCHEMA, rows);
            decoder.decode(&str_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === MIXED ===
    let mixed_raw = gen_mixed_data(rows, avg_len);
    let mixed_soe = gen_mixed_data_soe(rows, avg_len);
    let mixed_schema = create_mixed_schema();
    let mixed_layout = create_mixed_layout(rows, avg_len);

    let mixed_buf = vec![0u8; mixed_layout.total_size];
    let mixed_provider = PooledBufferProvider::new(mixed_buf, mixed_layout.clone());
    let mut mixed_pooled = PooledRecordDecoder::try_new(mixed_schema.clone(), mixed_provider).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "mixed"), |b| {
        b.iter(|| {
            mixed_pooled.decode(&mixed_raw, rows).unwrap();
            black_box(mixed_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "mixed"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(MIXED_SCHEMA, rows);
            decoder.decode(&mixed_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === DICTIONARY ===
    let dict_raw = gen_dict_data(rows);
    let dict_soe = gen_dict_data_soe(rows);
    let dict_schema = create_dict_schema();
    let dict_layout = create_dict_layout(rows);

    let dict_buf = vec![0u8; dict_layout.total_size.max(1)];
    let dict_provider = PooledBufferProvider::new(dict_buf, dict_layout.clone());
    let mut dict_pooled = PooledRecordDecoder::try_new(dict_schema.clone(), dict_provider).unwrap();
    // Set dictionary values for enum columns
    dict_pooled.set_dictionary_values(0, vec!["ACTIVE".to_string(), "INACTIVE".to_string(), "PENDING".to_string(), "ARCHIVED".to_string()]).unwrap();
    dict_pooled.set_dictionary_values(1, vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string(), "E".to_string()]).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "dictionary"), |b| {
        b.iter(|| {
            dict_pooled.decode(&dict_raw, rows).unwrap();
            black_box(dict_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "dictionary"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(DICT_SCHEMA, rows);
            decoder.decode(&dict_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === BOOLEAN + BINARY ===
    let bool_bin_raw = gen_bool_binary_data(rows);
    let bool_bin_soe = gen_bool_binary_data_soe(rows);
    let bool_bin_schema = create_bool_binary_schema();
    let bool_bin_layout = create_bool_binary_layout(rows);

    let bool_bin_buf = vec![0u8; bool_bin_layout.total_size.max(1)];
    let bool_bin_provider = PooledBufferProvider::new(bool_bin_buf, bool_bin_layout.clone());
    let mut bool_bin_pooled = PooledRecordDecoder::try_new(bool_bin_schema.clone(), bool_bin_provider).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "bool_binary"), |b| {
        b.iter(|| {
            bool_bin_pooled.decode(&bool_bin_raw, rows).unwrap();
            black_box(bool_bin_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "bool_binary"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(BOOL_BINARY_SCHEMA, rows);
            decoder.decode(&bool_bin_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === LIST ===
    let list_raw = gen_list_data(rows, avg_len);
    let list_soe = gen_list_data_soe(rows, avg_len);
    let list_schema = create_list_schema();
    let list_layout = create_list_layout(rows, avg_len);

    let list_buf = vec![0u8; list_layout.total_size.max(1)];
    let list_provider = PooledBufferProvider::new(list_buf, list_layout.clone());
    let mut list_pooled = PooledRecordDecoder::try_new(list_schema.clone(), list_provider).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "list"), |b| {
        b.iter(|| {
            list_pooled.decode(&list_raw, rows).unwrap();
            black_box(list_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "list"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(LIST_SCHEMA, rows);
            decoder.decode(&list_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === MAP ===
    let map_raw = gen_map_data(rows, avg_len);
    let map_soe = gen_map_data_soe(rows, avg_len);
    let map_schema = create_map_schema();
    let map_layout = create_map_layout(rows, avg_len);

    let map_buf = vec![0u8; map_layout.total_size.max(1)];
    let map_provider = PooledBufferProvider::new(map_buf, map_layout.clone());
    let mut map_pooled = PooledRecordDecoder::try_new(map_schema.clone(), map_provider).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "map"), |b| {
        b.iter(|| {
            map_pooled.decode(&map_raw, rows).unwrap();
            black_box(map_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "map"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(MAP_SCHEMA, rows);
            decoder.decode(&map_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === UNION (nullable) ===
    let union_raw = gen_union_data(rows, avg_len);
    let union_soe = gen_union_data_soe(rows, avg_len);
    let union_schema = create_union_schema();
    let union_layout = create_union_layout(rows, avg_len);

    let union_buf = vec![0u8; union_layout.total_size.max(1)];
    let union_provider = PooledBufferProvider::new(union_buf, union_layout.clone());
    let mut union_pooled = PooledRecordDecoder::try_new(union_schema.clone(), union_provider).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "union"), |b| {
        b.iter(|| {
            union_pooled.decode(&union_raw, rows).unwrap();
            black_box(union_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "union"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(UNION_SCHEMA, rows);
            decoder.decode(&union_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === STRUCT (nested record) ===
    let struct_raw = gen_struct_data(rows, avg_len);
    let struct_soe = gen_struct_data_soe(rows, avg_len);
    let struct_schema = create_struct_schema();
    let struct_layout = create_struct_layout(rows, avg_len);

    let struct_buf = vec![0u8; struct_layout.total_size.max(1)];
    let struct_provider = PooledBufferProvider::new(struct_buf, struct_layout.clone());
    let mut struct_pooled = PooledRecordDecoder::try_new(struct_schema.clone(), struct_provider).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "struct"), |b| {
        b.iter(|| {
            struct_pooled.decode(&struct_raw, rows).unwrap();
            black_box(struct_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "struct"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(STRUCT_SCHEMA, rows);
            decoder.decode(&struct_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    // === ALL TYPES COMBINED ===
    let all_raw = gen_all_types_data(rows, avg_len);
    let all_soe = gen_all_types_data_soe(rows, avg_len);
    let all_schema = create_all_types_schema();
    let all_layout = create_all_types_layout(rows, avg_len);

    let all_buf = vec![0u8; all_layout.total_size.max(1)];
    let all_provider = PooledBufferProvider::new(all_buf, all_layout.clone());
    let mut all_pooled = PooledRecordDecoder::try_new(all_schema.clone(), all_provider).unwrap();
    // Set dictionary values for enum columns (indices 3=status, 6=category)
    all_pooled.set_dictionary_values(3, vec!["ACTIVE".to_string(), "INACTIVE".to_string(), "PENDING".to_string()]).unwrap();
    all_pooled.set_dictionary_values(6, vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]).unwrap();

    group.bench_function(BenchmarkId::new("Pooled", "all_types"), |b| {
        b.iter(|| {
            all_pooled.decode(&all_raw, rows).unwrap();
            black_box(all_pooled.flush().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("Vec", "all_types"), |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(ALL_TYPES_SCHEMA, rows);
            decoder.decode(&all_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    group.finish();
}

/// Summary benchmark - just the key numbers
fn bench_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("Summary_Pooled_vs_Vec");
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));
    group.sample_size(30);

    let rows = 50000;
    let avg_len = 20;

    // All types combined - the most realistic scenario
    let all_raw = gen_all_types_data(rows, avg_len);
    let all_soe = gen_all_types_data_soe(rows, avg_len);
    let all_schema = create_all_types_schema();
    let all_layout = create_all_types_layout(rows, avg_len);

    group.throughput(Throughput::Elements(rows as u64));

    let all_buf = vec![0u8; all_layout.total_size.max(1)];
    let all_provider = PooledBufferProvider::new(all_buf, all_layout.clone());
    let mut all_pooled = PooledRecordDecoder::try_new(all_schema.clone(), all_provider).unwrap();
    // Set dictionary values for enum columns (indices 3=status, 6=category)
    all_pooled.set_dictionary_values(3, vec!["ACTIVE".to_string(), "INACTIVE".to_string(), "PENDING".to_string()]).unwrap();
    all_pooled.set_dictionary_values(6, vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]).unwrap();

    group.bench_function("Pooled_realistic", |b| {
        b.iter(|| {
            all_pooled.decode(&all_raw, rows).unwrap();
            black_box(all_pooled.flush().unwrap())
        })
    });

    group.bench_function("Vec_realistic", |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(ALL_TYPES_SCHEMA, rows);
            decoder.decode(&all_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    group.finish();
}

/// Comprehensive benchmark with ALL Avro types
fn bench_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comprehensive_All_Avro_Types");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(30);

    let rows = 50000;
    let avg_len = 20;

    // Comprehensive schema with ALL types
    let comp_raw = gen_comprehensive_data(rows, avg_len);
    let comp_soe = gen_comprehensive_data_soe(rows, avg_len);
    let comp_schema = create_comprehensive_schema();
    let comp_layout = create_comprehensive_layout(rows, avg_len);

    group.throughput(Throughput::Elements(rows as u64));

    // Pooled decoder
    let comp_buf = vec![0u8; comp_layout.total_size.max(1)];
    let comp_provider = PooledBufferProvider::new(comp_buf, comp_layout.clone());
    let mut comp_pooled = PooledRecordDecoder::try_new(comp_schema.clone(), comp_provider).unwrap();
    // Set dictionary values for status enum (column index 8)
    comp_pooled.set_dictionary_values(8, vec!["ACTIVE".to_string(), "INACTIVE".to_string(), "PENDING".to_string()]).unwrap();

    group.bench_function("Pooled_comprehensive", |b| {
        b.iter(|| {
            comp_pooled.decode(&comp_raw, rows).unwrap();
            black_box(comp_pooled.flush().unwrap())
        })
    });

    group.bench_function("Vec_comprehensive", |b| {
        b.iter(|| {
            let mut decoder = create_vec_decoder(COMPREHENSIVE_SCHEMA, rows);
            decoder.decode(&comp_soe).unwrap();
            black_box(decoder.flush().unwrap().unwrap())
        })
    });

    group.finish();
}

criterion_group! {
    name = all_types;
    config = Criterion::default().configure_from_args();
    targets = bench_all_types_comparison, bench_summary, bench_comprehensive
}
criterion_main!(all_types);
