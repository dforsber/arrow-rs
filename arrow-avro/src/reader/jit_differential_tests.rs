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

//! Differential compliance oracle for the JIT Avro→Arrow decoder.
//!
//! Strategy (independent ground truth, no self-encoding):
//! 1. Encode records with `apache_avro` (the canonical Apache reference writer)
//!    into raw Avro datum bytes — the same wire bytes the production Kafka path
//!    concatenates per record.
//! 2. Decode those bytes two ways:
//!    - **reference**: arrow-avro's own `RecordDecoder` (interpreted, JIT-free —
//!      verified: `record.rs` contains no `jit` references), and
//!    - **under test**: the Cranelift `JitDecoder` + `buffers_to_batch_owned`
//!      converter (the exact two stages the production path drives).
//! 3. Assert the two `RecordBatch`es are identical.
//!
//! Because the reference decoder is an independent implementation, a bug in
//! *either* JIT stage (Avro parse OR buffer→Arrow conversion — the boolean
//! byte-vs-bit class) makes the batches diverge. A symmetric self-encode/decode
//! round-trip could not catch that; a third-party encoder + independent decoder
//! can.
//!
//! Coverage discipline:
//! - **Verified baseline** (`Kind`): every kind must round-trip — exhaustive
//!   edge cases + a proptest over random multi-column batches (incl. nullable).
//! - **Loud rejection**: unsupported shapes (e.g. Map) must make the JIT path
//!   return `Err`, never silently-wrong data. `gate_*` tests assert this.
//! Add a kind only once it has a passing differential test.
//!
//! Deferred (need richer encoding plumbing, tracked for the next increment):
//! Decimal128/256 (`apache_avro::Decimal`), Duration (Avro `duration` fixed-12).

use super::jit_buffer_utils::{buffers_to_batch_owned, BufferSet};
use super::jit_decoder::{JitDecoder, OutputBuffer};
use super::record::RecordDecoder;
use crate::codec::AvroFieldBuilder;
use crate::schema::Schema as AvroSchema;
use apache_avro::{to_avro_datum, types::Value, Schema as ApacheSchema};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType, SchemaRef};
use proptest::prelude::*;

/// A generated Avro value. Each variant maps 1:1 to a [`Kind`] and to the Arrow
/// type the decoders must produce.
#[derive(Clone, Debug, PartialEq)]
enum Scalar {
    Bool(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Str(String),
    Bytes(Vec<u8>),
    Date(i32),
    TimeMillis(i32),
    TimeMicros(i64),
    TsMillis(i64),
    TsMicros(i64),
    Fixed(Vec<u8>),
}

impl Scalar {
    fn to_value(&self) -> Value {
        match self {
            Scalar::Bool(v) => Value::Boolean(*v),
            Scalar::Int(v) => Value::Int(*v),
            Scalar::Long(v) => Value::Long(*v),
            Scalar::Float(v) => Value::Float(*v),
            Scalar::Double(v) => Value::Double(*v),
            Scalar::Str(v) => Value::String(v.clone()),
            Scalar::Bytes(v) => Value::Bytes(v.clone()),
            Scalar::Date(v) => Value::Date(*v),
            Scalar::TimeMillis(v) => Value::TimeMillis(*v),
            Scalar::TimeMicros(v) => Value::TimeMicros(*v),
            Scalar::TsMillis(v) => Value::TimestampMillis(*v),
            Scalar::TsMicros(v) => Value::TimestampMicros(*v),
            Scalar::Fixed(v) => Value::Fixed(v.len(), v.clone()),
        }
    }
}

/// A column's logical kind: knows its Avro type-JSON fragment and how to
/// generate a value. The single source of truth for "supported".
#[derive(Clone, Debug)]
enum Kind {
    Bool,
    Int,
    Long,
    Float,
    Double,
    Str,
    Bytes,
    Date,
    TimeMillis,
    TimeMicros,
    TsMillis,
    TsMicros,
    /// Avro `fixed` of the given size → Arrow FixedSizeBinary.
    Fixed(usize),
}

impl Kind {
    /// The complete Avro type-JSON fragment for field index `i`. `fixed`
    /// requires a unique name per field.
    fn type_json(&self, i: usize) -> String {
        match self {
            Kind::Bool => "\"boolean\"".into(),
            Kind::Int => "\"int\"".into(),
            Kind::Long => "\"long\"".into(),
            Kind::Float => "\"float\"".into(),
            Kind::Double => "\"double\"".into(),
            Kind::Str => "\"string\"".into(),
            Kind::Bytes => "\"bytes\"".into(),
            Kind::Date => r#"{"type":"int","logicalType":"date"}"#.into(),
            Kind::TimeMillis => r#"{"type":"int","logicalType":"time-millis"}"#.into(),
            Kind::TimeMicros => r#"{"type":"long","logicalType":"time-micros"}"#.into(),
            Kind::TsMillis => r#"{"type":"long","logicalType":"timestamp-millis"}"#.into(),
            Kind::TsMicros => r#"{"type":"long","logicalType":"timestamp-micros"}"#.into(),
            Kind::Fixed(n) => format!(r#"{{"type":"fixed","name":"Fixed_f{i}","size":{n}}}"#),
        }
    }

    /// Generate a value from a deterministic PRNG state. Temporal values are
    /// constrained to valid domains (time-of-day ranges); the decoders pass the
    /// raw integers through, but staying in-range avoids encoder validation.
    fn gen_value(&self, state: &mut u64) -> Scalar {
        let r = next_u64(state);
        match self {
            Kind::Bool => Scalar::Bool(r & 1 == 1),
            Kind::Int => Scalar::Int(r as i32),
            Kind::Long => Scalar::Long(r as i64),
            Kind::Float => {
                let f = f32::from_bits(r as u32);
                Scalar::Float(if f.is_nan() { 0.0 } else { f })
            }
            Kind::Double => {
                let d = f64::from_bits(r);
                Scalar::Double(if d.is_nan() { 0.0 } else { d })
            }
            Kind::Str => {
                let len = (r % 12) as usize;
                let s: String = (0..len)
                    .map(|_| ((next_u64(state) % 26) as u8 + b'a') as char)
                    .collect();
                Scalar::Str(s)
            }
            Kind::Bytes => {
                let len = (r % 12) as usize;
                Scalar::Bytes((0..len).map(|_| next_u64(state) as u8).collect())
            }
            Kind::Date => Scalar::Date(r as i32),
            Kind::TimeMillis => Scalar::TimeMillis((r % 86_400_000) as i32),
            Kind::TimeMicros => Scalar::TimeMicros((r % 86_400_000_000) as i64),
            Kind::TsMillis => Scalar::TsMillis(r as i64),
            Kind::TsMicros => Scalar::TsMicros(r as i64),
            Kind::Fixed(n) => Scalar::Fixed((0..*n).map(|_| next_u64(state) as u8).collect()),
        }
    }
}

/// A column spec: kind + whether it is nullable (`["null", T]`).
type Col = (Kind, bool);

/// Build the Avro record-schema JSON. Nullable columns use a `["null", T]`
/// union (null first → default branch 0, matching the encoder's tagging).
fn record_json(cols: &[Col]) -> String {
    let mut fields = String::new();
    for (i, (kind, nullable)) in cols.iter().enumerate() {
        if i > 0 {
            fields.push(',');
        }
        let frag = kind.type_json(i);
        if *nullable {
            fields.push_str(&format!(r#"{{"name":"f{i}","type":["null",{frag}]}}"#));
        } else {
            fields.push_str(&format!(r#"{{"name":"f{i}","type":{frag}}}"#));
        }
    }
    format!(r#"{{"type":"record","name":"R","fields":[{fields}]}}"#)
}

/// Encode `rows` to a single concatenated raw-datum buffer via apache_avro.
/// `None` in a nullable column encodes as union branch 0 (`null`); a present
/// value in a nullable column encodes as branch 1.
fn encode_rows(json: &str, cols: &[Col], rows: &[Vec<Option<Scalar>>]) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(json).expect("apache parse_str");
    let mut body = Vec::new();
    for row in rows {
        let rec = Value::Record(
            row.iter()
                .enumerate()
                .map(|(i, cell)| {
                    let (_, nullable) = &cols[i];
                    let v = match cell {
                        Some(s) if *nullable => Value::Union(1, Box::new(s.to_value())),
                        Some(s) => s.to_value(),
                        None => Value::Union(0, Box::new(Value::Null)),
                    };
                    (format!("f{i}"), v)
                })
                .collect(),
        );
        body.extend_from_slice(&to_avro_datum(&schema, rec).expect("to_avro_datum"));
    }
    body
}

/// Reference decode via the interpreted `RecordDecoder`. Returns the batch and
/// the Arrow schema it derived (fed to the JIT path so both use an identical
/// schema — any difference is then purely in the values).
fn reference_decode(json: &str, body: &[u8], rows: usize) -> (RecordBatch, SchemaRef) {
    let avro_schema: AvroSchema = serde_json::from_str(json).expect("serde parse avro schema");
    let field = AvroFieldBuilder::new(&avro_schema)
        .build()
        .expect("AvroFieldBuilder::build");
    let mut rd = RecordDecoder::try_new_with_options(field.data_type()).expect("RecordDecoder");
    if rows > 0 {
        rd.decode(body, rows).expect("reference decode");
    }
    let batch = rd.flush().expect("reference flush");
    let schema = batch.schema();
    (batch, schema)
}

/// Packed primary-buffer byte size for a fixed-scalar column. Mirrors the
/// production `packed_column_bytes` in `src/kafka/pooled_avro_decoder`.
fn packed_column_bytes(data_type: &DataType) -> usize {
    match data_type {
        DataType::Struct(children) => {
            children.iter().map(|f| packed_column_bytes(f.data_type())).sum()
        }
        other => JitDecoder::fixed_size_for_type(other).unwrap_or(8),
    }
}

/// JIT decode via `JitDecoder` + `buffers_to_batch_owned`, mirroring the
/// production allocation in `decode_with_jit`. Returns `Err` for unsupported
/// shapes (the loud-rejection contract) rather than silently-wrong data.
fn jit_decode(arrow_schema: &SchemaRef, body: &[u8], rows: usize) -> Result<RecordBatch, ArrowError> {
    let decoder = JitDecoder::compile(arrow_schema)
        .map_err(|e| ArrowError::ComputeError(format!("JIT compile: {e}")))?;

    // Empty batch: production early-returns on empty records before allocating.
    // Mirror that — and avoid 0-length buffers whose dangling ptr is misaligned
    // for wide ScalarBuffer<T> (the array length would otherwise derive from
    // bytes/elem, breaking for FixedSizeBinary(1) and wide types alike).
    if rows == 0 {
        return Ok(RecordBatch::new_empty(arrow_schema.clone()));
    }

    let var_data_cap = (body.len() + 64).max(64);
    let mut sets: Vec<BufferSet> = Vec::with_capacity(arrow_schema.fields().len());
    for (ci, field) in arrow_schema.fields().iter().enumerate() {
        let slot = decoder
            .buffer_slot(ci)
            .ok_or_else(|| ArrowError::ComputeError(format!("missing buffer slot {ci}")))?;
        let nullable = field.is_nullable();
        let validity_bytes = if nullable { rows.div_ceil(8) } else { 0 };
        let bs = if slot.secondary.is_some() {
            let offsets = (rows + 1) * 4;
            let mut b = BufferSet::new_variable(offsets, var_data_cap);
            if nullable {
                b = b.with_validity(validity_bytes);
            }
            b
        } else {
            // Exactly rows*elem bytes — NOT max(1). A spare byte would make a
            // FixedSizeBinary(1) column report length 1 for an empty batch
            // (length = bytes/size). Production sizes this as row_count*elem too.
            let elem = packed_column_bytes(field.data_type());
            let mut b = BufferSet::new_fixed(rows * elem);
            if nullable {
                b = b.with_validity(validity_bytes);
            }
            b
        };
        sets.push(bs);
    }

    let mut outs: Vec<OutputBuffer> = Vec::new();
    for (ci, bs) in sets.iter_mut().enumerate() {
        let slot = decoder.buffer_slot(ci).unwrap();
        outs.push(OutputBuffer::new(bs.primary_ptr(), bs.primary.len()));
        if slot.secondary.is_some() {
            let cap = bs.secondary.as_ref().unwrap().len();
            outs.push(OutputBuffer::new(bs.secondary_ptr(), cap));
        }
        if slot.validity.is_some() {
            let cap = bs.validity.as_ref().unwrap().len();
            outs.push(OutputBuffer::new(bs.validity_ptr(), cap));
        }
    }

    if rows > 0 {
        decoder
            .decode_checked(body, &outs, rows)
            .map_err(ArrowError::ComputeError)?;
    }
    buffers_to_batch_owned(arrow_schema, sets, rows)
}

/// The differential assertion: encode, decode both ways, require equality.
fn assert_diff(cols: &[Col], rows: &[Vec<Option<Scalar>>]) {
    let json = record_json(cols);
    let body = encode_rows(&json, cols, rows);
    let (ref_batch, schema) = reference_decode(&json, &body, rows.len());
    let jit_batch = jit_decode(&schema, &body, rows.len())
        .unwrap_or_else(|e| panic!("JIT decode failed for {cols:?}: {e}"));
    assert_eq!(
        ref_batch, jit_batch,
        "JIT batch diverged from reference for columns {cols:?}"
    );
}

/// Convenience: single required (non-nullable) column, one value per row.
fn diff_single(kind: Kind, vals: Vec<Scalar>) {
    let cols = vec![(kind, false)];
    let rows: Vec<Vec<Option<Scalar>>> = vals.into_iter().map(|v| vec![Some(v)]).collect();
    assert_diff(&cols, &rows);
}

// ===========================================================================
// Verified baseline — exhaustive edge cases per supported kind.
// ===========================================================================

#[test]
fn diff_bool_edges() {
    diff_single(
        Kind::Bool,
        [true, false, false, true, true, true, false, false, true]
            .iter()
            .map(|&b| Scalar::Bool(b))
            .collect(),
    );
}

#[test]
fn diff_int_edges() {
    diff_single(
        Kind::Int,
        [i32::MIN, -1, 0, 1, 127, 128, i32::MAX]
            .iter()
            .map(|&v| Scalar::Int(v))
            .collect(),
    );
}

#[test]
fn diff_long_edges() {
    diff_single(
        Kind::Long,
        [i64::MIN, -1, 0, 1, i64::MAX]
            .iter()
            .map(|&v| Scalar::Long(v))
            .collect(),
    );
}

#[test]
fn diff_float_edges() {
    diff_single(
        Kind::Float,
        [0.0f32, -0.0, 1.5, -1.5, f32::MIN, f32::MAX, f32::INFINITY]
            .iter()
            .map(|&v| Scalar::Float(v))
            .collect(),
    );
}

#[test]
fn diff_double_edges() {
    diff_single(
        Kind::Double,
        [0.0f64, -0.0, 1.5, -1.5, f64::MIN, f64::MAX]
            .iter()
            .map(|&v| Scalar::Double(v))
            .collect(),
    );
}

#[test]
fn diff_string_edges() {
    diff_single(
        Kind::Str,
        ["", "a", "hello world", "ünïcödé→★", "with\ttab\nnewline"]
            .iter()
            .map(|s| Scalar::Str(s.to_string()))
            .collect(),
    );
}

#[test]
fn diff_bytes_edges() {
    diff_single(
        Kind::Bytes,
        [vec![], vec![0u8], vec![0u8, 255, 1, 254], vec![7u8; 100]]
            .iter()
            .map(|b| Scalar::Bytes(b.clone()))
            .collect(),
    );
}

#[test]
fn diff_date_edges() {
    diff_single(
        Kind::Date,
        [i32::MIN, -1, 0, 1, 19000, i32::MAX]
            .iter()
            .map(|&v| Scalar::Date(v))
            .collect(),
    );
}

#[test]
fn diff_time_millis_edges() {
    diff_single(
        Kind::TimeMillis,
        [0, 1, 1000, 86_399_999]
            .iter()
            .map(|&v| Scalar::TimeMillis(v))
            .collect(),
    );
}

#[test]
fn diff_time_micros_edges() {
    diff_single(
        Kind::TimeMicros,
        [0i64, 1, 1_000_000, 86_399_999_999]
            .iter()
            .map(|&v| Scalar::TimeMicros(v))
            .collect(),
    );
}

#[test]
fn diff_timestamp_millis_edges() {
    diff_single(
        Kind::TsMillis,
        [i64::MIN, -1, 0, 1, 1_700_000_000_000, i64::MAX]
            .iter()
            .map(|&v| Scalar::TsMillis(v))
            .collect(),
    );
}

#[test]
fn diff_timestamp_micros_edges() {
    diff_single(
        Kind::TsMicros,
        [i64::MIN, -1, 0, 1, 1_700_000_000_000_000, i64::MAX]
            .iter()
            .map(|&v| Scalar::TsMicros(v))
            .collect(),
    );
}

#[test]
fn diff_fixed_edges() {
    // Avro `fixed` → Arrow FixedSizeBinary(n). All values must be exactly n bytes.
    diff_single(
        Kind::Fixed(16),
        vec![
            Scalar::Fixed(vec![0u8; 16]),
            Scalar::Fixed((0u8..16).collect()),
            Scalar::Fixed(vec![255u8; 16]),
        ],
    );
}

#[test]
fn diff_list_long() {
    // List<long> encoded by apache_avro (canonical: count / items / 0-terminator
    // blocks). Reproduces the Tier-2 pooled-path List failure in the fast repo.
    let json = r#"{"type":"record","name":"R","fields":[
        {"name":"l","type":{"type":"array","items":"long"}}]}"#;
    let schema = ApacheSchema::parse_str(json).expect("apache parse");
    let lists: Vec<Vec<i64>> = vec![vec![], vec![1, -2, 3], vec![7; 5]];
    let mut body = Vec::new();
    for arr in &lists {
        let rec = Value::Record(vec![(
            "l".into(),
            Value::Array(arr.iter().map(|v| Value::Long(*v)).collect()),
        )]);
        body.extend_from_slice(&to_avro_datum(&schema, rec).expect("encode"));
    }
    let (ref_batch, arrow_schema) = reference_decode(json, &body, lists.len());
    let jit_batch = jit_decode(&arrow_schema, &body, lists.len())
        .unwrap_or_else(|e| panic!("JIT list decode failed: {e}"));
    assert_eq!(ref_batch, jit_batch, "List<long> JIT vs reference");
}

#[test]
fn diff_list_long_pooled_chunked() {
    // The pooled + decode_chunked path (what boilstream's production pooled
    // decoder drives), allocated like `allocate_pooled_buffers_adaptive`. A
    // non-nullable List<Int64> must produce element-indexed offsets [0,0,3,8]
    // and decode cleanly. (The Tier-2 failure that motivated this came from the
    // caller mapping the list child as nullable, compiling the wrong element
    // decoder — see boilstream's ArrowConverter fix.)
    use super::jit_buffer_utils::{pooled_buffers_to_batch_owned, PooledBufferSet};
    let json = r#"{"type":"record","name":"R","fields":[
        {"name":"l","type":{"type":"array","items":"long"}}]}"#;
    let schema = ApacheSchema::parse_str(json).expect("apache parse");
    let lists: Vec<Vec<i64>> = vec![vec![], vec![1, -2, 3], vec![7; 5]];
    let mut body = Vec::new();
    for arr in &lists {
        let rec = Value::Record(vec![(
            "l".into(),
            Value::Array(arr.iter().map(|v| Value::Long(*v)).collect()),
        )]);
        body.extend_from_slice(&to_avro_datum(&schema, rec).expect("encode"));
    }
    let (ref_batch, arrow_schema) = reference_decode(json, &body, lists.len());
    let rows = lists.len();
    let decoder = JitDecoder::compile(&arrow_schema).expect("compile");

    let mut bs = PooledBufferSet::new_variable((rows + 1) * 4, rows * 64);
    let outs = vec![
        OutputBuffer::new(bs.primary_ptr(), bs.primary_capacity()),
        OutputBuffer::new(bs.secondary_ptr(), bs.secondary_capacity()),
    ];
    let res = decoder.decode_chunked(&body, &outs, rows);
    assert_eq!(res.rows_decoded, rows, "chunked decode rows");
    let batch =
        pooled_buffers_to_batch_owned(&arrow_schema, vec![bs], res.rows_decoded).expect("convert");
    assert_eq!(ref_batch, batch, "pooled+chunked List<long> vs reference");
}

#[test]
fn diff_fixed1_three_rows() {
    // size-1 FixedSizeBinary: length = bytes/size, so off-by-one allocation is
    // visible here.
    diff_single(
        Kind::Fixed(1),
        vec![Scalar::Fixed(vec![1]), Scalar::Fixed(vec![2]), Scalar::Fixed(vec![3])],
    );
}

#[test]
fn diff_fixed_empty_batch() {
    // Empty batch + FixedSizeBinary(1): array length must be 0, not 1.
    assert_diff(&[(Kind::Fixed(1), false)], &[]);
}

/// Minimal big-endian two's-complement encoding of an i128 (what real Avro
/// `bytes`-backed decimal producers emit — variable length, sign-significant).
fn i128_be_min(v: i128) -> Vec<u8> {
    let full = v.to_be_bytes();
    let mut start = 0usize;
    if v >= 0 {
        while start < 15 && full[start] == 0x00 && full[start + 1] & 0x80 == 0 {
            start += 1;
        }
    } else {
        while start < 15 && full[start] == 0xFF && full[start + 1] & 0x80 != 0 {
            start += 1;
        }
    }
    full[start..].to_vec()
}

#[test]
fn diff_decimal128_bytes() {
    // bytes-backed decimal: zigzag length + minimal two's-complement bytes.
    // Sign-boundary values (127/128, 255/256, ...) force length transitions in
    // the minimal encoding, exercising the JIT's variable-length sign-extension.
    use apache_avro::Decimal;
    let json = r#"{"type":"record","name":"R","fields":[
        {"name":"d","type":{"type":"bytes","logicalType":"decimal","precision":18,"scale":4}}]}"#;
    let schema = ApacheSchema::parse_str(json).expect("apache parse");
    let vals: Vec<i128> = vec![
        0, 1, -1, 127, 128, -128, -129, 255, 256, -256, 32767, -32768, 65535, -65536, 12_345_678,
        -12_345_678, 999_999_999_999_999_999, -999_999_999_999_999_999,
    ];
    let mut body = Vec::new();
    for v in &vals {
        let rec = Value::Record(vec![(
            "d".into(),
            Value::Decimal(Decimal::from(i128_be_min(*v))),
        )]);
        body.extend_from_slice(&to_avro_datum(&schema, rec).expect("encode decimal"));
    }
    let (ref_batch, arrow_schema) = reference_decode(json, &body, vals.len());
    let jit_batch = jit_decode(&arrow_schema, &body, vals.len())
        .unwrap_or_else(|e| panic!("JIT bytes-decimal decode failed: {e}"));
    assert_eq!(ref_batch, jit_batch, "bytes-backed Decimal128(18,4) JIT vs reference");
}

#[test]
fn diff_decimal128_bytes_nullable() {
    use apache_avro::Decimal;
    let json = r#"{"type":"record","name":"R","fields":[
        {"name":"d","type":["null",{"type":"bytes","logicalType":"decimal","precision":12,"scale":2}]}]}"#;
    let schema = ApacheSchema::parse_str(json).expect("apache parse");
    let vals: Vec<Option<i128>> = vec![Some(1), None, Some(-12345), None, None, Some(999_999_999)];
    let mut body = Vec::new();
    for v in &vals {
        let cell = match v {
            Some(x) => Value::Union(1, Box::new(Value::Decimal(Decimal::from(i128_be_min(*x))))),
            None => Value::Union(0, Box::new(Value::Null)),
        };
        let rec = Value::Record(vec![("d".into(), cell)]);
        body.extend_from_slice(&to_avro_datum(&schema, rec).expect("encode"));
    }
    let (ref_batch, arrow_schema) = reference_decode(json, &body, vals.len());
    let jit_batch = jit_decode(&arrow_schema, &body, vals.len())
        .unwrap_or_else(|e| panic!("JIT nullable bytes-decimal decode failed: {e}"));
    assert_eq!(ref_batch, jit_batch, "nullable bytes-backed Decimal128 JIT vs reference");
}

#[test]
fn diff_decimal256_bytes() {
    // bytes-backed Decimal256 (precision 76): sub-i128 values (sign-extended to
    // 32) plus crafted >16-byte values to exercise the full 32-byte path.
    use apache_avro::Decimal;
    let json = r#"{"type":"record","name":"R","fields":[
        {"name":"d","type":{"type":"bytes","logicalType":"decimal","precision":76,"scale":0}}]}"#;
    let schema = ApacheSchema::parse_str(json).expect("apache parse");
    let mut bodies: Vec<Vec<u8>> = vec![0i128, 1, -1, 255, -256, i128::MAX, i128::MIN]
        .iter()
        .map(|v| i128_be_min(*v))
        .collect();
    // 20-byte positive (leading 0x01, MSB clear) and negative (leading 0xFE, MSB set).
    let mut big_pos = vec![0x01u8];
    big_pos.extend(std::iter::repeat(0xABu8).take(19));
    bodies.push(big_pos);
    let mut big_neg = vec![0xFEu8];
    big_neg.extend(std::iter::repeat(0x01u8).take(19));
    bodies.push(big_neg);

    let rows = bodies.len();
    let mut body = Vec::new();
    for bytes in &bodies {
        let rec = Value::Record(vec![("d".into(), Value::Decimal(Decimal::from(bytes.clone())))]);
        body.extend_from_slice(&to_avro_datum(&schema, rec).expect("encode"));
    }
    let (ref_batch, arrow_schema) = reference_decode(json, &body, rows);
    let jit_batch = jit_decode(&arrow_schema, &body, rows)
        .unwrap_or_else(|e| panic!("JIT bytes-decimal256 decode failed: {e}"));
    assert_eq!(ref_batch, jit_batch, "bytes-backed Decimal256(76,0) JIT vs reference");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Random bytes-backed Decimal128 values (precision 38) — exercises every
    /// minimal-encoding byte length and sign pattern against the reference.
    #[test]
    fn prop_decimal128_bytes(raw in prop::collection::vec(any::<i128>(), 1..=24)) {
        use apache_avro::Decimal;
        // Constrain to fit precision 38 (|v| < 10^38, which is < i128::MAX).
        let limit = 10i128.pow(38);
        let vals: Vec<i128> = raw.iter().map(|v| v % limit).collect();
        let json = r#"{"type":"record","name":"R","fields":[
            {"name":"d","type":{"type":"bytes","logicalType":"decimal","precision":38,"scale":0}}]}"#;
        let schema = ApacheSchema::parse_str(json).expect("apache parse");
        let mut body = Vec::new();
        for v in &vals {
            let rec = Value::Record(vec![(
                "d".into(),
                Value::Decimal(Decimal::from(i128_be_min(*v))),
            )]);
            body.extend_from_slice(&to_avro_datum(&schema, rec).expect("encode"));
        }
        let (ref_batch, arrow_schema) = reference_decode(json, &body, vals.len());
        let jit_batch = jit_decode(&arrow_schema, &body, vals.len())
            .map_err(|e| TestCaseError::fail(format!("jit decode: {e}")))?;
        prop_assert_eq!(ref_batch, jit_batch);
    }
}

#[test]
fn diff_decimal128_fixed16() {
    // SUPPORTED decimal path: `fixed(16)`-backed decimal — raw 16 big-endian
    // bytes, no length prefix (what emit_decimal128_decode reads).
    use apache_avro::Decimal;
    let json = r#"{"type":"record","name":"R","fields":[
        {"name":"d","type":{"type":"fixed","size":16,"name":"D16","logicalType":"decimal","precision":10,"scale":2}}]}"#;
    let schema = ApacheSchema::parse_str(json).expect("apache parse");
    let vals: Vec<i128> = vec![0, 1, -1, 12345, -98765, 9_999_999_999, -9_999_999_999];
    let mut body = Vec::new();
    for v in &vals {
        // fixed(16): exactly 16 big-endian bytes (sign-extended i128).
        let rec = Value::Record(vec![(
            "d".into(),
            Value::Decimal(Decimal::from(v.to_be_bytes().to_vec())),
        )]);
        body.extend_from_slice(&to_avro_datum(&schema, rec).expect("encode decimal"));
    }
    let (ref_batch, arrow_schema) = reference_decode(json, &body, vals.len());
    let jit_batch = jit_decode(&arrow_schema, &body, vals.len())
        .unwrap_or_else(|e| panic!("JIT decimal(fixed16) decode failed: {e}"));
    assert_eq!(ref_batch, jit_batch, "Decimal128(10,2) fixed(16) JIT vs reference");
}

// GAP (documented, not yet supported — found by this oracle):
// - `bytes`-backed decimal (length-prefixed minimal two's-complement, the more
//   common Avro decimal encoding) is mis-decoded: emit_decimal128_decode reads a
//   fixed 16 raw bytes with NO length prefix, so it misreads the varint length
//   byte as data → bounds violation or silent corruption. boilstream's
//   ArrowConverter maps all decimals → Decimal128, so a bytes-backed producer
//   would corrupt. Needs JIT support for length-prefixed decimal bytes, or a
//   loud rejection of bytes-backed decimals at ingestion.
// - Avro `duration` (fixed-12: months/days/MILLIS) → Arrow Interval(MonthDayNano):
//   emit_interval_decode expects 16 Arrow-native bytes (months/days/NANOS), not
//   the 12-byte Avro layout — incompatible. (boilstream maps Avro fixed→
//   FixedSizeBinary, so it does not hit this Interval path.)

#[test]
fn diff_multi_column_mixed() {
    let cols = vec![
        (Kind::Bool, false),
        (Kind::Int, false),
        (Kind::Str, false),
        (Kind::TsMicros, false),
    ];
    let rows = vec![
        vec![
            Some(Scalar::Bool(true)),
            Some(Scalar::Int(7)),
            Some(Scalar::Str("x".into())),
            Some(Scalar::TsMicros(1)),
        ],
        vec![
            Some(Scalar::Bool(false)),
            Some(Scalar::Int(-9)),
            Some(Scalar::Str("".into())),
            Some(Scalar::TsMicros(-1)),
        ],
    ];
    assert_diff(&cols, &rows);
}

#[test]
fn diff_empty_batch() {
    // Zero rows must still produce equal (empty) batches with the right schema.
    assert_diff(&[(Kind::Int, false)], &[]);
}

#[test]
fn diff_nullable_scalars_edges() {
    // ["null", T] unions — the most common real Avro shape. Interleave present
    // and null values so validity bitmaps span byte boundaries.
    let cols = vec![(Kind::Long, true), (Kind::Str, true), (Kind::Bool, true)];
    let rows: Vec<Vec<Option<Scalar>>> = vec![
        vec![Some(Scalar::Long(1)), None, Some(Scalar::Bool(true))],
        vec![None, Some(Scalar::Str("x".into())), None],
        vec![Some(Scalar::Long(-9)), Some(Scalar::Str("".into())), Some(Scalar::Bool(false))],
        vec![None, None, None],
        vec![Some(Scalar::Long(i64::MAX)), Some(Scalar::Str("ünï".into())), Some(Scalar::Bool(true))],
    ];
    assert_diff(&cols, &rows);
}

#[test]
fn diff_nullable_mixed_with_required() {
    let cols = vec![(Kind::Int, false), (Kind::Double, true), (Kind::Date, true)];
    let rows: Vec<Vec<Option<Scalar>>> = vec![
        vec![Some(Scalar::Int(7)), Some(Scalar::Double(1.5)), None],
        vec![Some(Scalar::Int(-1)), None, Some(Scalar::Date(100))],
        vec![Some(Scalar::Int(0)), None, None],
    ];
    assert_diff(&cols, &rows);
}

// ===========================================================================
// Loud-rejection gate — unsupported shapes must Err, never silently corrupt.
// ===========================================================================

#[test]
fn gate_map_is_loudly_rejected() {
    let json = r#"{"type":"record","name":"R","fields":[
        {"name":"m","type":{"type":"map","values":"long"}}]}"#;
    let schema = ApacheSchema::parse_str(json).expect("apache parse");
    let rec = Value::Record(vec![(
        "m".into(),
        Value::Map(std::collections::HashMap::from([("k".to_string(), Value::Long(1))])),
    )]);
    let body = to_avro_datum(&schema, rec).expect("encode map");

    let (_ref_batch, arrow_schema) = reference_decode(json, &body, 1);
    let jit_result = jit_decode(&arrow_schema, &body, 1);
    assert!(
        jit_result.is_err(),
        "Map must be loudly rejected by the JIT path, got Ok (silent-wrong risk)"
    );
}

// ===========================================================================
// Property test — random multi-column batches over the supported kinds.
// proptest controls (kinds, per-column nullability, row count, value seed); a
// deterministic SplitMix64 fills values from the seed so failures replay
// exactly. proptest shrinks structure (drop columns/rows) to a minimal repro.
// ===========================================================================

fn kind_strategy() -> impl Strategy<Value = Kind> {
    prop_oneof![
        Just(Kind::Bool),
        Just(Kind::Int),
        Just(Kind::Long),
        Just(Kind::Float),
        Just(Kind::Double),
        Just(Kind::Str),
        Just(Kind::Bytes),
        Just(Kind::Date),
        Just(Kind::TimeMillis),
        Just(Kind::TimeMicros),
        Just(Kind::TsMillis),
        Just(Kind::TsMicros),
        (1usize..=20).prop_map(Kind::Fixed),
    ]
}

/// SplitMix64 — reproducible from the proptest-chosen seed, no rand-version dep.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn prop_differential_supported_kinds(
        kinds in prop::collection::vec(kind_strategy(), 1..=5),
        nullable_flags in prop::collection::vec(any::<bool>(), 1..=5),
        n_rows in 0usize..=48,
        seed in any::<u64>(),
    ) {
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        let cols: Vec<Col> = kinds
            .iter()
            .enumerate()
            .map(|(i, k)| (k.clone(), *nullable_flags.get(i).unwrap_or(&false)))
            .collect();
        let rows: Vec<Vec<Option<Scalar>>> = (0..n_rows)
            .map(|_| {
                cols.iter()
                    .map(|(kind, nullable)| {
                        // ~1 in 4 cells null for nullable columns.
                        if *nullable && next_u64(&mut state) % 4 == 0 {
                            None
                        } else {
                            Some(kind.gen_value(&mut state))
                        }
                    })
                    .collect()
            })
            .collect();
        assert_diff(&cols, &rows);
    }
}
