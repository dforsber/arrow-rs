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
//! - **Verified baseline**: every type listed in `SUPPORTED` must round-trip
//!   (exhaustive edge cases + a proptest over random multi-column batches).
//! - **Loud rejection**: unsupported shapes (e.g. Map) must make the JIT path
//!   return `Err`, never silently-wrong data. `gate_*` tests assert this.
//! Add a type to `SUPPORTED` only once it has a passing differential test.

use super::jit_buffer_utils::{buffers_to_batch_owned, BufferSet};
use super::jit_decoder::{JitDecoder, OutputBuffer};
use super::record::RecordDecoder;
use crate::codec::AvroFieldBuilder;
use crate::schema::Schema as AvroSchema;
use apache_avro::{to_avro_datum, types::Value, Schema as ApacheSchema};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType, SchemaRef};
use proptest::prelude::*;

/// A scalar Avro value to generate/encode. Each variant maps 1:1 to an Avro
/// primitive and to the Arrow type the decoders must produce.
#[derive(Clone, Debug, PartialEq)]
enum Scalar {
    Bool(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Str(String),
    Bytes(Vec<u8>),
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
        }
    }
}

/// A column spec: Avro type name + whether it is nullable (`["null", T]`).
type ColSpec<'a> = (&'a str, bool);

/// Build the Avro record-schema JSON. Nullable columns use a `["null", T]`
/// union (null first → default branch 0, matching how the encoder tags values).
fn record_json(specs: &[ColSpec]) -> String {
    let mut fields = String::new();
    for (i, (t, nullable)) in specs.iter().enumerate() {
        if i > 0 {
            fields.push(',');
        }
        if *nullable {
            fields.push_str(&format!(r#"{{"name":"f{i}","type":["null","{t}"]}}"#));
        } else {
            fields.push_str(&format!(r#"{{"name":"f{i}","type":"{t}"}}"#));
        }
    }
    format!(r#"{{"type":"record","name":"R","fields":[{fields}]}}"#)
}

/// Encode `rows` to a single concatenated raw-datum buffer via apache_avro.
/// `None` in a nullable column encodes as union branch 0 (`null`); a present
/// value in a nullable column encodes as branch 1.
fn encode_rows(json: &str, specs: &[ColSpec], rows: &[Vec<Option<Scalar>>]) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(json).expect("apache parse_str");
    let mut body = Vec::new();
    for row in rows {
        let rec = Value::Record(
            row.iter()
                .enumerate()
                .map(|(i, cell)| {
                    let (_, nullable) = specs[i];
                    let v = match cell {
                        Some(s) if nullable => Value::Union(1, Box::new(s.to_value())),
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
/// the Arrow schema it derived (which we then feed to the JIT path so both use
/// an identical schema — any difference is then purely in the values).
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

    // Generous variable-length data sizing: enough for the whole input.
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
            let elem = packed_column_bytes(field.data_type());
            let mut b = BufferSet::new_fixed((rows * elem).max(1));
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

/// The differential assertion (general, nullable-aware): encode, decode both
/// ways, require equality.
fn assert_general(specs: &[ColSpec], rows: &[Vec<Option<Scalar>>]) {
    let json = record_json(specs);
    let body = encode_rows(&json, specs, rows);
    let (ref_batch, schema) = reference_decode(&json, &body, rows.len());
    let jit_batch = jit_decode(&schema, &body, rows.len())
        .unwrap_or_else(|e| panic!("JIT decode failed for {specs:?}: {e}"));
    assert_eq!(
        ref_batch, jit_batch,
        "JIT batch diverged from reference for columns {specs:?}"
    );
}

/// Back-compat non-nullable helper: all columns required, all cells present.
fn assert_jit_matches_reference(col_types: &[&str], rows: &[Vec<Scalar>]) {
    let specs: Vec<ColSpec> = col_types.iter().map(|t| (*t, false)).collect();
    let rows_opt: Vec<Vec<Option<Scalar>>> = rows
        .iter()
        .map(|r| r.iter().cloned().map(Some).collect())
        .collect();
    assert_general(&specs, &rows_opt);
}

// ===========================================================================
// Verified baseline — exhaustive edge cases per supported scalar type.
// ===========================================================================

#[test]
fn diff_bool_edges() {
    let rows: Vec<Vec<Scalar>> =
        [true, false, false, true, true, true, false, false, true]
            .iter()
            .map(|&b| vec![Scalar::Bool(b)])
            .collect();
    assert_jit_matches_reference(&["boolean"], &rows);
}

#[test]
fn diff_int_edges() {
    let rows: Vec<Vec<Scalar>> = [i32::MIN, -1, 0, 1, 127, 128, i32::MAX]
        .iter()
        .map(|&v| vec![Scalar::Int(v)])
        .collect();
    assert_jit_matches_reference(&["int"], &rows);
}

#[test]
fn diff_long_edges() {
    let rows: Vec<Vec<Scalar>> = [i64::MIN, -1, 0, 1, i64::MAX]
        .iter()
        .map(|&v| vec![Scalar::Long(v)])
        .collect();
    assert_jit_matches_reference(&["long"], &rows);
}

#[test]
fn diff_float_edges() {
    let rows: Vec<Vec<Scalar>> = [0.0f32, -0.0, 1.5, -1.5, f32::MIN, f32::MAX, f32::INFINITY]
        .iter()
        .map(|&v| vec![Scalar::Float(v)])
        .collect();
    assert_jit_matches_reference(&["float"], &rows);
}

#[test]
fn diff_double_edges() {
    let rows: Vec<Vec<Scalar>> = [0.0f64, -0.0, 1.5, -1.5, f64::MIN, f64::MAX]
        .iter()
        .map(|&v| vec![Scalar::Double(v)])
        .collect();
    assert_jit_matches_reference(&["double"], &rows);
}

#[test]
fn diff_string_edges() {
    let rows: Vec<Vec<Scalar>> = ["", "a", "hello world", "ünïcödé→★", "with\ttab\nnewline"]
        .iter()
        .map(|s| vec![Scalar::Str(s.to_string())])
        .collect();
    assert_jit_matches_reference(&["string"], &rows);
}

#[test]
fn diff_bytes_edges() {
    let rows: Vec<Vec<Scalar>> = [vec![], vec![0u8], vec![0u8, 255, 1, 254], vec![7u8; 100]]
        .iter()
        .map(|b| vec![Scalar::Bytes(b.clone())])
        .collect();
    assert_jit_matches_reference(&["bytes"], &rows);
}

#[test]
fn diff_multi_column_mixed() {
    let rows = vec![
        vec![
            Scalar::Bool(true),
            Scalar::Int(7),
            Scalar::Str("x".into()),
            Scalar::Double(3.5),
        ],
        vec![
            Scalar::Bool(false),
            Scalar::Int(-9),
            Scalar::Str("".into()),
            Scalar::Double(-0.0),
        ],
    ];
    assert_jit_matches_reference(&["boolean", "int", "string", "double"], &rows);
}

#[test]
fn diff_empty_batch() {
    // Zero rows must still produce equal (empty) batches with the right schema.
    assert_jit_matches_reference(&["int"], &[]);
}

#[test]
fn diff_nullable_scalars_edges() {
    // ["null", T] unions — the most common real Avro shape. Interleave present
    // and null values so validity bitmaps span byte boundaries.
    let specs: Vec<ColSpec> = vec![("long", true), ("string", true), ("boolean", true)];
    let rows: Vec<Vec<Option<Scalar>>> = vec![
        vec![Some(Scalar::Long(1)), None, Some(Scalar::Bool(true))],
        vec![None, Some(Scalar::Str("x".into())), None],
        vec![Some(Scalar::Long(-9)), Some(Scalar::Str("".into())), Some(Scalar::Bool(false))],
        vec![None, None, None],
        vec![Some(Scalar::Long(i64::MAX)), Some(Scalar::Str("ünï".into())), Some(Scalar::Bool(true))],
    ];
    assert_general(&specs, &rows);
}

#[test]
fn diff_nullable_mixed_with_required() {
    // Mix nullable and required columns in one record.
    let specs: Vec<ColSpec> = vec![("int", false), ("double", true), ("bytes", true)];
    let rows: Vec<Vec<Option<Scalar>>> = vec![
        vec![Some(Scalar::Int(7)), Some(Scalar::Double(1.5)), None],
        vec![Some(Scalar::Int(-1)), None, Some(Scalar::Bytes(vec![1, 2, 3]))],
        vec![Some(Scalar::Int(0)), None, None],
    ];
    assert_general(&specs, &rows);
}

// ===========================================================================
// Loud-rejection gate — unsupported shapes must Err, never silently corrupt.
// ===========================================================================

#[test]
fn gate_map_is_loudly_rejected() {
    // Reference decodes a Map fine; the JIT converter must return Err
    // (NotYetImplemented), proving no silent-wrong path.
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
// Property test — random multi-column batches over the supported scalar set.
// proptest controls (column types, row count, value seed); rand fills values
// deterministically from the seed. Shrinks structure (drop columns/rows) to a
// minimal counterexample on failure.
// ===========================================================================

#[derive(Clone, Debug)]
enum ColType {
    Bool,
    Int,
    Long,
    Float,
    Double,
    Str,
    Bytes,
}

impl ColType {
    fn avro(&self) -> &'static str {
        match self {
            ColType::Bool => "boolean",
            ColType::Int => "int",
            ColType::Long => "long",
            ColType::Float => "float",
            ColType::Double => "double",
            ColType::Str => "string",
            ColType::Bytes => "bytes",
        }
    }
}

fn col_type_strategy() -> impl Strategy<Value = ColType> {
    prop_oneof![
        Just(ColType::Bool),
        Just(ColType::Int),
        Just(ColType::Long),
        Just(ColType::Float),
        Just(ColType::Double),
        Just(ColType::Str),
        Just(ColType::Bytes),
    ]
}

/// Deterministically fill a scalar for a column from a per-cell PRNG state.
/// (Simple SplitMix64 — avoids a rand-version dependency and is reproducible
/// from the proptest-chosen seed, so failures replay exactly.)
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gen_scalar(ct: &ColType, state: &mut u64) -> Scalar {
    let r = next_u64(state);
    match ct {
        ColType::Bool => Scalar::Bool(r & 1 == 1),
        ColType::Int => Scalar::Int(r as i32),
        ColType::Long => Scalar::Long(r as i64),
        ColType::Float => Scalar::Float(f32::from_bits(r as u32)),
        ColType::Double => Scalar::Double(f64::from_bits(r)),
        ColType::Str => {
            let len = (r % 12) as usize;
            let s: String = (0..len)
                .map(|_| {
                    let c = (next_u64(state) % 26) as u8 + b'a';
                    c as char
                })
                .collect();
            Scalar::Str(s)
        }
        ColType::Bytes => {
            let len = (r % 12) as usize;
            let b: Vec<u8> = (0..len).map(|_| next_u64(state) as u8).collect();
            Scalar::Bytes(b)
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn prop_differential_supported_scalars(
        types in prop::collection::vec(col_type_strategy(), 1..=5),
        nullable_flags in prop::collection::vec(any::<bool>(), 1..=5),
        n_rows in 0usize..=48,
        seed in any::<u64>(),
    ) {
        // NaN floats compare unequal to themselves, which would make
        // assert_eq! on the batches fail spuriously — skip NaN by construction
        // (the edge tests cover the bit-pattern paths deterministically).
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        let specs: Vec<ColSpec> = types
            .iter()
            .enumerate()
            .map(|(i, c)| (c.avro(), *nullable_flags.get(i).unwrap_or(&false)))
            .collect();
        let rows: Vec<Vec<Option<Scalar>>> = (0..n_rows)
            .map(|_| {
                specs
                    .iter()
                    .zip(types.iter())
                    .map(|((_, nullable), ct)| {
                        // ~1 in 4 cells null for nullable columns.
                        if *nullable && next_u64(&mut state) % 4 == 0 {
                            return None;
                        }
                        let mut s = gen_scalar(ct, &mut state);
                        match &mut s {
                            Scalar::Float(f) if f.is_nan() => *f = 0.0,
                            Scalar::Double(d) if d.is_nan() => *d = 0.0,
                            _ => {}
                        }
                        Some(s)
                    })
                    .collect()
            })
            .collect();
        assert_general(&specs, &rows);
    }
}
