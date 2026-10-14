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

//! Recursive-record safety tests. Apache Avro owns wire encoding; the separate
//! RecordDecoder is the value oracle. Both owned and pooled assembly are checked.
use super::*;
use crate::reader::jit_buffer_utils::{
    PooledBufferSet, buffers_to_batch, pooled_buffers_to_batch_owned,
};
use crate::reader::jit_decoder::{ColumnType, DecodeStatus};

const RICH: &str = r#"{"type":"record","name":"Root","fields":[
 {"name":"record","type":["null",{"type":"record","name":"Outer","fields":[
  {"name":"packed","type":{"type":"record","name":"Packed","fields":[{"name":"id","type":"long"},{"name":"ok","type":"boolean"}]}},
  {"name":"recursive","type":{"type":"record","name":"Rich","fields":[
   {"name":"label","type":["string","null"]},
   {"name":"blob","type":["null","bytes"]},
   {"name":"tag","type":{"type":"fixed","name":"Tag4","size":4}},
   {"name":"list","type":[{"type":"array","items":"long"},"null"]}
  ]}}
 ]}]},
 {"name":"after","type":"long"}
]}"#;

fn union(value: Option<Value>, null_first: bool) -> Value {
    let index = if value.is_none() {
        usize::from(!null_first)
    } else {
        usize::from(null_first)
    };
    Value::Union(index as u32, Box::new(value.unwrap_or(Value::Null)))
}

fn rich_rows(rows: usize, nulls: bool) -> (Vec<u8>, Vec<usize>) {
    let schema = ApacheSchema::parse_str(RICH).unwrap();
    let mut input = Vec::new();
    let mut ends = Vec::new();
    for row in 0..rows {
        let record = Value::Record(vec![
            (
                "packed".into(),
                Value::Record(vec![
                    ("id".into(), Value::Long(-100 - row as i64)),
                    ("ok".into(), Value::Boolean(row % 2 == 0)),
                ]),
            ),
            (
                "recursive".into(),
                Value::Record(vec![
                    (
                        "label".into(),
                        union(
                            (!nulls || row % 3 != 1)
                                .then(|| Value::String(format!("row-{row:02}-λ"))),
                            false,
                        ),
                    ),
                    (
                        "blob".into(),
                        union(
                            (!nulls || row % 4 != 1).then(|| Value::Bytes(vec![row as u8, 0, 255])),
                            true,
                        ),
                    ),
                    ("tag".into(), Value::Fixed(4, vec![0, row as u8, 254, 255])),
                    (
                        "list".into(),
                        union(
                            (!nulls || row % 3 != 2).then(|| {
                                Value::Array(vec![Value::Long(row as i64), Value::Long(-9)])
                            }),
                            false,
                        ),
                    ),
                ]),
            ),
        ]);
        let record = union((!nulls || row % 5 != 0).then_some(record), true);
        let datum = Value::Record(vec![
            ("record".into(), record),
            ("after".into(), Value::Long(row as i64 + 500)),
        ]);
        input.extend(to_avro_datum(&schema, datum).unwrap());
        ends.push(input.len());
    }
    (input, ends)
}

fn schema() -> SchemaRef {
    JitDecoder::compile_avro(RICH).unwrap().schema().clone()
}

fn make_outputs(
    decoder: &JitDecoder,
    rows: usize,
    var: usize,
) -> (Vec<BufferSet>, Vec<OutputBuffer>) {
    let mut sets: Vec<_> = (0..decoder.num_columns())
        .map(|i| BufferSet::for_slot(decoder.buffer_slot(i).unwrap(), rows, var))
        .collect();
    let mut output = Vec::new();
    for set in &mut sets {
        set.append_output_buffers(&mut output);
    }
    (sets, output)
}

#[test]
fn rich_record_owned_pooled_and_parent_null_children_match_reference() {
    use arrow_array::{Array, StructArray};
    let (input, ends) = rich_rows(23, true);
    let (expected, schema) = reference_decode(RICH, &input, ends.len());
    let decoder = JitDecoder::compile(&schema).unwrap();
    assert_eq!(decoder.column_type(0), Some(ColumnType::Record));
    assert_eq!(decoder.required_capacity(0, 23), None);
    assert!(decoder.has_variable_columns());
    let (sets, output) = make_outputs(&decoder, 23, 4096);
    unsafe { decoder.decode_records_checked(&input, &output, &ends) }.unwrap();
    let actual = buffers_to_batch_owned(&schema, sets, 23).unwrap();
    assert_eq!(expected, actual);
    let outer = actual
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    for child in outer.columns() {
        assert_eq!(child.len(), 23);
    }
    let rich = outer
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    for child in rich.columns() {
        assert_eq!(child.len(), 23);
    }
    // Nonnullable descendants under null parents have initialized placeholders;
    // their nullable grandchildren are null, not phantom nonnull values.
    for row in (0..23).step_by(5) {
        assert!(outer.is_null(row));
        assert!(rich.column(0).is_null(row));
        assert!(rich.column(1).is_null(row));
        assert!(rich.column(3).is_null(row));
    }
    let mut pooled: Vec<_> = (0..decoder.num_columns())
        .map(|i| PooledBufferSet::for_slot(decoder.buffer_slot(i).unwrap(), 23, 4096))
        .collect();
    let mut output = Vec::new();
    for set in &mut pooled {
        set.append_output_buffers(&mut output);
    }
    unsafe { decoder.decode_records_checked(&input, &output, &ends) }.unwrap();
    let pooled = pooled_buffers_to_batch_owned(&schema, pooled, 23).unwrap();
    assert_eq!(expected, pooled);
    drop(decoder); // Arrays own every child allocation independently of native code.
    assert_eq!(actual, pooled);
}

#[test]
fn every_recursive_buffer_rejects_short_capacity_without_guard_writes() {
    let (input, ends) = rich_rows(1, false);
    let schema = schema();
    let decoder = JitDecoder::compile(&schema).unwrap();
    let mut capacities = vec![0; decoder.total_buffers()];
    decoder.visit_buffer_slots(&mut |slot| {
        capacities[slot.primary] = slot.primary_capacity(1).unwrap();
        if let Some(index) = slot.secondary {
            capacities[index] = 128;
        }
        if let Some(index) = slot.validity {
            capacities[index] = 1;
        }
    });
    for deficient in 0..capacities.len() {
        if capacities[deficient] == 0 {
            continue;
        } // Container has no data writes.
        let mut caps = capacities.clone();
        caps[deficient] = 0;
        let mut guarded: Vec<_> = caps.iter().map(|&cap| vec![0xa5; cap + 32]).collect();
        let outputs: Vec<_> = guarded
            .iter_mut()
            .zip(&caps)
            .map(|(bytes, &cap)| OutputBuffer::new(unsafe { bytes.as_mut_ptr().add(16) }, cap))
            .collect();
        assert!(
            unsafe { decoder.decode_records_checked(&input, &outputs, &ends) }.is_err(),
            "buffer {deficient}"
        );
        for (index, (bytes, &cap)) in guarded.iter().zip(&caps).enumerate() {
            assert_eq!(&bytes[..16], &[0xa5; 16], "prefix of buffer {index}");
            assert_eq!(
                &bytes[cap + 16..],
                &[0xa5; 16],
                "suffix of buffer {index}, deficient {deficient}"
            );
        }
    }
}

#[test]
fn every_nested_truncation_and_frame_mismatch_is_rejected() {
    let (input, _) = rich_rows(1, false);
    let schema = schema();
    let decoder = JitDecoder::compile(&schema).unwrap();
    let (_sets, outputs) = make_outputs(&decoder, 1, 4096);
    for cut in 0..input.len() {
        assert!(
            unsafe { decoder.decode_records_checked(&input[..cut], &outputs, &[cut]) }.is_err(),
            "cut {cut}"
        );
    }
    let mut trailing = input.clone();
    trailing.push(0);
    assert!(
        unsafe { decoder.decode_records_checked(&trailing, &outputs, &[trailing.len()]) }.is_err()
    );
    let (input, ends) = rich_rows(2, false);
    let (_sets, outputs) = make_outputs(&decoder, 2, 4096);
    for delta in [-1isize, 1] {
        assert!(
            unsafe {
                decoder.decode_records_checked(
                    &input,
                    &outputs,
                    &[(ends[0] as isize + delta) as usize, ends[1]],
                )
            }
            .is_err()
        );
    }
    let mut invalid = input.clone();
    invalid[0] = 4; // Third branch is not in parent union.
    assert!(unsafe { decoder.decode_records_checked(&invalid, &outputs, &ends) }.is_err());
    // Nonnullable packed child writes its long followed by one boolean byte.
    // Apache's first row long -100 occupies two bytes, after the parent's tag.
    let mut invalid = input;
    invalid[3] = 2;
    assert!(unsafe { decoder.decode_records_checked(&invalid, &outputs, &ends) }.is_err());
}

#[test]
fn recursive_overflow_rerenders_complete_prefix_and_continues_exactly() {
    let (input, ends) = rich_rows(12, false);
    let (expected, schema) = reference_decode(RICH, &input, 12);
    let decoder = JitDecoder::compile(&schema).unwrap();
    // Each list is 16 output bytes; this capacity permits exactly three rows.
    // A failed fourth row has already written other descendants when it stops.
    let (sets, outputs) = make_outputs(&decoder, 12, 48);
    let mut offset = 0;
    let mut row = 0;
    while row < 12 {
        let result = unsafe { decoder.decode_chunked(&input[offset..], &outputs, 12 - row) };
        assert_eq!(result.rows_decoded, 3);
        assert_eq!(result.bytes_consumed, ends[row + 2] - offset);
        assert_eq!(
            result.status,
            if row == 9 {
                DecodeStatus::Ok
            } else {
                DecodeStatus::OutputOverflow
            }
        );
        let actual = buffers_to_batch(&schema, &sets, 3).unwrap();
        assert_eq!(actual, expected.slice(row, 3));
        row += result.rows_decoded;
        offset += result.bytes_consumed;
    }
    assert_eq!(offset, input.len());
}

#[test]
fn nested_empty_records_preserve_explicit_lengths() {
    use arrow_array::{Array, StructArray};
    let json = r#"{"type":"record","name":"R","fields":[{"name":"s","type":["null",{"type":"record","name":"S","fields":[{"name":"empty","type":{"type":"record","name":"Empty","fields":[]}}]}]}]}"#;
    let avro = ApacheSchema::parse_str(json).unwrap();
    let decoder = JitDecoder::compile_avro(json).unwrap();
    let mut input = Vec::new();
    let mut ends = Vec::new();
    for row in 0..11 {
        let record =
            (row % 3 != 0).then(|| Value::Record(vec![("empty".into(), Value::Record(vec![]))]));
        input.extend(
            to_avro_datum(
                &avro,
                Value::Record(vec![("s".into(), union(record, true))]),
            )
            .unwrap(),
        );
        ends.push(input.len());
    }
    let (sets, output) = make_outputs(&decoder, 11, 0);
    unsafe { decoder.decode_records_checked(&input, &output, &ends) }.unwrap();
    let batch = buffers_to_batch_owned(decoder.schema(), sets, 11).unwrap();
    let outer = batch
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert_eq!(outer.len(), 11);
    assert_eq!(outer.column(0).len(), 11);
    for row in 0..11 {
        assert_eq!(outer.is_null(row), row % 3 == 0);
    }
    let mut pooled = vec![PooledBufferSet::for_slot(
        decoder.buffer_slot(0).unwrap(),
        11,
        0,
    )];
    let mut outputs = Vec::new();
    pooled[0].append_output_buffers(&mut outputs);
    unsafe { decoder.decode_records_checked(&input, &outputs, &ends) }.unwrap();
    assert_eq!(
        batch,
        pooled_buffers_to_batch_owned(decoder.schema(), pooled, 11).unwrap()
    );
}

#[test]
fn recursive_compilation_enforces_depth_and_field_limits() {
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    fn nested(depth: usize) -> SchemaRef {
        let mut kind = DataType::Int64;
        for _ in 0..depth {
            kind = DataType::Struct(vec![Field::new("child", kind, true)].into());
        }
        Arc::new(Schema::new(vec![Field::new("root", kind, true)]))
    }
    // Exercise the accepted boundary, including 32 recursive null branches.
    assert!(JitDecoder::compile(&nested(32)).is_ok());
    let too_deep = match JitDecoder::compile(&nested(33)) {
        Err(error) => error,
        Ok(_) => panic!("over-depth schema compiled"),
    };
    assert_eq!(too_deep, "JIT schema complexity limit exceeded");
    let children = (0..1024)
        .map(|i| Field::new(format!("f{i}"), DataType::Int64, true))
        .collect::<Vec<_>>();
    let wide = Arc::new(Schema::new(vec![Field::new(
        "root",
        DataType::Struct(children.into()),
        true,
    )]));
    let too_wide = match JitDecoder::compile(&wide) {
        Err(error) => error,
        Ok(_) => panic!("over-field-limit schema compiled"),
    };
    assert_eq!(too_wide, "JIT schema complexity limit exceeded");
}

#[test]
fn recursive_validity_is_reset_when_buffers_are_reused() {
    let (nonnull, nonnull_ends) = rich_rows(23, false);
    let (mixed, ends) = rich_rows(23, true);
    let (expected, schema) = reference_decode(RICH, &mixed, 23);
    let decoder = JitDecoder::compile(&schema).unwrap();
    let (sets, outputs) = make_outputs(&decoder, 23, 4096);
    unsafe { decoder.decode_records_checked(&nonnull, &outputs, &nonnull_ends) }.unwrap();
    unsafe { decoder.decode_records_checked(&mixed, &outputs, &ends) }.unwrap();
    assert_eq!(expected, buffers_to_batch_owned(&schema, sets, 23).unwrap());
}
