//! Regressions for capacity, malformed input, and buffer assembly.
use super::jit_buffer_utils::{PooledBufferSet, pooled_buffers_to_batch_owned};
use super::jit_decoder::{DecodeStatus, JitDecoder, OutputBuffer};
use arrow_array::{Array, Int64Array, StructArray};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

#[test]
fn list_respects_secondary_capacity() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).unwrap();
    let mut offsets = [0u8; 8];
    // Physical guard space makes an out-of-contract write observable without heap UB.
    let mut data = [0xa5u8; 1024];
    let outputs = [
        OutputBuffer::new(offsets.as_mut_ptr(), 8),
        OutputBuffer::new(data.as_mut_ptr(), 64),
    ];
    let mut input = vec![200, 1];
    input.extend([2; 100]);
    input.push(0);
    let result = unsafe { decoder.decode_chunked(&input, &outputs, 1) };
    assert_eq!(result.status, DecodeStatus::OutputOverflow);
    assert!(data[64..].iter().all(|b| *b == 0xa5));
}

#[test]
fn validity_capacity_is_checked() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
    let decoder = JitDecoder::compile(&schema).unwrap();
    let mut data = [0u8; 72];
    let mut validity = [0u8; 16];
    let outputs = [
        OutputBuffer::new(data.as_mut_ptr(), 72),
        OutputBuffer::new(validity.as_mut_ptr(), 1),
    ];
    assert!(unsafe { decoder.decode_checked(&[2, 2].repeat(9), &outputs, 9) }.is_err());
    assert_eq!(validity[1], 0);
}

#[test]
fn partial_conversion_honors_row_count() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let mut buffers = PooledBufferSet::new_fixed(24);
    buffers.primary[..8].copy_from_slice(&42i64.to_ne_bytes());
    let batch = pooled_buffers_to_batch_owned(&schema, vec![buffers], 1).unwrap();
    assert_eq!(batch.num_rows(), 1);
}

#[test]
fn nullable_struct_keeps_row_stride() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Struct(vec![Field::new("x", DataType::Int64, false)].into()),
        true,
    )]));
    let decoder = JitDecoder::compile(&schema).unwrap();
    let mut b = PooledBufferSet::new_fixed(16).with_validity(1);
    let outputs = [
        OutputBuffer::new(b.primary_ptr(), 16),
        OutputBuffer::new(b.validity_ptr(), 1),
    ];
    unsafe { decoder.decode_checked(&[0, 2, 84], &outputs, 2) }.unwrap();
    let batch = pooled_buffers_to_batch_owned(&schema, vec![b], 2).unwrap();
    let a = batch
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert!(a.is_null(0));
    assert_eq!(
        a.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(1),
        42
    );
}

#[test]
fn malformed_scalar_encodings_are_rejected() {
    for (dt, nullable, bytes) in [
        (DataType::Int64, false, [vec![128u8; 10], vec![2]].concat()),
        (DataType::Int64, false, [vec![255u8; 9], vec![2]].concat()),
        (DataType::Int32, false, vec![128, 128, 128, 128, 32]),
        (DataType::Int64, true, vec![4, 84]),
        (DataType::Boolean, false, vec![255]),
    ] {
        let schema = Arc::new(Schema::new(vec![Field::new("v", dt.clone(), nullable)]));
        let decoder = JitDecoder::compile(&schema).unwrap();
        let mut data = [0u8; 8];
        let mut validity = [0u8; 1];
        let mut outputs = vec![OutputBuffer::new(data.as_mut_ptr(), 8)];
        if nullable {
            outputs.push(OutputBuffer::new(validity.as_mut_ptr(), 1));
        }
        assert!(
            unsafe { decoder.decode_checked(&bytes, &outputs, 1) }.is_err(),
            "accepted {dt:?}: {bytes:?}"
        );
    }
}

#[test]
fn invalid_utf8_returns_error_without_panicking() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)]));
    let mut b = PooledBufferSet::new_variable(8, 1);
    b.primary[4..8].copy_from_slice(&1i32.to_ne_bytes());
    b.secondary.as_mut().unwrap()[0] = 255;
    let result = std::panic::catch_unwind(|| pooled_buffers_to_batch_owned(&schema, vec![b], 1));
    assert!(result.is_ok(), "malformed input must not panic");
    assert!(result.unwrap().is_err());
}

#[test]
fn unsupported_layouts_fail_before_native_compilation() {
    use arrow_schema::{UnionFields, UnionMode};
    for dt in [
        DataType::LargeUtf8,
        DataType::LargeList(Arc::new(Field::new("item", DataType::Int64, false))),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int64, false)), 2),
        DataType::Union(
            UnionFields::new(
                vec![0, 1],
                vec![
                    Field::new("i", DataType::Int32, false),
                    Field::new("s", DataType::Utf8, false),
                ],
            ),
            UnionMode::Dense,
        ),
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8, false),
                        Field::new("value", DataType::Int64, false),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        ),
    ] {
        let schema = Arc::new(Schema::new(vec![Field::new("v", dt.clone(), false)]));
        assert!(
            JitDecoder::compile(&schema).is_err(),
            "accepted unsupported {dt:?}"
        );
    }
}

#[test]
fn reusing_validity_buffers_clears_previous_values() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
    let decoder = JitDecoder::compile(&schema).unwrap();
    let mut b = PooledBufferSet::new_fixed(8).with_validity(1);
    let out = [
        OutputBuffer::new(b.primary_ptr(), 8),
        OutputBuffer::new(b.validity_ptr(), 1),
    ];
    unsafe { decoder.decode_checked(&[2, 84], &out, 1) }.unwrap();
    unsafe { decoder.decode_checked(&[0], &out, 1) }.unwrap();
    let batch = pooled_buffers_to_batch_owned(&schema, vec![b], 1).unwrap();
    assert!(batch.column(0).is_null(0));
}

#[test]
fn nullable_list_offsets_include_null_rows() {
    let decoder=JitDecoder::compile_avro(r#"{"type":"record","name":"R","fields":[{"name":"v","type":["null",{"type":"array","items":"long"}]}]}"#).unwrap();
    let mut b = PooledBufferSet::new_variable(12, 8).with_validity(1);
    let out = [
        OutputBuffer::new(b.primary_ptr(), 12),
        OutputBuffer::new(b.secondary_ptr(), 8),
        OutputBuffer::new(b.validity_ptr(), 1),
    ];
    unsafe { decoder.decode_checked(&[0, 2, 2, 84, 0], &out, 2) }.unwrap();
    let batch = pooled_buffers_to_batch_owned(decoder.schema(), vec![b], 2).unwrap();
    let list = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::ListArray>()
        .unwrap();
    assert!(list.is_null(0));
    assert_eq!(
        list.value(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        42
    );
}

#[test]
fn negative_list_block_size_is_validated() {
    let decoder=JitDecoder::compile_avro(r#"{"type":"record","name":"R","fields":[{"name":"v","type":{"type":"array","items":"long"}}]}"#).unwrap();
    let mut b = PooledBufferSet::new_variable(8, 8);
    let out = [
        OutputBuffer::new(b.primary_ptr(), 8),
        OutputBuffer::new(b.secondary_ptr(), 8),
    ];
    // count=-1, declared byte size=0, one long, terminator: size is inconsistent.
    assert!(unsafe { decoder.decode_checked(&[1, 0, 84, 0], &out, 1) }.is_err());
    assert!(unsafe { decoder.decode_checked(&[1, 2, 84, 0], &out, 1) }.is_ok());
}

#[test]
fn enum_indices_cannot_wrap_to_valid_symbols() {
    let decoder=JitDecoder::compile_avro(r#"{"type":"record","name":"R","fields":[{"name":"v","type":{"type":"enum","name":"E","symbols":["A"]}}]}"#).unwrap();
    let mut b = PooledBufferSet::new_fixed(4);
    let out = [OutputBuffer::new(b.primary_ptr(), 4)];
    assert!(unsafe { decoder.decode_checked(&[128, 128, 128, 128, 32], &out, 1) }.is_err());
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(512))]
    #[test]
    fn arbitrary_payloads_never_write_past_advertised_capacity(
        kind in 0u8..5, nullable in proptest::bool::ANY,
        input in proptest::collection::vec(proptest::num::u8::ANY, 0..128),
        rows in 0usize..16, capacity in 0usize..65,
    ) {
        let dt=match kind {
            0=>DataType::Int64, 1=>DataType::Boolean, 2=>DataType::Utf8,
            3=>DataType::List(Arc::new(Field::new("item",DataType::Int64,false))),
            _=>DataType::Struct(vec![Field::new("x",DataType::Int64,false),Field::new("y",DataType::Boolean,false)].into()),
        };
        let schema=Arc::new(Schema::new(vec![Field::new("v",dt,nullable)]));
        let decoder=JitDecoder::compile(&schema).unwrap();
        let mut buffers:Vec<_>=(0..decoder.total_buffers()).map(|_|vec![0xa5u8;512]).collect();
        let outputs:Vec<_>=buffers.iter_mut().map(|b|OutputBuffer::new(b.as_mut_ptr(),capacity)).collect();
        // The guard allocations are larger than their advertised capacity, so a
        // regression is a deterministic assertion, not undefined heap access.
        let result=unsafe { decoder.decode_chunked(&input,&outputs,rows) };
        proptest::prop_assert!(result.bytes_consumed<=input.len());
        proptest::prop_assert!(result.rows_decoded<=rows);
        for b in buffers { proptest::prop_assert!(b[capacity..].iter().all(|b|*b==0xa5)); }
    }
}

#[test]
fn avro_compilation_keeps_supported_decimal_output_width() {
    let decoder = JitDecoder::compile_avro(r#"{"type":"record","name":"R","fields":[{"name":"v","type":{"type":"bytes","logicalType":"decimal","precision":8,"scale":2}}]}"#).unwrap();
    assert_eq!(
        decoder.schema().field(0).data_type(),
        &DataType::Decimal128(8, 2)
    );
}

#[test]
fn required_capacity_overflow_is_reported() {
    let decoder = JitDecoder::compile(&Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )])))
    .unwrap();
    assert_eq!(decoder.required_capacity(0, usize::MAX), None);
}

#[test]
fn schemas_without_a_supported_row_layout_are_rejected() {
    for schema in [
        Schema::empty(),
        Schema::new(vec![Field::new("v", DataType::Null, false)]),
    ] {
        assert!(JitDecoder::compile(&Arc::new(schema)).is_err());
    }
}
