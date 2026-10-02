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

use super::*;

struct Fixture {
    schema: SchemaRef,
    rows: i64,
    nodes: Vec<crate::FieldNode>,
    buffers: Vec<crate::Buffer>,
    variadic: Vec<i64>,
    body: Buffer,
    dictionaries: HashMap<i64, ArrayRef>,
}

impl Fixture {
    fn primitive() -> Self {
        Self {
            schema: Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)])),
            rows: 2,
            nodes: vec![crate::FieldNode::new(2, 0)],
            buffers: vec![crate::Buffer::new(0, 0), crate::Buffer::new(0, 8)],
            variadic: vec![],
            body: Buffer::from_slice_ref([7_i32, 7]),
            dictionaries: HashMap::new(),
        }
    }

    fn decode(&self, projection: Option<&[usize]>) -> Result<RecordBatch, ArrowError> {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let nodes = builder.create_vector(&self.nodes);
        let buffers = builder.create_vector(&self.buffers);
        let variadic = builder.create_vector(&self.variadic);
        let batch = crate::RecordBatch::create(
            &mut builder,
            &crate::RecordBatchArgs {
                length: self.rows,
                nodes: Some(nodes),
                buffers: Some(buffers),
                variadicBufferCounts: Some(variadic),
                ..Default::default()
            },
        );
        builder.finish(batch, None);
        let batch = flatbuffers::root::<crate::RecordBatch>(builder.finished_data()).unwrap();
        read_record_batch(
            &self.body,
            batch,
            self.schema.clone(),
            &self.dictionaries,
            projection,
            &MetadataVersion::V5,
        )
    }
}

#[test]
fn missing_buffers_are_errors_in_full_and_projected_reads() {
    for projection in [None, Some([].as_slice())] {
        let mut fixture = Fixture::primitive();
        fixture.buffers.clear();
        assert!(fixture.decode(projection).is_err());
    }
}

#[test]
fn invalid_buffer_ranges_are_errors_in_full_and_projected_reads() {
    for (offset, length) in [(-1, 8), (0, -1), (8, 8), (i64::MAX, i64::MAX)] {
        for projection in [None, Some([].as_slice())] {
            let mut fixture = Fixture::primitive();
            fixture.buffers[1] = crate::Buffer::new(offset, length);
            assert!(
                fixture.decode(projection).is_err(),
                "offset={offset}, length={length}"
            );
        }
    }
}

#[test]
fn invalid_variadic_counts_return_errors_without_allocating_from_the_count() {
    for count in [-2, -1, i64::MAX, 3] {
        for projection in [None, Some([].as_slice())] {
            let mut fixture = Fixture::primitive();
            fixture.schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8View, true)]));
            fixture.variadic = vec![count];
            assert!(fixture.decode(projection).is_err(), "count={count}");
        }
    }
}

#[test]
fn leftover_variadic_counts_are_errors() {
    for projection in [None, Some([].as_slice())] {
        let mut fixture = Fixture::primitive();
        fixture.variadic = vec![0];
        assert!(fixture.decode(projection).is_err());
    }
}

#[test]
fn negative_row_and_field_lengths_are_errors() {
    let mut fixture = Fixture::primitive();
    fixture.rows = -1;
    assert!(fixture.decode(None).is_err());
    fixture.rows = 2;
    for (length, nulls) in [(-1, 0), (2, -1), (2, 3)] {
        for projection in [None, Some([].as_slice())] {
            fixture.nodes[0] = crate::FieldNode::new(length, nulls);
            assert!(fixture.decode(projection).is_err());
        }
    }
}

#[test]
fn short_union_buffers_return_errors() {
    for mode in [UnionMode::Sparse, UnionMode::Dense] {
        let mut fixture = Fixture::primitive();
        fixture.schema = Arc::new(Schema::new(vec![Field::new(
            "u",
            DataType::Union(
                UnionFields::new(vec![0], vec![Field::new("x", DataType::Int32, false)]),
                mode,
            ),
            false,
        )]));
        fixture.buffers[0] = crate::Buffer::new(0, 0);
        assert!(fixture.decode(None).is_err(), "short type-id buffer");
        if mode == UnionMode::Dense {
            fixture.buffers[0] = crate::Buffer::new(0, 2);
            fixture.buffers[1] = crate::Buffer::new(0, 3);
            assert!(fixture.decode(None).is_err(), "short offsets buffer");
        }
    }
}

#[test]
fn short_struct_validity_returns_error() {
    let mut fixture = Fixture::primitive();
    fixture.schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        DataType::Struct(Fields::empty()),
        true,
    )]));
    fixture.rows = 9;
    fixture.nodes = vec![crate::FieldNode::new(9, 1)];
    fixture.buffers = vec![crate::Buffer::new(0, 1)];
    assert!(fixture.decode(None).is_err());
}

#[test]
fn unsupported_layout_returns_error() {
    let mut fixture = Fixture::primitive();
    fixture.schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::ListView(Arc::new(Field::new("item", DataType::Int32, true))),
        true,
    )]));
    assert!(fixture.decode(None).is_err());
    assert!(fixture.decode(Some(&[])).is_err());
}

#[test]
fn valid_primitive_preserves_duplicates_and_shares_aligned_storage() {
    let fixture = Fixture::primitive();
    let batch = fixture.decode(None).unwrap();
    let array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(array.values().as_ref(), &[7, 7]);
    assert_eq!(array.values().inner().as_ptr(), fixture.body.as_ptr());
    let projected = fixture.decode(Some(&[])).unwrap();
    assert_eq!((projected.num_rows(), projected.num_columns()), (2, 0));
}

fn encode(
    batch: &RecordBatch,
    version: MetadataVersion,
    compression: Option<crate::CompressionType>,
) -> crate::writer::EncodedData {
    use crate::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
    let options = IpcWriteOptions::try_new(64, false, version)
        .unwrap()
        .try_with_compression(compression)
        .unwrap();
    IpcDataGenerator::default()
        .encode(
            batch,
            &mut DictionaryTracker::new(false),
            &options,
            &mut Default::default(),
        )
        .unwrap()
        .1
}

fn union_batch(dense: bool) -> RecordBatch {
    use arrow_array::builder::UnionBuilder;
    use arrow_array::types::Int32Type;
    let mut builder = if dense {
        UnionBuilder::new_dense()
    } else {
        UnionBuilder::new_sparse()
    };
    builder.append::<Int32Type>("a", 7).unwrap();
    builder.append::<Int32Type>("a", 7).unwrap();
    RecordBatch::try_from_iter(vec![
        ("union", Arc::new(builder.build().unwrap()) as ArrayRef),
        ("tail", Arc::new(Int32Array::from(vec![91, 92])) as ArrayRef),
    ])
    .unwrap()
}

#[test]
fn valid_union_projection_keeps_following_column_for_each_metadata_version() {
    for version in [MetadataVersion::V4, MetadataVersion::V5] {
        for dense in [true, false] {
            let batch = union_batch(dense);
            let encoded = encode(&batch, version, None);
            let message = crate::root_as_message(&encoded.ipc_message).unwrap();
            let body = Buffer::from(encoded.arrow_data);
            let result = read_record_batch(
                &body,
                message.header_as_record_batch().unwrap(),
                batch.schema(),
                &HashMap::new(),
                Some(&[1]),
                &version,
            )
            .unwrap();
            assert_eq!(result, batch.project(&[1]).unwrap());
        }
    }
}

#[test]
fn union_offsets_respect_alignment_policy_and_share_aligned_storage() {
    let batch = union_batch(true);
    let encoded = encode(&batch, MetadataVersion::V5, None);
    let message = crate::root_as_message(&encoded.ipc_message).unwrap();
    let header = message.header_as_record_batch().unwrap();
    let body = Buffer::from_slice_ref(&encoded.arrow_data);
    let decoded = read_record_batch(
        &body,
        header,
        batch.schema(),
        &HashMap::new(),
        None,
        &MetadataVersion::V5,
    )
    .unwrap();
    assert_eq!(decoded, batch);
    let union = decoded
        .column(0)
        .as_any()
        .downcast_ref::<UnionArray>()
        .unwrap();
    let descriptor = header.buffers().unwrap().get(1);
    assert_eq!(
        union.offsets().unwrap().inner().as_ptr(),
        body.slice(descriptor.offset() as usize).as_ptr()
    );

    let mut shifted = MutableBuffer::new(body.len() + 1);
    shifted.push(0_u8);
    shifted.extend_from_slice(&body);
    let shifted = Buffer::from(shifted).slice(1);
    let dictionaries = HashMap::new();
    for require_alignment in [false, true] {
        let result = RecordBatchDecoder::try_new(
            &shifted,
            header,
            batch.schema(),
            &dictionaries,
            &MetadataVersion::V5,
        )
        .unwrap()
        .with_require_alignment(require_alignment)
        .read_record_batch();
        if require_alignment {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap(), batch);
        }
    }
}

#[test]
fn valid_external_views_and_nullable_empty_struct_roundtrip() {
    let batch = RecordBatch::try_from_iter(vec![
        (
            "views",
            Arc::new(StringViewArray::from(vec![
                Some("a string longer than twelve bytes"),
                None,
                Some("a string longer than twelve bytes"),
            ])) as ArrayRef,
        ),
        (
            "struct",
            Arc::new(StructArray::new_empty_fields(
                3,
                Some(NullBuffer::from(vec![true, false, true])),
            )) as ArrayRef,
        ),
        (
            "tail",
            Arc::new(Int32Array::from(vec![9, 8, 9])) as ArrayRef,
        ),
    ])
    .unwrap();
    let encoded = encode(&batch, MetadataVersion::V5, None);
    let message = crate::root_as_message(&encoded.ipc_message).unwrap();
    let body = Buffer::from(encoded.arrow_data);
    for projection in [None, Some([2].as_slice())] {
        let decoded = read_record_batch(
            &body,
            message.header_as_record_batch().unwrap(),
            batch.schema(),
            &HashMap::new(),
            projection,
            &MetadataVersion::V5,
        )
        .unwrap();
        assert_eq!(
            decoded,
            if let Some(columns) = projection {
                batch.project(columns).unwrap()
            } else {
                batch.clone()
            }
        );
    }
}

#[test]
fn decoded_byte_budget_is_aggregate_and_projection_does_not_expand_skipped_buffers() {
    let batch = RecordBatch::try_from_iter(vec![
        ("a", Arc::new(Int32Array::from(vec![7; 4096])) as ArrayRef),
        ("b", Arc::new(Int32Array::from(vec![9; 4096])) as ArrayRef),
    ])
    .unwrap();
    #[allow(unused_mut)]
    let mut compressions = vec![None];
    #[cfg(feature = "lz4")]
    compressions.push(Some(crate::CompressionType::LZ4_FRAME));
    #[cfg(feature = "zstd")]
    compressions.push(Some(crate::CompressionType::ZSTD));
    for compression in compressions {
        let encoded = encode(&batch, MetadataVersion::V5, compression);
        let message = crate::root_as_message(&encoded.ipc_message).unwrap();
        let body = Buffer::from(encoded.arrow_data);
        let dictionaries = HashMap::new();
        // IPC writes one validity bit and four value bytes per row, including
        // an all-valid bitmap. The limit covers both decoded column buffers.
        let column_bytes = 4096 * 4 + 4096 / 8;
        for (limit, projection, pass) in [
            (2 * column_bytes, None, true),
            (2 * column_bytes - 1, None, false),
            (column_bytes, Some([1].as_slice()), true),
            (column_bytes - 1, Some([1].as_slice()), false),
            (0, Some([].as_slice()), true),
        ] {
            let result = RecordBatchDecoder::try_new(
                &body,
                message.header_as_record_batch().unwrap(),
                batch.schema(),
                &dictionaries,
                &MetadataVersion::V5,
            )
            .unwrap()
            .with_projection(projection)
            .with_max_decoded_bytes(limit)
            .read_record_batch();
            if pass {
                assert_eq!(
                    result.unwrap(),
                    if let Some(columns) = projection {
                        batch.project(columns).unwrap()
                    } else {
                        batch.clone()
                    }
                );
            } else {
                assert!(result.is_err(), "limit={limit}");
            }
        }
    }
}

#[test]
fn short_validity_buffers_return_errors_for_primitive_list_and_dictionary() {
    let mut fixture = Fixture::primitive();
    fixture.rows = 9;
    fixture.nodes = vec![crate::FieldNode::new(9, 1)];
    fixture.buffers = vec![crate::Buffer::new(0, 1), crate::Buffer::new(8, 36)];
    fixture.body = Buffer::from(vec![0_u8; 48]);
    assert!(fixture.decode(None).is_err());

    #[allow(deprecated)]
    let field = Field::new_dict(
        "d",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int32)),
        true,
        0,
        false,
    );
    fixture.schema = Arc::new(Schema::new(vec![field]));
    fixture
        .dictionaries
        .insert(0, Arc::new(Int32Array::from(vec![7])));
    assert!(fixture.decode(None).is_err());

    fixture.schema = Arc::new(Schema::new(vec![Field::new(
        "l",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        true,
    )]));
    fixture.nodes.push(crate::FieldNode::new(0, 0));
    fixture.buffers[1] = crate::Buffer::new(8, 40);
    fixture
        .buffers
        .extend([crate::Buffer::new(0, 0), crate::Buffer::new(0, 0)]);
    assert!(fixture.decode(None).is_err());
}

#[test]
fn struct_parent_length_must_match_its_children_even_without_nulls() {
    let mut fixture = Fixture::primitive();
    fixture.rows = 3;
    fixture.schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        DataType::Struct(vec![Field::new("x", DataType::Int32, false)].into()),
        true,
    )]));
    fixture.nodes = vec![crate::FieldNode::new(2, 0), crate::FieldNode::new(3, 0)];
    fixture.buffers = vec![
        crate::Buffer::new(0, 0),
        crate::Buffer::new(0, 0),
        crate::Buffer::new(0, 12),
    ];
    fixture.body = Buffer::from_slice_ref([1_i32, 2, 3]);
    assert!(fixture.decode(None).is_err());
}
