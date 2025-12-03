//! Test creating many JIT decoders and using them concurrently across threads

use arrow_avro::reader::jit_decoder::{JitDecoder, OutputBuffer};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::thread;

fn main() {
    println!("=== Test 1: Create 100 decoders sequentially ===");
    test_sequential_creation();

    println!("\n=== Test 2: Create decoders from multiple threads ===");
    test_concurrent_creation();

    println!("\n=== Test 3: Use single decoder from multiple threads ===");
    test_concurrent_decode_single_decoder();

    println!("\n=== Test 4: Multiple decoders used from multiple threads ===");
    test_concurrent_decode_multiple_decoders();

    println!("\n=== All tests passed! ===");
}

fn test_sequential_creation() {
    let mut decoders = Vec::new();

    for i in 0..100 {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        match JitDecoder::compile(&schema) {
            Ok(decoder) => {
                decoders.push(decoder);
            }
            Err(e) => {
                panic!("Failed to create decoder {}: {}", i, e);
            }
        }
    }

    println!("Created {} decoders successfully", decoders.len());

    // Test decoding with all decoders
    for (i, decoder) in decoders.iter().enumerate() {
        let data: Vec<u8> = vec![0]; // Single varint-encoded 0
        let mut output = vec![0u8; 8];
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
        if let Err(e) = decoder.decode_checked(&data, &buffers, 1) {
            panic!("Decoder {} failed to decode: {}", i, e);
        }
    }
    println!("All decoders decoded successfully");
}

fn test_concurrent_creation() {
    let num_threads = 8;
    let decoders_per_thread = 10;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            thread::spawn(move || {
                let mut decoders = Vec::new();
                for i in 0..decoders_per_thread {
                    let schema = Arc::new(Schema::new(vec![
                        Field::new("v", DataType::Int64, false),
                    ]));
                    match JitDecoder::compile(&schema) {
                        Ok(decoder) => {
                            decoders.push(decoder);
                        }
                        Err(e) => {
                            panic!("Thread {} failed to create decoder {}: {}", thread_id, i, e);
                        }
                    }
                }
                println!("Thread {} created {} decoders", thread_id, decoders.len());
                decoders
            })
        })
        .collect();

    let all_decoders: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let total: usize = all_decoders.iter().map(|v| v.len()).sum();
    println!("Total decoders created concurrently: {}", total);
}

fn test_concurrent_decode_single_decoder() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = Arc::new(JitDecoder::compile(&schema).expect("compile"));

    let num_threads = 8;
    let iterations_per_thread = 10000;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let decoder = Arc::clone(&decoder);
            thread::spawn(move || {
                for i in 0..iterations_per_thread {
                    // Encode value as varint
                    let value = (thread_id * iterations_per_thread + i) as i64;
                    let mut data = Vec::new();
                    encode_varint(&mut data, value);

                    let mut output = vec![0u8; 8];
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    decoder.decode_checked(&data, &buffers, 1).expect("decode");

                    // Verify the result
                    let result = i64::from_le_bytes(output[..8].try_into().unwrap());
                    assert_eq!(result, value, "Thread {} iteration {} mismatch", thread_id, i);
                }
                println!("Thread {} completed {} iterations", thread_id, iterations_per_thread);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    println!("Single decoder used from {} threads successfully", num_threads);
}

fn test_concurrent_decode_multiple_decoders() {
    // Create different decoders for different types
    let schemas: Vec<Arc<Schema>> = vec![
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)])),
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
        Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, false)])),
        Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)])),
        Arc::new(Schema::new(vec![Field::new("v", DataType::Boolean, false)])),
    ];

    let decoders: Vec<Arc<JitDecoder>> = schemas
        .iter()
        .map(|s| Arc::new(JitDecoder::compile(s).expect("compile")))
        .collect();

    let num_threads = 16;
    let iterations_per_thread = 5000;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let decoders = decoders.clone();
            thread::spawn(move || {
                for i in 0..iterations_per_thread {
                    // Pick a decoder based on iteration
                    let decoder_idx = i % decoders.len();
                    let decoder = &decoders[decoder_idx];

                    // Generate appropriate test data
                    let (data, output_size) = match decoder_idx {
                        0 => { // Int32
                            let mut d = Vec::new();
                            encode_varint(&mut d, i as i64);
                            (d, 4)
                        }
                        1 => { // Int64
                            let mut d = Vec::new();
                            encode_varint(&mut d, i as i64);
                            (d, 8)
                        }
                        2 => { // Float32
                            ((i as f32).to_le_bytes().to_vec(), 4)
                        }
                        3 => { // Float64
                            ((i as f64).to_le_bytes().to_vec(), 8)
                        }
                        4 => { // Boolean
                            (vec![if i % 2 == 0 { 1u8 } else { 0u8 }], 1)
                        }
                        _ => unreachable!(),
                    };

                    let mut output = vec![0u8; output_size];
                    let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];
                    decoder.decode_checked(&data, &buffers, 1).expect("decode");
                }
                println!("Thread {} completed {} iterations across {} decoders",
                         thread_id, iterations_per_thread, decoders.len());
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    println!("Multiple decoders used from {} threads successfully", num_threads);
}

/// Encode a varint (zig-zag + variable length)
fn encode_varint(buf: &mut Vec<u8>, value: i64) {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    let mut v = zigzag;
    loop {
        if v < 0x80 {
            buf.push(v as u8);
            break;
        }
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
}
