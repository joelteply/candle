// What this catches: disabling the optional archive must leave callable APIs
// with an explicit feature error, rather than dangling FFI link references.
#[cfg(not(feature = "fused-moe"))]
#[test]
fn missing_fused_moe_is_explicit() -> candle::Result<()> {
    let input = candle::Tensor::new(&[1f32], &candle::Device::Cpu)?;
    let ids = candle::Tensor::new(&[0u32], &candle::Device::Cpu)?;
    let error = candle_nn::moe::moe_gemm(&input, &input, &None, &ids, &ids, 1, false)
        .expect_err("specialized MoE cannot execute without its feature");
    assert!(error.to_string().contains("requires the fused-moe feature"));
    Ok(())
}
