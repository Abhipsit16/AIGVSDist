//! Runtime dispatch once per index; portable scalar path on other CPUs.
pub type Dot = fn(&[f32], &[f32]) -> f32;

pub fn select_dot() -> Dot {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |a, b| {
            assert_eq!(a.len(), b.len());
            // Safe: feature check above and bounded loads including the tail.
            unsafe { dot_avx2(a, b) }
        };
    }
    scalar_dot
}

pub fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sums = [0.0; 4];
    let end = a.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        for j in 0..4 {
            sums[j] += a[i + j] * b[i + j];
        }
    }
    let mut total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for i in end..a.len() {
        total += a[i] * b[i];
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut x = _mm256_setzero_ps();
    let mut y = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        x = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            x,
        );
        y = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            y,
        );
        i += 16;
    }
    if i + 8 <= a.len() {
        x = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            x,
        );
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(x, y));
    let mut total: f32 = lanes.iter().sum();
    while i < a.len() {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

/// f64 norm avoids overflow / underflow for all finite nonzero f32 inputs.
pub fn normalize(row: &mut [f32]) -> Result<(), String> {
    if row.iter().any(|x| !x.is_finite()) {
        return Err("embeddings must contain only finite values".into());
    }
    let norm = row
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err("zero embeddings have undefined cosine similarity".into());
    }
    for x in row {
        *x = (*x as f64 / norm) as f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simd_matches_scalar_including_tails() {
        for n in 1..130 {
            let a: Vec<_> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
            let b: Vec<_> = (0..n).map(|i| (i as f32 * 0.71).cos()).collect();
            assert!((select_dot()(&a, &b) - scalar_dot(&a, &b)).abs() < 1e-4);
        }
    }
    #[test]
    fn normalization_extremes_and_invalid_values() {
        for x in [f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
            let mut v = vec![x; 3];
            normalize(&mut v).unwrap();
            assert!((scalar_dot(&v, &v) - 1.0).abs() < 1e-6);
        }
        for x in [0.0, f32::NAN, f32::INFINITY] {
            assert!(normalize(&mut [x]).is_err());
        }
    }
}
