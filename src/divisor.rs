use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_mulhi_epu16, _mm256_mullo_epi16, _mm256_set1_epi16,
    _mm256_srl_epi16, _mm256_sub_epi16, _mm_set1_epi64x,
};

struct Divisor {
    divisor: u16,
    multiplier: u16,
    inner_shift: u16,
    outer_shift: u16,
}

impl Divisor {
    fn new(divisor: u16) -> Self {
        debug_assert!(divisor != 0);

        let log2_divisor = (2 * (divisor as u32) - 1).log2() as u16;

        let multiplier = (((1u32 << 16) * ((1u32 << log2_divisor) - (divisor as u32)))
            / (divisor as u32)
            + 1) as u16;

        let inner_shift = log2_divisor.min(1);

        let outer_shift = if log2_divisor == 0 {
            0
        } else {
            log2_divisor - 1
        };

        Self {
            divisor,
            multiplier,
            inner_shift,
            outer_shift,
        }
    }

    fn divide(&self, numerator: u16) -> u16 {
        let t = numerator.widening_mul(self.multiplier).1;
        (t + ((numerator - t) >> self.inner_shift)) >> self.outer_shift
    }

    fn modulo(&self, numerator: u16) -> u16 {
        numerator - self.divide(numerator) * self.divisor
    }

    #[cfg(target_feature = "avx2")]
    unsafe fn divide_m256i(&self, numerators: __m256i) -> __m256i {
        let t = _mm256_mulhi_epu16(numerators, _mm256_set1_epi16(self.multiplier as i16));

        let inner = _mm256_srl_epi16(
            _mm256_sub_epi16(numerators, t),
            _mm_set1_epi64x(self.inner_shift as i64),
        );

        _mm256_srl_epi16(
            _mm256_add_epi16(t, inner),
            _mm_set1_epi64x(self.outer_shift as i64),
        )
    }

    #[cfg(target_feature = "avx2")]
    unsafe fn modulo_m256i(&self, numerators: __m256i) -> __m256i {
        let quotients = self.divide_m256i(numerators);
        _mm256_sub_epi16(
            numerators,
            _mm256_mullo_epi16(quotients, _mm256_set1_epi16(self.divisor as i16)),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::divisor::Divisor;
    use core::arch::x86_64::_mm256_loadu_si256;
    use core::iter;
    use core::mem::transmute;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_divide_modulo() {
        let cases = (1..5000)
            .chain((5000..u16::MAX).step_by(17))
            .chain((0..16).map(|i| 1 << i))
            .chain(iter::once(u16::MAX))
            .collect::<Vec<u16>>();

        for &denominator in &cases {
            let divisor = Divisor::new(denominator);

            for &numerator in iter::once(&0u16).chain(cases.iter()) {
                assert_eq!(divisor.divide(numerator), numerator / denominator);
                assert_eq!(divisor.modulo(numerator), numerator % denominator);
            }
        }
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn test_divide_modulo_m256() {
        let cases = (0..5000)
            .chain((5000..u16::MAX).step_by(17))
            .chain((0..16).map(|i| 1 << i))
            .chain(iter::once(u16::MAX))
            .collect::<Vec<u16>>();

        let mut rng = thread_rng();

        for &denominator in cases.iter().skip(1) {
            let divisor = Divisor::new(denominator);

            for _ in 0..100 {
                let numerators_vec = (0..16)
                    .map(|_| rng.gen_range(0..(cases.len() as u16)))
                    .collect::<Vec<_>>();

                let quotients_vec = numerators_vec
                    .iter()
                    .map(|numerator| numerator / denominator)
                    .collect::<Vec<_>>();

                let residues_vec = numerators_vec
                    .iter()
                    .map(|numerator| numerator % denominator)
                    .collect::<Vec<_>>();

                let numerators = unsafe { _mm256_loadu_si256(transmute(numerators_vec.as_ptr())) };
                let quotients: [u16; 16] = unsafe { transmute(divisor.divide_m256i(numerators)) };
                let residues: [u16; 16] = unsafe { transmute(divisor.modulo_m256i(numerators)) };
                assert_eq!(&quotients, quotients_vec.as_slice());
                assert_eq!(&residues, residues_vec.as_slice());
            }
        }
    }
}
