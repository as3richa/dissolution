#[cfg(target_feature = "avx2")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_mulhi_epu16, _mm256_mullo_epi16, _mm256_set1_epi16,
    _mm256_srl_epi16, _mm256_sub_epi16, _mm_set1_epi64x,
};

pub struct Divisor {
    divisor: u16,
    multiplier: u16,
    inner_shift: u16,
    outer_shift: u16,
}

impl Divisor {
    pub fn new(divisor: u16) -> Self {
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

    pub fn divisor(&self) -> u16 {
        self.divisor
    }

    pub fn divide(&self, numerator: u16) -> u16 {
        let t = numerator.widening_mul(self.multiplier).1;
        (t + ((numerator - t) >> self.inner_shift)) >> self.outer_shift
    }

    pub fn modulo(&self, numerator: u16) -> u16 {
        numerator - self.divide(numerator) * self.divisor
    }

    #[cfg(target_feature = "avx2")]
    pub unsafe fn divide_m256i(&self, numerators: __m256i) -> __m256i {
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
    pub unsafe fn modulo_m256i(&self, numerators: __m256i) -> __m256i {
        let quotients = self.divide_m256i(numerators);
        _mm256_sub_epi16(
            numerators,
            _mm256_mullo_epi16(quotients, _mm256_set1_epi16(self.divisor as i16)),
        )
    }
}

pub struct Divisor32x16 {
    divisor: u16,
    l: u16,
    m_prime: u16,
    d_norm: u16,
}

impl Divisor32x16 {
    pub fn new(divisor: u16) -> Self {
        let l = (1 + divisor.log2()) as u16;

        let m_prime =
            (((1u32 << 16) * ((1u32 << l) - (divisor as u32)) - 1) / (divisor as u32)) as u16;

        let d_norm = divisor << (16 - l);

        Self {
            divisor,
            l,
            m_prime,
            d_norm,
        }
    }

    pub fn divisor(&self) -> u16 {
        self.divisor
    }

    pub fn divide(&self, numerator: u32) -> (u16, u16) {
        fn high(n: u32) -> u16 {
            (n >> 16) as u16
        }

        fn low(n: u32) -> u16 {
            n as u16
        }

        let n_2 = {
            let n_2x = high(numerator) << (16 - self.l);
            if self.l == 16 {
                n_2x
            } else {
                n_2x + (low(numerator) >> self.l)
            }
        };

        let n_10 = low(numerator) << (16 - self.l);

        let minus_n_1 = ((n_10 as i16) >> 15) as u16;

        let n_adj = n_10.wrapping_add(minus_n_1 & self.d_norm);

        let q_1 = {
            let q_1x = (self.m_prime as u32) * ((n_2.wrapping_sub(minus_n_1 as u16)) as u32)
                + (n_adj as u32);
            n_2 + high(q_1x)
        };

        let dr = (numerator
            .wrapping_sub((q_1 as u32) * (self.divisor as u32))
            .wrapping_sub(self.divisor as u32)) as i32;

        debug_assert!(-(self.divisor as i32) <= dr && dr < (self.divisor as i32));

        (
            q_1.wrapping_add(1).wrapping_sub(1 & high(dr as u32)),
            (dr as u16).wrapping_add(self.divisor & high(dr as u32)),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::divisor::{Divisor, Divisor32x16};
    use core::arch::x86_64::_mm256_loadu_si256;
    use core::iter;
    use rand::{thread_rng, Rng};

    #[cfg(target_feature = "avx2")]
    use core::mem::transmute;

    fn cases() -> Vec<u16> {
        let mut rng = thread_rng();
        let mut cases = (1..1000)
            .chain((1000..u16::MAX).step_by(100))
            .chain((0..16).map(|i| 1 << i))
            .chain(iter::once(u16::MAX))
            .chain((0..1000).map(|_| rng.gen::<u16>()))
            .collect::<Vec<_>>();
        cases.sort_unstable();
        cases
    }

    #[test]
    fn test_divide_modulo() {
        let cases = cases();

        for &denominator in &cases {
            if denominator == 0 {
                continue;
            }

            let divisor = Divisor::new(denominator);

            for &numerator in &cases {
                assert_eq!(divisor.divide(numerator), numerator / denominator);
                assert_eq!(divisor.modulo(numerator), numerator % denominator);
            }
        }
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn test_divide_modulo_m256() {
        let mut rng = thread_rng();
        let cases = cases();

        for &denominator in &cases {
            if denominator == 0 {
                continue;
            }

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

    #[test]
    fn test_divide_modulo2() {
        let mut rng = thread_rng();

        let denominators = cases();

        let numerators = {
            let mut numerators = denominators
                .iter()
                .map(|&i| i as u32)
                .chain((16..32).map(|i| 1u32 << i))
                .chain(iter::once(u32::MAX))
                .chain((0..1000).map(|_| rng.gen::<u32>()))
                .collect::<Vec<_>>();
            numerators.sort_unstable();
            numerators
        };

        for &denominator in &denominators {
            if denominator == 0 {
                continue;
            }

            let divisor = Divisor32x16::new(denominator);

            for &numerator in &numerators {
                let expected_quotient = numerator / (denominator as u32);

                if expected_quotient > (u16::MAX as u32) {
                    break;
                }

                let expected_quotient = expected_quotient as u16;
                let expected_residue = (numerator % (denominator as u32)) as u16;

                let (quotient, residue) = divisor.divide(numerator);
                assert_eq!(quotient, expected_quotient);
                assert_eq!(residue, expected_residue);
            }
        }
    }
}
