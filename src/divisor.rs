#[cfg(target_feature = "avx2")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_mullo_epi32,
    _mm256_set1_epi32, _mm256_sll_epi16, _mm256_srai_epi16, _mm256_srl_epi16, _mm256_srli_epi32,
    _mm256_sub_epi16, _mm256_sub_epi32, _mm_set1_epi64x,
};

pub struct Divisor32x16 {
    divisor: u16,
    l: u16,
    m_prime: u16,
    d_norm: u16,
}

impl Divisor32x16 {
    pub const fn new(divisor: u16) -> Self {
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

    pub const fn divisor(&self) -> u16 {
        self.divisor
    }

    pub fn modulo(&self, numerator: u32) -> u16 {
        const fn high(n: u32) -> u16 {
            (n >> 16) as u16
        }

        const fn low(n: u32) -> u16 {
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
            let q_1x =
                (self.m_prime as u32) * (n_2.wrapping_sub(minus_n_1) as u32) + (n_adj as u32);
            n_2 + high(q_1x)
        };

        let dr = (numerator
            .wrapping_sub((q_1 as u32) * (self.divisor as u32))
            .wrapping_sub(self.divisor as u32)) as i32;

        debug_assert!(-(self.divisor as i32) <= dr && dr < (self.divisor as i32));

        (dr as u16).wrapping_add(self.divisor & high(dr as u32))
    }

    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    pub unsafe fn modulo_m256i(&self, numerators: __m256i) -> __m256i {
        unsafe fn high(n: __m256i) -> __m256i {
            _mm256_srli_epi32(n, 16)
        }

        unsafe fn low(n: __m256i) -> __m256i {
            _mm256_and_si256(n, _mm256_set1_epi32((1 << 16) - 1))
        }

        let n_2 = {
            let n_2x = _mm256_sll_epi16(high(numerators), _mm_set1_epi64x((16 - self.l) as i64));
            let n_2y = _mm256_srl_epi16(low(numerators), _mm_set1_epi64x(self.l as i64));
            _mm256_add_epi16(n_2x, n_2y)
        };

        let n_10 = _mm256_sll_epi16(low(numerators), _mm_set1_epi64x((16 - self.l) as i64));

        let minus_n_1 = _mm256_srai_epi16(n_10, 15);

        let n_adj = {
            let n_adj_x = _mm256_and_si256(minus_n_1, _mm256_set1_epi32(self.d_norm as i32));
            _mm256_add_epi16(n_10, n_adj_x)
        };

        let q_1 = {
            let q_1x = _mm256_add_epi32(
                _mm256_mullo_epi32(
                    _mm256_set1_epi32(self.m_prime as i32),
                    _mm256_sub_epi16(n_2, minus_n_1),
                ),
                n_adj,
            );
            _mm256_add_epi16(n_2, high(q_1x))
        };

        let divisor = _mm256_set1_epi32(self.divisor as i32);

        let dr = _mm256_sub_epi32(
            _mm256_sub_epi32(numerators, _mm256_mullo_epi32(q_1, divisor)),
            divisor,
        );

        _mm256_add_epi16(low(dr), _mm256_and_si256(divisor, high(dr)))
    }
}

#[cfg(test)]
mod tests {
    use crate::divisor::Divisor32x16;
    use core::arch::x86_64::_mm256_loadu_si256;
    use core::iter;
    use rand::{thread_rng, Rng};

    #[cfg(target_feature = "avx2")]
    use core::mem::transmute;

    #[test]
    fn test_modulo() {
        let mut rng = thread_rng();

        let denominators = (1..1000)
            .chain((1000..u16::MAX).step_by(10))
            .chain(iter::once(u16::MAX))
            .chain((0..16).map(|i| 1 << i));

        for denominator in denominators {
            let divisor = Divisor32x16::new(denominator);

            for _ in 0..1000 {
                let numerator = rng.gen_range(0..(denominator as u32) * (denominator as u32));
                let expected_residue = (numerator % (denominator as u32)) as u16;
                let residue = divisor.modulo(numerator);
                assert_eq!(residue, expected_residue);
            }
        }
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn test_modulo_m256() {
        let mut rng = thread_rng();

        let denominators = (1..1000)
            .chain((1000..u16::MAX).step_by(100))
            .chain(iter::once(u16::MAX))
            .chain((0..16).map(|i| 1 << i));

        for denominator in denominators {
            let divisor = Divisor32x16::new(denominator);

            for _ in 0..250 {
                let numerators = (0..8)
                    .map(|_| rng.gen_range(0..(denominator as u32) * (denominator as u32)))
                    .collect::<Vec<_>>();

                let expected_residues = numerators
                    .iter()
                    .map(|numerator| numerator % (denominator as u32))
                    .collect::<Vec<_>>();

                let numerators = unsafe { _mm256_loadu_si256(transmute(numerators.as_ptr())) };
                let residues: [u32; 8] = unsafe { transmute(divisor.modulo_m256i(numerators)) };
                assert_eq!(&residues, expected_residues.as_slice());
            }
        }
    }
}
