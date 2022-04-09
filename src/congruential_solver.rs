use core::arch::x86_64::{
    _mm256_cmpeq_epi32, _mm256_cvtepi32_ps, _mm256_load_si256, _mm256_movemask_ps,
    _mm256_setzero_si256,
};
use core::mem::transmute;
use std::alloc::{alloc_zeroed, dealloc, Layout};

struct CongruentialSolver {
    rows: usize,
    columns: usize,
    modulus: u32,
    coefficients: *mut u32,
}

enum SolveResult {
    Determinate { residues: Vec<u32> },
    Incompatible { coefficients: Vec<u32> },
    Compatible,
}

impl CongruentialSolver {
    pub fn new(rows: usize, columns: usize, modulus: u32) -> CongruentialSolver {
        let size = rows * pad_columns(columns);
        let coefficients = {
            let size = rows * pad_columns(columns);
            unsafe {
                let layout = Layout::from_size_align_unchecked(4 * size, 32);
                alloc_zeroed(layout) as *mut u32
            }
        };

        CongruentialSolver {
            rows,
            columns,
            modulus,
            coefficients,
        }
    }

    pub fn solve(self) -> SolveResult {
        for i in 0..self.rows {}

        unimplemented!();
    }

    unsafe fn row(&self, row: usize) -> *const u32 {
        self.coefficients.add(row * pad_columns(self.columns))
    }

    unsafe fn non_zero_coefficient(&self, row: usize, from: usize) -> Option<(usize, u32)> {
        non_zero_element(self.row(row), from, self.columns)
    }
}

impl Drop for CongruentialSolver {
    fn drop(&mut self) {
        let size = self.rows * pad_columns(self.columns);

        unsafe {
            let layout = Layout::from_size_align_unchecked(4 * size, 32);
            dealloc(self.coefficients as *mut u8, layout);
        }
    }
}

fn pad_columns(columns: usize) -> usize {
    #[cfg(target_feature = "avx2")]
    {
        pad8(columns)
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        columns
    }
}

fn pad8(value: usize) -> usize {
    value + ((8 - (value % 8)) % 8)
}

// Given relatively prime p and q, compute the modular inverse of q modulo p using the Extended Euclidean Algorithm
fn modular_inverse(p: i64, q: i64) -> i64 {
    let p0 = p;
    let q0 = q;

    let mut p = p;
    let mut q = q;

    let mut s0 = 1;
    let mut t0 = 0;

    let mut s1 = 0;
    let mut t1 = 1;

    while q != 0 {
        let u = p / q;
        let r = p % q;

        p = q;
        q = r;

        let s2 = s0 - u * s1;
        s0 = s1;
        s1 = s2;

        let t2 = t0 - u * t1;
        t0 = t1;
        t1 = t2;
    }

    debug_assert!(p == 1 && q == 0);
    debug_assert!(p0 * s0 + q0 * t0 == 1);

    (t0 + p0) % p0
}

unsafe fn non_zero_element(data: *const u32, from: usize, len: usize) -> Option<(usize, u32)> {
    debug_assert!(data.align_offset(32) == 0);

    #[cfg(target_feature = "avx2")]
    {
        for i in from..pad8(from) {
            let value = *data.add(i);
            if value != 0 {
                return Some((i, value));
            }
        }

        for i in (pad8(from)..len).step_by(8) {
            let values = _mm256_load_si256(transmute(data.add(i)));
            let zeros = _mm256_cmpeq_epi32(values, _mm256_setzero_si256());
            let non_zero_mask = _mm256_movemask_ps(_mm256_cvtepi32_ps(zeros)) ^ 0b11111111;

            if non_zero_mask != 0 {
                let j = non_zero_mask.trailing_zeros() as usize;
                let coefficient = *data.add(i + j);
                return Some((i + j, coefficient));
            }
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        for i in from..len {
            let value = *data.add(i);
            if value != 0 {
                return Some((i, value));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use crate::congruential_solver::{modular_inverse, non_zero_element, pad8};
    use core::slice;
    use std::alloc::{alloc_zeroed, dealloc, Layout};

    #[test]
    fn test_non_zero_element() {
        let layout = unsafe { Layout::from_size_align_unchecked(4 * 10000, 32) };

        let data: &mut [u32] =
            unsafe { slice::from_raw_parts_mut(alloc_zeroed(layout) as *mut u32, 10000) };

        for len in [1, 2, 3, 5, 7, 10, 13, 25, 100, 1000, 10000] {
            assert_eq!(unsafe { non_zero_element(data.as_ptr(), 0, len) }, None);

            for i in 0..len {
                if i > 0 {
                    data[i - 1] = 0;
                }
                data[i] = (i + 1) as u32;

                assert_eq!(
                    unsafe { non_zero_element(data.as_ptr(), 0, len) },
                    Some((i, (i + 1) as u32))
                );

                assert_eq!(
                    unsafe { non_zero_element(data.as_ptr(), i, len) },
                    Some((i, (i + 1) as u32))
                );

                assert_eq!(unsafe { non_zero_element(data.as_ptr(), i + 1, len) }, None);
                assert_eq!(unsafe { non_zero_element(data.as_ptr(), len, len) }, None);
            }

            data[len - 1] = 0
        }

        unsafe {
            dealloc(data.as_mut_ptr() as *mut u8, layout);
        }
    }

    #[test]
    fn test_pad8() {
        for columns in 0..100000 {
            let padded = pad8(columns);
            assert!(padded >= columns && padded - columns < 8 && padded % 8 == 0);
        }
    }

    #[test]
    fn test_modular_inverse() {
        for &p in &[2, 3, 5, 7, 11, 13, 17, 19, 9001] {
            for i in 1..p {
                let inverse = modular_inverse(p, i);
                assert_eq!((i * inverse) % p, 1);
            }
        }
    }
}
