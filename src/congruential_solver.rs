use crate::divisor::Divisor;

use std::alloc::{alloc_zeroed, dealloc, Layout};

struct CongruentialSolver {
    rows: usize,
    columns: usize,
    divisor: Divisor,
    coefficients: *mut u16,
}

enum SolveResult {
    Determinate { residues: Vec<u16> },
    Incompatible { coefficients: Vec<u16> },
    Compatible,
}

impl CongruentialSolver {
    pub fn new(rows: usize, columns: usize, modulus: u16) -> CongruentialSolver {
        let coefficients = unsafe { alloc_zeroed(Self::layout(rows, columns)) as *mut u16 };

        let divisor = Divisor::new(modulus);

        CongruentialSolver {
            rows,
            columns,
            divisor,
            coefficients,
        }
    }

    pub(crate) fn from_vec(coefficients: Vec<Vec<u16>>, modulus: u16) -> Self {
        assert!(!coefficients.is_empty());
        assert!(!coefficients[0].is_empty());

        let rows = coefficients.len();
        let columns = coefficients[0].len() - 1;

        for row in coefficients.iter().skip(1) {
            assert!(row.len() == columns + 1);
        }

        let mut solver = Self::new(rows, columns, modulus);

        for (i, row) in coefficients.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                solver.set(i, j, value);
            }
        }

        solver
    }

    pub fn get(&self, i: usize, j: usize) -> u16 {
        assert!(i < self.rows);
        assert!(j <= self.columns);
        unsafe { *self.row(i).add(j) }
    }

    pub fn set(&mut self, i: usize, j: usize, value: u16) {
        assert!(i < self.rows);
        assert!(j <= self.columns);
        unsafe {
            *self.row_mut(i).add(j) = value;
        }
    }

    pub fn solve(self) -> SolveResult {
        if self.rows < self.columns {
            return SolveResult::Compatible;
        }

        for j in 0..self.columns {
            let (_i, leading_coefficient) = match unsafe { self.find_non_zero_coefficient(j) } {
                Some(pair) => pair,
                None => return SolveResult::Compatible,
            };

            let _inverse =
                modular_inverse(leading_coefficient as i32, self.divisor.divisor() as i32);

            // Swap i, j and normalize
            // Eliminate
        }

        // Check extra rows
        // Back-substitute

        unimplemented!();
    }

    fn width(columns: usize) -> usize {
        #[cfg(target_feature = "avx2")]
        {
            pad8(columns + 1)
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            columns + 1
        }
    }

    fn layout(rows: usize, columns: usize) -> Layout {
        let size = rows * Self::width(columns);
        unsafe { Layout::from_size_align_unchecked(2 * size, 32) }
    }

    unsafe fn row(&self, i: usize) -> *const u16 {
        debug_assert!(i < self.rows);
        self.coefficients.add(i * Self::width(self.columns))
    }

    unsafe fn row_mut(&mut self, i: usize) -> *mut u16 {
        debug_assert!(i < self.rows);
        self.coefficients.add(i * Self::width(self.columns))
    }

    unsafe fn find_non_zero_coefficient(&self, column: usize) -> Option<(usize, u16)> {
        debug_assert!(column < self.columns);
        debug_assert!(self.columns <= self.rows);

        for i in column..self.rows {
            let value = *self.row(i).add(column);
            if value != 0 {
                return Some((i, value));
            }
        }

        None
    }
}

impl Drop for CongruentialSolver {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.coefficients as *mut u8,
                Self::layout(self.rows, self.columns),
            );
        }
    }
}

fn pad8(value: usize) -> usize {
    value + ((8 - (value % 8)) % 8)
}

// Given relatively prime p and q, compute the modular inverse of q modulo p using the Extended Euclidean Algorithm
fn modular_inverse(p: i32, q: i32) -> i32 {
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
    debug_assert!(-p0 < t0 && t0 < p0);

    if t0 < 0 {
        t0 + p0
    } else {
        t0
    }
}

unsafe fn modular_multiply_and_swap(
    destination: *mut u16,
    source: *mut u16,
    _from: usize,
    _len: usize,
    _multiplier: u16,
    _divisor: Divisor,
) {
    #[cfg(target_feature = "avx2")]
    {
        debug_assert!(destination.align_offset(32) == 0);
        debug_assert!(source.align_offset(32) == 0);
        unimplemented!();
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        for i in from..len {
            let t = *destination.add(i);
            let x = *source.add(i);
            let y = x.widening_mul(multiplier);
            *destination.add(i) = (source as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::congruential_solver::{modular_inverse, pad8};

    #[test]
    fn test_pad8() {
        for columns in 0..1000000 {
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
