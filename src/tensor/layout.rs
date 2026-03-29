/// Describes how a logical tensor maps onto a flat storage buffer.
///
/// All zero-copy views — transpose, permute, slice, broadcast — are
/// represented by producing a new `Layout` over the same `Arc<Storage>`.
/// No data movement ever occurs for shape operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Logical shape of the tensor (number of elements per dimension).
    pub shape: Vec<usize>,
    /// Stride per dimension in elements. A stride of 0 indicates a
    /// broadcast dimension — the same element is reused across that axis.
    pub strides: Vec<usize>,
    /// Offset into the flat buffer where this view begins.
    pub offset: usize,
}

impl Layout {
    /// Build a contiguous (row-major / C-order) layout for `shape`.
    ///
    /// Strides are computed as: `strides[i] = shape[i+1] * … * shape[n-1]`.
    pub fn contiguous(shape: Vec<usize>) -> Self {
        let strides = Self::row_major_strides(&shape);
        Self { shape, strides, offset: 0 }
    }

    /// Compute row-major strides for the given shape.
    pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
        let n = shape.len();
        let mut strides = vec![1usize; n];
        for i in (0..n.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Total number of logical elements (product of all dimensions).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns `true` if the layout is contiguous with no offset.
    ///
    /// A contiguous layout has row-major strides and `offset == 0`.
    /// Broadcast layouts (stride 0) are never contiguous.
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        // Any 0-stride means broadcast — not contiguous.
        if self.strides.contains(&0) {
            return false;
        }
        let expected = Self::row_major_strides(&self.shape);
        self.strides == expected
    }

    /// Compute the flat buffer index for a multi-dimensional index.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != self.ndim()` or any index is out of bounds.
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "index rank {} does not match tensor rank {}",
            indices.len(),
            self.shape.len()
        );
        for (dim, (&idx, &dim_size)) in indices.iter().zip(self.shape.iter()).enumerate() {
            assert!(
                idx < dim_size,
                "index {} out of bounds for dimension {} with size {}",
                idx,
                dim,
                dim_size
            );
        }
        self.offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>()
    }

    /// Return a new layout with dimensions permuted according to `dims`.
    ///
    /// `dims` must be a permutation of `0..ndim`.
    ///
    /// # Panics
    ///
    /// Panics if `dims` is not a valid permutation of the axis indices.
    pub fn permute(&self, dims: &[usize]) -> Self {
        assert_eq!(
            dims.len(),
            self.ndim(),
            "permute: dims length {} does not match tensor rank {}",
            dims.len(),
            self.ndim()
        );
        let mut seen = vec![false; self.ndim()];
        for &d in dims {
            assert!(
                d < self.ndim(),
                "permute: axis {} out of range for rank {}",
                d,
                self.ndim()
            );
            assert!(!seen[d], "permute: axis {} appears more than once", d);
            seen[d] = true;
        }
        let shape = dims.iter().map(|&d| self.shape[d]).collect();
        let strides = dims.iter().map(|&d| self.strides[d]).collect();
        Self { shape, strides, offset: self.offset }
    }

    /// Swap two axes. Equivalent to `permute` with two axes swapped.
    ///
    /// # Panics
    ///
    /// Panics if `dim0` or `dim1` are out of range.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        assert!(
            dim0 < self.ndim(),
            "transpose: dim0 {} out of range for rank {}",
            dim0,
            self.ndim()
        );
        assert!(
            dim1 < self.ndim(),
            "transpose: dim1 {} out of range for rank {}",
            dim1,
            self.ndim()
        );
        let mut dims: Vec<usize> = (0..self.ndim()).collect();
        dims.swap(dim0, dim1);
        self.permute(&dims)
    }

    /// Return a new layout reshaped to `new_shape`.
    ///
    /// Only valid if the layout is contiguous. Non-contiguous tensors must
    /// be materialized before reshaping.
    ///
    /// # Panics
    ///
    /// Panics if the layout is not contiguous, or if `new_shape` has a
    /// different number of elements than the current shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        assert!(
            self.is_contiguous(),
            "reshape: tensor must be contiguous (call .contiguous() first)"
        );
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            new_numel,
            self.numel(),
            "reshape: new shape {:?} has {} elements but tensor has {}",
            new_shape,
            new_numel,
            self.numel()
        );
        Self::contiguous(new_shape)
    }

    /// Insert a size-1 dimension at position `axis`.
    ///
    /// # Panics
    ///
    /// Panics if `axis > ndim`.
    pub fn unsqueeze(&self, axis: usize) -> Self {
        assert!(
            axis <= self.ndim(),
            "unsqueeze: axis {} out of range for rank {}",
            axis,
            self.ndim()
        );
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        // The stride for the new size-1 dim does not affect addressing;
        // use the stride of the next dim, or 1 if inserting at the end.
        let new_stride = if axis < self.ndim() { self.strides[axis] } else { 1 };
        shape.insert(axis, 1);
        strides.insert(axis, new_stride);
        Self { shape, strides, offset: self.offset }
    }

    /// Remove all size-1 dimensions, or a specific one if `axis` is given.
    ///
    /// # Panics
    ///
    /// Panics if `axis` is `Some(a)` and `a` is out of range or the
    /// dimension at `a` is not size 1.
    pub fn squeeze(&self, axis: Option<usize>) -> Self {
        match axis {
            None => {
                let (shape, strides): (Vec<_>, Vec<_>) = self
                    .shape
                    .iter()
                    .zip(self.strides.iter())
                    .filter(|(&s, _)| s != 1)
                    .unzip();
                Self { shape, strides, offset: self.offset }
            }
            Some(ax) => {
                assert!(
                    ax < self.ndim(),
                    "squeeze: axis {} out of range for rank {}",
                    ax,
                    self.ndim()
                );
                assert_eq!(
                    self.shape[ax], 1,
                    "squeeze: dimension {} has size {} (expected 1)",
                    ax, self.shape[ax]
                );
                let mut shape = self.shape.clone();
                let mut strides = self.strides.clone();
                shape.remove(ax);
                strides.remove(ax);
                Self { shape, strides, offset: self.offset }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_2d() {
        let layout = Layout::contiguous(vec![3, 4]);
        assert_eq!(layout.strides, vec![4, 1]);
        assert_eq!(layout.offset, 0);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn contiguous_strides_3d() {
        let layout = Layout::contiguous(vec![2, 3, 4]);
        assert_eq!(layout.strides, vec![12, 4, 1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn contiguous_strides_1d() {
        let layout = Layout::contiguous(vec![5]);
        assert_eq!(layout.strides, vec![1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn contiguous_scalar() {
        let layout = Layout::contiguous(vec![]);
        assert_eq!(layout.numel(), 1);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn flat_index_2d() {
        let layout = Layout::contiguous(vec![3, 4]);
        // row 1, col 2 → 1*4 + 2 = 6
        assert_eq!(layout.flat_index(&[1, 2]), 6);
    }

    #[test]
    fn flat_index_3d() {
        let layout = Layout::contiguous(vec![2, 3, 4]);
        // [1, 2, 3] → 1*12 + 2*4 + 3*1 = 23
        assert_eq!(layout.flat_index(&[1, 2, 3]), 23);
    }

    #[test]
    fn transpose_2d() {
        let layout = Layout::contiguous(vec![3, 4]);
        let t = layout.transpose(0, 1);
        assert_eq!(t.shape, vec![4, 3]);
        assert_eq!(t.strides, vec![1, 4]);
        assert!(!t.is_contiguous());
    }

    #[test]
    fn permute_3d() {
        let layout = Layout::contiguous(vec![2, 3, 4]);
        // [0,1,2] → strides [12,4,1]; permute [2,0,1]
        let p = layout.permute(&[2, 0, 1]);
        assert_eq!(p.shape, vec![4, 2, 3]);
        assert_eq!(p.strides, vec![1, 12, 4]);
    }

    #[test]
    fn reshape_contiguous() {
        let layout = Layout::contiguous(vec![2, 6]);
        let r = layout.reshape(vec![3, 4]);
        assert_eq!(r.shape, vec![3, 4]);
        assert_eq!(r.strides, vec![4, 1]);
        assert!(r.is_contiguous());
    }

    #[test]
    #[should_panic(expected = "reshape: tensor must be contiguous")]
    fn reshape_non_contiguous_panics() {
        let layout = Layout::contiguous(vec![3, 4]);
        let t = layout.transpose(0, 1);
        t.reshape(vec![12]);
    }

    #[test]
    fn unsqueeze_front() {
        let layout = Layout::contiguous(vec![3, 4]);
        let u = layout.unsqueeze(0);
        assert_eq!(u.shape, vec![1, 3, 4]);
    }

    #[test]
    fn unsqueeze_back() {
        let layout = Layout::contiguous(vec![3, 4]);
        let u = layout.unsqueeze(2);
        assert_eq!(u.shape, vec![3, 4, 1]);
    }

    #[test]
    fn squeeze_all() {
        let layout = Layout::contiguous(vec![1, 3, 1, 4]);
        let s = layout.squeeze(None);
        assert_eq!(s.shape, vec![3, 4]);
    }

    #[test]
    fn squeeze_axis() {
        let layout = Layout::contiguous(vec![1, 3, 4]);
        let s = layout.squeeze(Some(0));
        assert_eq!(s.shape, vec![3, 4]);
    }

    #[test]
    fn broadcast_layout_not_contiguous() {
        // A broadcast layout has stride 0 on the broadcast dimension.
        let layout = Layout {
            shape: vec![3, 4],
            strides: vec![0, 1],
            offset: 0,
        };
        assert!(!layout.is_contiguous());
    }

    #[test]
    fn numel() {
        let layout = Layout::contiguous(vec![2, 3, 4]);
        assert_eq!(layout.numel(), 24);
    }

    #[test]
    fn ndim() {
        let layout = Layout::contiguous(vec![2, 3, 4]);
        assert_eq!(layout.ndim(), 3);
    }
}
