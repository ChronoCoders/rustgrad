mod layout;
mod storage;
// The inner module shares the name of the parent — intentional: the public
// type `Tensor` is the defining export of the `tensor` module.
#[allow(clippy::module_inception)]
mod tensor;

pub use layout::Layout;
pub use storage::{DType, Device, Storage};
pub use tensor::Tensor;
