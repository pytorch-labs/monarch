/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use async_trait::async_trait;
use ndslice::Range;
use ndslice::Shape;
use ndslice::ShapeError;
use ndslice::SliceIterator;

/// A mesh of nodes, organized into the topology described by its shape (see [`Shape`]).
#[async_trait]
pub trait Mesh {
    /// The type of the node contained in the mesh.
    type Node;

    /// The type of a slice of this mesh. Slices should not outlive their
    /// parent mesh.
    type Sliced<'a>: Mesh<Node = Self::Node> + 'a
    where
        Self: 'a;

    /// The shape of this mesh.
    fn shape(&self) -> &Shape;

    /// Sub-slice this mesh, specifying the included ranges for
    /// the dimension with the labeled name.
    fn select<R: Into<Range>>(&self, label: &str, range: R)
    -> Result<Self::Sliced<'_>, ShapeError>;

    /// Retrieve contained node at the provided index. The index is
    /// relative to the shape of the mesh.
    fn get(&self, index: usize) -> Option<Self::Node>;

    /// Iterate over all the nodes in this mesh.
    fn iter(&self) -> MeshIter<'_, Self> {
        MeshIter {
            mesh: self,
            slice_iter: self.shape().slice().iter(),
        }
    }
}

/// An iterator over the nodes of a mesh.
pub struct MeshIter<'a, M: Mesh + ?Sized> {
    mesh: &'a M,
    slice_iter: SliceIterator<'a>,
}

impl<M: Mesh> Iterator for MeshIter<'_, M> {
    type Item = M::Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.slice_iter
            .next()
            .map(|index| self.mesh.get(index).unwrap())
    }
}
