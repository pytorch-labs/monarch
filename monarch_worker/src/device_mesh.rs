#![allow(dead_code)] // Temporary, until code is exercised by the worker.

use std::collections::HashMap;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use itertools::Itertools;
use itertools::izip;
use ndslice::Slice;
use pyo3::pyclass;
use serde::Deserialize;
use serde::Serialize;

/// A single dimension in a [`DeviceMesh`], relative to a specific rank.
/// Each dimension is the set of ranks that share all _other_ mesh coordinates
/// with the owning rank while ranging over the dimension's coordinates.
///
/// For example, if a mesh has 3 dimensions, and the owning rank's coordinates
/// are `(1, 2, 3)`, then:
/// * [`Dim`] 0 will be `(*, 2, 3)`
/// * [`Dim`] 1 will be `(1, *, 3)`
/// * [`Dim`] 2 will be `(1, 2, *)`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(frozen, module = "monarch_worker._internal")]
#[pyo3(get_all)]
pub struct Dim {
    /// The name of the dimension.
    name: String,
    /// The rank of this worker within the dimension's process group.
    rank: usize,
    /// The size of the dimension.
    size: usize,
    /// The ordered set of ranks within the dimension. `members[rank]` is always
    /// equal to the owning rank.
    members: Vec<usize>,
}

impl Dim {
    pub fn members(&self) -> &[usize] {
        &self.members
    }
    pub fn size(&self) -> usize {
        self.size
    }
}

/// A device mesh represents each (named) dimension ([`Dim`]) of a
/// multi-dimensional mesh, relative to a specific rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(frozen, module = "monarch_worker._internal")]
#[pyo3(get_all)]
pub struct DeviceMesh {
    /// Each dim in the mesh.
    dims: HashMap<String, Dim>,
    /// All ranks (i.e., the full device mesh).
    all_ranks: Vec<usize>,
}

impl DeviceMesh {
    /// Create a new [`DeviceMesh`] with the provided dimension names and
    /// multi-dimensional slice. `rank` is the owning (self) rank.
    pub fn new(names: Vec<String>, ranks: Slice, rank: usize) -> Result<Self> {
        let mut dims = HashMap::new();
        let coordinates = ranks.coordinates(rank)?;

        // Check that all vecs are the same length.
        // coordinates == sizes == strides is enforced by Slice.
        ensure!(
            names.len() == coordinates.len(),
            "names and coordinates mismatch in length: names={names:#?}, coordinates={coordinates:#?}",
        );
        for (coordinate, name, size, stride) in
            izip!(coordinates, names, ranks.sizes(), ranks.strides())
        {
            let start = rank - stride * coordinate;
            let members: Vec<usize> = (start..start + stride * size).step_by(*stride).collect();
            assert_eq!(members[coordinate], rank);
            dims.insert(
                name.clone(),
                Dim {
                    name,
                    rank: coordinate,
                    size: *size,
                    members,
                },
            );
        }

        Ok(Self {
            dims,
            all_ranks: ranks.iter().collect(),
        })
    }

    /// Return a dict of dimension names to their corresponding ranks.
    pub fn ranks(&self) -> HashMap<String, usize> {
        self.dims.iter().map(|(n, d)| (n.clone(), d.rank)).collect()
    }

    /// Return a dict of dimension names to their corresponding size.
    pub fn sizes(&self) -> HashMap<String, usize> {
        self.dims.iter().map(|(n, d)| (n.clone(), d.size)).collect()
    }

    pub fn dim(&self, name: &str) -> Option<&Dim> {
        self.dims.get(name)
    }

    /// Return all the ranks that participate in collectives across the given
    /// dim names.
    pub fn get_ranks_for_dim_slice(&self, names: &[String]) -> Result<Vec<usize>> {
        // Early returns for empty case.
        if names.is_empty() {
            return Ok(vec![]);
        }

        // Early return for single dimension cases
        if let [name] = names {
            return Ok(self
                .dims
                .get(name)
                .with_context(|| format!("no dim with name {}", name))?
                .members
                .clone());
        }

        // Get all the dimensions
        let dims: Vec<&Dim> = names
            .iter()
            .map(|n| {
                self.dims
                    .get(n)
                    .with_context(|| format!("no dim with name {}", n))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Calculate strides
        let strides: Vec<usize> = dims
            .iter()
            .map(|d| match d.members.as_slice() {
                [d0, d1, ..] => d1 - d0,
                _ => 0,
            })
            .collect();

        // Calculate start value
        let start = dims[0].members[dims[0].rank]
            - dims
                .iter()
                .zip(&strides)
                .map(|(d, &s)| s * d.rank)
                .sum::<usize>();

        // Generate all combinations of indices and calculate ranks
        Ok(dims
            .iter()
            .map(|d| 0..d.size)
            .multi_cartesian_product()
            .map(|idxs| {
                start
                    + idxs
                        .into_iter()
                        .zip(&strides)
                        .map(|(i, &s)| i * s)
                        .sum::<usize>()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let names = vec!["x".to_string(), "y".to_string()];
        let ranks = Slice::new(0, vec![2, 3], vec![3, 1]).unwrap();
        let mesh = DeviceMesh::new(names, ranks, 1).unwrap();
        assert_eq!(mesh.dims.len(), 2);
        assert_eq!(mesh.all_ranks.len(), 6);
        assert_eq!(mesh.dims["x"].rank, 0);
        assert_eq!(mesh.dims["x"].members, vec![1, 4]);
        assert_eq!(mesh.dims["y"].rank, 1);
        assert_eq!(mesh.dims["y"].members, vec![0, 1, 2]);
    }

    #[test]
    fn get_ranks_for_dim_slice() -> Result<()> {
        // 2D mesh test (2x2)
        let names = vec!["x".to_string(), "y".to_string()];
        let ranks = Slice::new(0, vec![2, 3], vec![3, 1])?;
        let mesh = DeviceMesh::new(names, ranks, 1)?;
        assert!(mesh.get_ranks_for_dim_slice(&[])?.is_empty());
        assert_eq!(
            mesh.get_ranks_for_dim_slice(&["y".to_string()])?,
            mesh.dims["y"].members,
        );
        assert_eq!(
            mesh.get_ranks_for_dim_slice(&["x".to_string(), "y".to_string()])?,
            mesh.all_ranks,
        );

        // 3D mesh test (2x2x2)
        let names = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let ranks = Slice::new(0, vec![2, 2, 2], vec![4, 2, 1])?;
        let mesh = DeviceMesh::new(names, ranks, 1)?;
        assert!(mesh.get_ranks_for_dim_slice(&[])?.is_empty());
        assert_eq!(
            mesh.get_ranks_for_dim_slice(&["x".to_string()])?,
            vec![1, 5],
        );
        assert_eq!(
            mesh.get_ranks_for_dim_slice(&["y".to_string()])?,
            vec![1, 3],
        );
        assert_eq!(
            mesh.get_ranks_for_dim_slice(&["z".to_string()])?,
            vec![0, 1],
        );
        assert_eq!(
            mesh.get_ranks_for_dim_slice(&["x".to_string(), "y".to_string()])?,
            vec![1, 3, 5, 7],
        );
        assert_eq!(
            mesh.get_ranks_for_dim_slice(&["y".to_string(), "z".to_string()])?,
            vec![0, 1, 2, 3],
        );
        assert_eq!(
            mesh.get_ranks_for_dim_slice(&["x".to_string(), "y".to_string(), "z".to_string()])?,
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );

        Ok(())
    }
}
