// until used publically
#![allow(dead_code)]

use hyperactor::actor::RemoteActor;
use ndslice::Selection;
use ndslice::selection::selection_from;
use ndslice::shape::Range;
use ndslice::shape::Shape;

use crate::reference::ActorMeshRef;

/// Specifies how to handle dimensions that exist in one mesh but not the other
#[derive(Clone, Copy)]
pub enum MappingMode {
    /// Broadcast to any dimensions that exist only in the target mesh
    /// Dimensions only in origin mesh are handled by rank iteration
    BroadcastMissing,
    /// Error if any dimensions exist in only one of the meshes
    ExactMatch,
}

/// Describes how two meshes should be aligned
#[derive(Clone, Copy)]
pub enum AlignPolicy {
    /// Creates a mapping from each origin rank to the entire target mesh
    Broadcast,
    /// Maps matching dimensions 1:1 and handles missing dimensions according to MappingMode
    Mapped {
        /// Specifies how to handle dimensions that exist in one mesh but not the other
        mode: MappingMode,
    },
}

/// Verifies that any matching dimensions between origin and target shapes have equal sizes
fn verify_dimension_sizes(origin: &Shape, target: &Shape) -> Result<(), anyhow::Error> {
    for label in origin.labels() {
        if let Some(target_pos) = target.labels().iter().position(|l| l == label) {
            let origin_pos = origin.labels().iter().position(|l| l == label).unwrap();
            let origin_size = origin.slice().sizes()[origin_pos];
            let target_size = target.slice().sizes()[target_pos];

            if origin_size != target_size {
                return Err(anyhow::Error::msg(format!(
                    "dimension {} has mismatched sizes: {} vs {}",
                    label, origin_size, target_size
                )));
            }
        }
    }
    Ok(())
}

/// Given a set of coordinates, create a set of single-value ranges for each dimension..
fn exact_mapping<'a>(
    target_selection: &Selection,
    coords: &'a [(String, usize)],
    origin_rank: &usize,
) -> Result<Vec<(&'a str, Range)>, anyhow::Error> {
    let coord_dim = coords.iter().map(|(_, d)| *d).collect::<Vec<_>>();
    if target_selection.contains(&coord_dim) {
        Ok(coords
            .iter()
            .map(|(label, index)| (label.as_str(), Range::from(*index)))
            .collect::<Vec<(&'a str, Range)>>())
    } else {
        Err(anyhow::Error::msg(format!(
            "origin rank {} is not selected in target",
            origin_rank
        )))
    }
}

/// Maps the target mesh to the origin mesh according to the alignment policy.
/// For Mapped policy:
/// - Matching dimensions must have equal sizes and are mapped 1:1
/// - Dimensions only in origin mesh are handled by rank iteration
/// - Dimensions only in target mesh are broadcast (full range)
///   Example:
///   origin: {replica=2, host=4}, target: {host=4, gpu=8}
/// - host maps 1:1 (sizes match)
/// - gpu is broadcast (target-only)
/// - replica is handled by rank iteration (origin-only)
pub fn map_meshes<A: RemoteActor, B: RemoteActor>(
    origin_mesh: &ActorMeshRef<A>,
    origin_rank: usize,
    target_mesh: &ActorMeshRef<B>,
    policy: AlignPolicy,
) -> Result<ActorMeshRef<B>, anyhow::Error> {
    let origin_shape = &origin_mesh.clone().shape();
    let target_shape = &target_mesh.clone().shape();

    let coords = &origin_shape.coordinates(origin_rank)?;
    let coord_dim = coords.iter().map(|(_, d)| *d).collect::<Vec<_>>();
    if !origin_mesh.clone().selection().contains(&coord_dim) {
        return Err(anyhow::Error::msg(format!(
            "rank {} is not selected in origin selection {}",
            origin_rank,
            origin_mesh.clone().selection()
        )));
    }

    verify_dimension_sizes(origin_shape, target_shape)?;

    match policy {
        AlignPolicy::Broadcast => Ok(target_mesh.clone()),
        AlignPolicy::Mapped { mode } => {
            let target_selection = &target_mesh.clone().selection();

            if matches!(mode, MappingMode::ExactMatch) {
                let origin_dims: std::collections::HashSet<_> =
                    origin_shape.labels().iter().collect();
                let target_dims: std::collections::HashSet<_> =
                    target_shape.labels().iter().collect();
                if origin_dims != target_dims {
                    return Err(anyhow::Error::msg(format!(
                        "dimension mismatch: origin has {:?}, target has {:?}",
                        origin_dims, target_dims
                    )));
                }
            }

            let target_ranges = match mode {
                MappingMode::BroadcastMissing => {
                    unimplemented!()
                }
                MappingMode::ExactMatch => exact_mapping(target_selection, coords, &origin_rank)?,
            };
            let new_selection = selection_from(target_shape, &target_ranges)?;
            let selection =
                Selection::reduce_intersection(target_mesh.clone().selection(), new_selection);
            Ok(ActorMeshRef::attest_with_selection(
                target_mesh.clone().into_mesh_id(),
                target_shape.clone(),
                selection,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use ndslice::selection::parse;

    use super::*;
    use crate::mesh_id;
    use crate::reference::ActorMeshRef;
    use crate::shape;
    use crate::test_utils::EmptyActor;

    macro_rules! compare_selections {
        ($a:expr, $b:expr) => {
            crate::selection::structurally_equal(&$a, &parse::expression($b).unwrap().1)
        };
    }

    #[test]
    fn test_broadcast() {
        let origin_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );
        let target_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );

        let mapping = map_meshes(&origin_mesh, 0, &target_mesh, AlignPolicy::Broadcast)
            .expect("broadcast mapping should succeed");
        assert_eq!(mapping.clone().shape(), target_mesh.clone().shape());
    }

    #[test]
    fn test_mapped_exact_match() {
        let origin_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );
        let target_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );

        let mapping = map_meshes(
            &origin_mesh,
            0,
            &target_mesh,
            AlignPolicy::Mapped {
                mode: MappingMode::ExactMatch,
            },
        )
        .expect("exact match mapping should succeed");
        assert_eq!(mapping.clone().shape(), target_mesh.clone().shape());
        compare_selections!(mapping.clone().selection(), "0, 0");

        let mapping = map_meshes(
            &origin_mesh,
            3,
            &target_mesh,
            AlignPolicy::Mapped {
                mode: MappingMode::ExactMatch,
            },
        )
        .expect("exact match mapping should succeed");
        compare_selections!(mapping.clone().selection(), "0, 2");

        let mapping = map_meshes(
            &origin_mesh,
            7,
            &target_mesh,
            AlignPolicy::Mapped {
                mode: MappingMode::ExactMatch,
            },
        )
        .expect("exact match mapping should succeed");
        compare_selections!(mapping.clone().selection(), "1, 3");
    }

    #[test]
    fn test_dimension_mismatch() {
        let origin_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );
        let target_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 3, host = 4 },
        );

        let result = map_meshes(
            &origin_mesh,
            0,
            &target_mesh,
            AlignPolicy::Mapped {
                mode: MappingMode::ExactMatch,
            },
        );
        assert!(
            result.is_err(),
            "should fail due to dimension size mismatch"
        );
    }

    #[test]
    #[should_panic(expected = "not implemented")]
    fn test_broadcast_missing_panic() {
        let origin_mesh =
            ActorMeshRef::<EmptyActor>::attest(mesh_id!(proc.actor), shape! { replica = 2 });
        let target_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );

        // This should panic because BroadcastMissing is not implemented
        map_meshes(
            &origin_mesh,
            0,
            &target_mesh,
            AlignPolicy::Mapped {
                mode: MappingMode::BroadcastMissing,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_empty_selection() {
        let origin_mesh = ActorMeshRef::<EmptyActor>::attest_with_selection(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
            Selection::False,
        );
        let target_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );

        let result = map_meshes(&origin_mesh, 0, &target_mesh, AlignPolicy::Broadcast);
        assert!(result.is_err(), "should fail due to empty origin selection");
    }

    #[test]
    fn test_out_of_range_rank() {
        let origin_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );
        let target_mesh = ActorMeshRef::<EmptyActor>::attest(
            mesh_id!(proc.actor),
            shape! { replica = 2, host = 4 },
        );

        let result = map_meshes(&origin_mesh, 10, &target_mesh, AlignPolicy::Broadcast);
        assert!(result.is_err(), "should fail due to out-of-range rank");
    }
}
