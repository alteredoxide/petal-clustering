use ndarray::{Array1, ArrayBase, ArrayView1, ArrayView2, Data, Ix2, Axis};
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{AddAssign, Div, DivAssign, Sub};
use succinct::{BitVecMut, BitVector};

use super::Fit;
use petal_neighbors::distance::{Euclidean, Metric};
use petal_neighbors::BallTree;

#[derive(Debug)]
pub enum HDbscanError {
    CondensedTreeNotComputed,  // when the min spanning tree has not been computed
}

impl std::fmt::Display for HDbscanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Self::CondensedTreeNotComputed => {
                write!( f, r#"The condensed tree has not been computed;
- ensure `store_condensed` is true,
- call `fit()` first."#)
            }
        }
    }
}


#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum ClusterSelection {
    /// Excess of Mass (EOM) method tends to select fewer, larger clusters.
    Eom,
    /// Tends to select a greater number of smaller, more granular clusters.
    Leaf,
}


#[derive(Debug, Deserialize, Serialize)]
pub struct HDbscan<A, M> {
    /// The radius of a neighborhood.
    pub eps: A,
    pub alpha: A,
    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,
    pub min_cluster_size: usize,
    pub metric: M,
    pub boruvka: bool,
    pub store_condensed: bool,
    pub condensed_tree: Option<Array1<(usize, usize, A, usize)>>,
    /// A mapping of the output (contiguous) cluster id to the original cluster id.
    pub cluster_relabels: HashMap<usize, usize>,
    pub allow_single_cluster: Option<bool>,
    pub cluster_selection_method: ClusterSelection,
}


impl<A, M> HDbscan<A, M>
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Sync + Send + TryFrom<u32>,
    <A as std::convert::TryFrom<u32>>::Error: Debug,
    M: Metric<A> + Clone + Sync + Send,
{
    /// Returns a vector of exemplars for a specified cluster id.
    ///
    /// # Returns
    ///
    /// - `Vec<usize>` where each element in the vec is the index of a single
    ///   datapoint in the input data.
    ///
    /// # Example
    /// ```
    /// # use ndarray::array;
    /// # use petal_clustering::{Fit, HDbscan, HDbscanError};
    /// 
    /// let data = array![
    ///     [-1.4, 2.0],
    ///     [-1.5, 2.1],
    ///     [-1.6, 2.0],
    ///     [-1.7, 2.0],
    ///
    ///     [1.7, 2.0],
    ///
    ///     [3.0, 2.0],
    ///     [3.1, 2.1],
    ///     [3.2, 2.1],
    ///
    ///     [4.0, 2.0],
    ///     [4.1, 2.1],
    ///     [4.2, 2.0],
    /// ];
    /// let mut hdbscan = HDbscan {
    ///     min_samples: 3,
    ///     min_cluster_size: 2,
    ///     store_condensed: true,
    ///     ..Default::default()
    /// };
    /// let (clusters, outliers) = hdbscan.fit(&data);
    /// let exemplars = match hdbscan.exemplars(0) {
    ///     Ok(ex) => ex,
    ///     Err(e) => panic!("{}", e)
    /// };
    /// # let expected = vec![1, 2];
    /// # assert_eq!(exemplars, expected);
    /// # // NOTE: the expected values can be confirmed by inspecting the
    /// # // condensed tree and seeing that 1 and 2 share the max lambda val for
    /// # // elemts of cluster id 12.
    /// ```
    pub fn exemplars(&self, cluster_id: usize)
        -> Result<Vec<usize>, HDbscanError>
    {
        let cluster_id = match self.cluster_relabels.get(&cluster_id) {
            Some(id) => *id,
            None => return Err(HDbscanError::CondensedTreeNotComputed)
        };
        if self.condensed_tree.is_none() {
            return Err(HDbscanError::CondensedTreeNotComputed)
        }
        let mut out = Vec::new();
        let leaves = match self.recurse_leaf_dfs(cluster_id) {
            Err(e) => return Err(e),
            Ok(l) => l
        };
        let condensed_tree = self.condensed_tree.as_ref().unwrap();
        for leaf in leaves {
            let leaf_max_lambda: A = condensed_tree
                .iter()
                .fold(A::zero(), |l_max, &v| {
                    match v.0 == leaf {
                        true => A::max(l_max, v.2),
                        false => l_max
                    }
                });
            let point_ids: Vec<usize> = condensed_tree
                .iter()
                .filter_map(|&v| {
                    match (v.0 == leaf) && (v.2 == leaf_max_lambda) {
                        true => Some(v.1),
                        false => None,
                    }
                })
                .collect();
            out.extend(point_ids)
        }
        Ok(out)
    }

    /// Returns a vector of all leaf nodes for a specified cluster id.
    fn recurse_leaf_dfs(&self, cluster_id: usize)
        -> Result<Vec<usize>, HDbscanError>
    {
        if self.condensed_tree.is_none() {
            return Err(HDbscanError::CondensedTreeNotComputed)
        }
        let cluster_tree: Array1<&(usize, usize, A, usize)> = self.condensed_tree
            .as_ref()
            .unwrap()
            .iter()
            .filter(|&v| v.3 > 1)
            .collect();
        let leaves = recurse_leaf_dfs(&cluster_tree, cluster_id);
        Ok(leaves)
    }
}

impl<A> Default for HDbscan<A, Euclidean>
where
    A: Float,
{
    #[must_use]
    fn default() -> Self {
        Self {
            eps: A::zero(),
            alpha: A::one(),
            min_samples: 15,
            min_cluster_size: 15,
            metric: Euclidean::default(),
            boruvka: true,
            store_condensed: false,
            condensed_tree: None,
            allow_single_cluster: None,
            cluster_relabels: HashMap::new(),
            cluster_selection_method: ClusterSelection::Eom,
        }
    }
}

fn relabel_clusters(
    clusters: HashMap<usize, Vec<usize>>,
) -> (HashMap<usize, Vec<usize>>, HashMap<usize, usize>) {
    let mut relabels = HashMap::new();
    let mut sorted_clusters = clusters.into_iter().collect::<Vec<_>>();
    sorted_clusters.sort_unstable_by_key(|v| v.0);
    let new_clusters = sorted_clusters
        .into_iter()
        .enumerate()
        .map(|(i, (id, cluster))| {
            relabels.insert(i, id);
            (i, cluster)
        })
        .collect();
    (new_clusters, relabels)
}

impl<S, A, M> Fit<ArrayBase<S, Ix2>, (HashMap<usize, Vec<usize>>, Vec<usize>)> for HDbscan<A, M>
where
    A: Debug + AddAssign + DivAssign + Float + FromPrimitive + Sync + Send + TryFrom<u32>,
    <A as std::convert::TryFrom<u32>>::Error: Debug,
    S: Data<Elem = A>,
    M: Metric<A> + Clone + Sync + Send,
{
    fn fit(&mut self, input: &ArrayBase<S, Ix2>) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        if input.is_empty() {
            return (HashMap::new(), Vec::new());
        }
        let input = input.as_standard_layout();
        let db = BallTree::new(input.view(), self.metric.clone()).expect("non-empty array");

        let mut mst = if self.boruvka {
            let boruvka = Boruvka::new(db, self.min_samples, self.alpha, self.eps);
            boruvka.min_spanning_tree().into_raw_vec()
        } else {
            let core_distances = Array1::from_vec(
                input
                    .rows()
                    .into_iter()
                    .map(|r| {
                        // plus one to account for the point itself
                        db.query(&r, self.min_samples + 1)
                            .1
                            .last()
                            .copied()
                            .expect("at least one point should be returned")
                    })
                    .collect(),
            );
            mst_linkage(
                input.view(),
                &self.metric,
                core_distances.view(),
                self.alpha,
            )
            .into_raw_vec()
        };

        mst.sort_unstable_by(|a, b| a.2.partial_cmp(&(b.2)).expect("invalid distance"));
        let sorted_mst = Array1::from_vec(mst);
        let labeled = label(sorted_mst);
        let condensed = Array1::from_vec(condense_mst(labeled.iter().collect(), self.min_cluster_size));
        let (clusters, outliers) = find_clusters(
            condensed.view(),
            &self.eps,
            self.cluster_selection_method,
            self.allow_single_cluster
        );
        if self.store_condensed {
            self.condensed_tree = Some(condensed)
        }
        let (new_clusters, cluster_relabels) = relabel_clusters(clusters);
        self.cluster_relabels = cluster_relabels;
        (new_clusters, outliers)
    }
}

/// Given a condensed tree and a starting cluster id, use recursive
/// depth-first-search to gather and return all leaf node ids.
fn recurse_leaf_dfs<A>(
    condensed_tree: &Array1<&(usize, usize, A, usize)>,
    current_node: usize
) -> Vec<usize>
    where
        A: AddAssign + DivAssign + Float + FromPrimitive + Sync + Send + TryFrom<u32>,
{
    let children = condensed_tree
        .iter()
        .filter_map(|v| if v.0 == current_node { Some(v.1) } else { None })
        .collect::<Vec<_>>();
    if children.is_empty() {
        return vec![current_node];
    } 
    let out = children.into_iter()
        .fold(vec![], |mut nodes, child| {
            nodes.extend(recurse_leaf_dfs(condensed_tree, child));
            nodes
        });
    out
}


fn mst_linkage<A: Float>(
    input: ArrayView2<A>,
    metric: &dyn Metric<A>,
    core_distances: ArrayView1<A>,
    alpha: A,
) -> Array1<(usize, usize, A)> {
    let nrows = input.nrows();

    assert_eq!(
        nrows,
        core_distances.len(),
        "dimensions of distance_metric and core_distances should match"
    );

    assert!(
        nrows >= 2,
        "dimensions of distance_metric and core_distances should be greater than 1"
    );

    let mut mst = Array1::<(usize, usize, A)>::uninit(nrows - 1);
    let mut in_tree: Vec<bool> = vec![false; nrows];
    let mut cur = 0;
    // edge uv: shortest_edges[v] = (mreachability_as_||uv||, u)
    // shortest as in shortest edges to v among  all nodes currently in tree
    let mut shortest_edges: Vec<(A, usize)> = vec![(A::max_value(), 1); nrows];

    for i in 0..nrows - 1 {
        // Add `cur` to tree
        in_tree[cur] = true;
        let core_cur = core_distances[cur];

        // next edge to add to tree
        let mut source: usize = 0;
        let mut next: usize = 0;
        let mut distance = A::max_value();

        for j in 0..nrows {
            if in_tree[j] {
                // skip if j is already in the tree
                continue;
            }

            let right = shortest_edges[j];
            let mut left = (metric.distance(&input.row(cur), &input.row(j)), cur);

            if alpha != A::from(1).expect("conversion failure") {
                left.0 = left.0 / alpha;
            } // convert distance matrix to `distance_metric / alpha` ?

            let core_j = core_distances[j];

            // right < MReachability_cur_j
            if (right.0 < core_cur || right.0 < left.0 || right.0 < core_j) && right.0 < distance {
                next = j;
                distance = right.0;
                source = right.1;
            }

            let tmp = if core_j > core_cur { core_j } else { core_cur };
            if tmp > left.0 {
                left.0 = tmp;
            }

            if left.0 < right.0 {
                shortest_edges[j] = left;
                if left.0 < distance {
                    distance = left.0;
                    source = left.1;
                    next = j;
                }
            } else if right.0 < distance {
                distance = right.0;
                source = right.1;
                next = j;
            }
        }

        mst[i] = MaybeUninit::new((source, next, distance)); // check MaybeUninit usage!
        cur = next;
    }

    unsafe { mst.assume_init() }
}

fn label<A: Float>(mst: Array1<(usize, usize, A)>) -> Array1<(usize, usize, A, usize)> {
    let n = mst.len() + 1;
    let mut uf = UnionFind::new(n);
    mst.into_iter()
        .map(|(mut a, mut b, delta)| {
            a = uf.fast_find(a);
            b = uf.fast_find(b);
            (a, b, delta, uf.union(a, b))
        })
        .collect()
}

fn condense_mst<A: Float + Div>(
    mst: Array1<&(usize, usize, A, usize)>,
    min_cluster_size: usize,
) -> Vec<(usize, usize, A, usize)> {
    let root = mst.len() * 2;
    let n = mst.len() + 1;

    let mut relabel = Array1::<usize>::uninit(root + 1);
    relabel[root] = MaybeUninit::new(n);
    let mut next_label = n + 1;
    let mut ignore = vec![false; root + 1];
    let mut result = Vec::new();

    let bsf = bfs_mst(&mst, root);
    for node in bsf {
        if node < n {
            continue;
        }
        if ignore[node] {
            continue;
        }
        let info = mst[node - n];
        let lambda = if info.2 > A::zero() {
            A::one() / info.2
        } else {
            A::max_value()
        };
        let left = info.0;
        let left_count = if left < n { 1 } else { mst[left - n].3 };

        let right = info.1;
        let right_count = if right < n { 1 } else { mst[right - n].3 };

        match (
            left_count >= min_cluster_size,
            right_count >= min_cluster_size,
        ) {
            (true, true) => {
                relabel[left] = MaybeUninit::new(next_label);
                result.push((
                    unsafe { relabel[node].assume_init() },
                    next_label,
                    lambda,
                    left_count,
                ));
                next_label += 1;

                relabel[right] = MaybeUninit::new(next_label);
                result.push((
                    unsafe { relabel[node].assume_init() },
                    next_label,
                    lambda,
                    right_count,
                ));
                next_label += 1;
            }
            (true, false) => {
                relabel[left] = relabel[node];
                for child in bfs_mst(&mst, right) {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
            (false, true) => {
                relabel[right] = relabel[node];
                for child in bfs_mst(&mst, left) {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
            (false, false) => {
                for child in bfs_mst(&mst, left).into_iter() {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
                for child in bfs_mst(&mst, right).into_iter() {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
        }
    }
    result
}

fn get_stability<A: Float + AddAssign + Sub + TryFrom<u32>>(
    condensed_tree: ArrayView1<(usize, usize, A, usize)>,
) -> HashMap<usize, A>
where
    <A as TryFrom<u32>>::Error: Debug,
{
    let mut births: HashMap<_, _> = condensed_tree.iter().fold(HashMap::new(), |mut births, v| {
        let entry = births.entry(v.1).or_insert(v.2);
        if *entry > v.2 {
            *entry = v.2;
        }
        births
    });

    let min_parent = condensed_tree
        .iter()
        .min_by_key(|v| v.0)
        .expect("couldn't find the smallest cluster")
        .0;

    let entry = births.entry(min_parent).or_insert_with(A::zero);
    *entry = A::zero();

    condensed_tree.iter().fold(
        HashMap::new(),
        |mut stability, (parent, _child, lambda, size)| {
            let entry = stability.entry(*parent).or_insert_with(A::zero);
            let birth = births.get(parent).expect("invalid child node.");
            *entry += (*lambda - *birth)
                * A::try_from(u32::try_from(*size).expect("out of bound")).expect("out of bound");
            stability
        },
    )
}

fn get_parent_child_distance<A: Float + AddAssign + Sub + TryFrom<u32>>(
    condensed_tree: &ArrayView1<(usize, usize, A, usize)>,
    parent: usize,
    child: usize,
) -> Option<A>
where
    <A as TryFrom<u32>>::Error: Debug,
{
    for (p, c, distance, _) in condensed_tree.iter() {
        if parent == *p && child == *c {
            return Some(*distance);
        }
    }
    None
}

fn traverse_upwards<A: Float + AddAssign + Sub + TryFrom<u32>>(
    cluster_tree: &Array1<&(usize, usize, A, usize)>,
    leaf: usize,
    epsilon: &A,
    allow_single_cluster: Option<bool>,
) -> usize
where
    <A as TryFrom<u32>>::Error: Debug,
{
    let root = cluster_tree.iter().map(|v| v.0).min().expect("no root found");
    let parent = cluster_tree.iter().find(|v| v.1 == leaf).expect("no parent found");
    if parent.0 == root {
        match allow_single_cluster {
            Some(true) => return root,
            _ => return leaf,
        }
    }
    let parent_lambda = parent.2;
    let parent_eps = A::one() / parent_lambda;
    if &parent_eps > epsilon {
        return parent.0;
    }
    traverse_upwards(cluster_tree, parent.0, epsilon, allow_single_cluster)
}

fn epsilon_search<A: Float + AddAssign + Sub + TryFrom<u32>>(
    leaves: HashSet<usize>,
    cluster_tree: &Array1<&(usize, usize, A, usize)>,
    epsilon: &A,
    allow_single_cluster: Option<bool>,
) -> HashSet<usize>
where
    <A as TryFrom<u32>>::Error: Debug,
{
    let mut selected_clusters = HashSet::new();
    let mut processed = HashSet::new();
    let tree_vec = cluster_tree.iter().map(|v| (v.0, v.1)).collect::<Vec<_>>();
    for leaf in leaves {
        let eps = A::one() / cluster_tree.iter()
            .find(|v| v.1 == leaf)
            .expect("no parent found").2;
        if &eps < epsilon {
            if !processed.contains(&leaf) {
                let cluster = traverse_upwards(cluster_tree, leaf, epsilon, allow_single_cluster);
                selected_clusters.insert(cluster);
                processed.insert(leaf);
                // remove all leaves that are in the same cluster
                for sub_node in bfs_mst(&cluster_tree, cluster) {
                    if sub_node != cluster {
                        processed.insert(sub_node);
                    }
                }
            } else {
                selected_clusters.insert(leaf);
            }
        }
    }
    selected_clusters
}

fn get_cluster_tree_leaves<A>(cluster_tree: &Array1<&(usize, usize, A, usize)>)
    -> Vec<usize>
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Sync + Send + TryFrom<u32>,
{
    if cluster_tree.is_empty() {
        return vec![];
    }
    let root = cluster_tree.iter().map(|v| v.0).min().expect("no root found");
    return recurse_leaf_dfs(&cluster_tree, root)
}

fn find_clusters<A>(
    condensed_tree: ArrayView1<(usize, usize, A, usize)>,
    epsilon: &A,
    cluster_selection_method: ClusterSelection,
    allow_single_cluster: Option<bool>,
) -> (HashMap<usize, Vec<usize>>, Vec<usize>)
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Sync + Send + TryFrom<u32>,
    <A as TryFrom<u32>>::Error: Debug,
{
    let mut stability = get_stability(condensed_tree);
    let mut nodes: Vec<_> = stability.keys().copied().collect();
    nodes.sort_unstable();
    nodes.reverse();
    match allow_single_cluster {
        Some(true) => {},
        // remove root otherwise
        _ => { nodes.remove(nodes.len() - 1); },
    }

    let tree: Vec<_> = condensed_tree
        .iter()
        .filter_map(|(p, c, _, s)| if *s > 1 { Some((*p, *c)) } else { None })
        .collect();

    let mut clusters: HashSet<_> = stability.keys().copied().collect();

    match cluster_selection_method {
        ClusterSelection::Eom => {
            for node in nodes {
                let subtree_stability = tree.iter().fold(A::zero(), |acc, (p, c)| {
                    if *p == node {
                        acc + *stability.get(c).expect("corruptted stability dictionary")
                    } else {
                        acc
                    }
                });

                stability.entry(node).and_modify(|v| {
                    if *v < subtree_stability {
                        clusters.remove(&node);
                        *v = subtree_stability;
                    } else {
                        let bfs = bfs_tree(&tree, node);
                        for child in bfs {
                            if child != node {
                                clusters.remove(&child);
                            }
                        }
                    }
                });
            }

            if epsilon != &A::zero() && !tree.is_empty() {
                let root = tree.iter().map(|v| v.0).min().expect("no root found");
                let allow_single_cluster = match allow_single_cluster {
                    Some(v) => v,
                    None => false,
                };
                if !(clusters.len() == 1 && clusters.contains(&root) && allow_single_cluster) {
                    clusters = epsilon_search(
                        clusters,
                        &condensed_tree.iter().collect(),
                        epsilon,
                        Some(allow_single_cluster)
                    );
                }
            }
        },
        ClusterSelection::Leaf => {
            let cluster_tree: Array1<&(usize, usize, A, usize)> = condensed_tree
                .iter()
                .filter(|&v| v.3 > 1)
                .collect();
            let leaves = get_cluster_tree_leaves(&cluster_tree);
            if leaves.is_empty() {
                clusters.clear();
                let root = condensed_tree.iter().map(|v| v.0).min().expect("no root found");
                clusters.insert(root);
            }
            if epsilon != &A::zero() {
                clusters = epsilon_search(
                    leaves.into_iter().collect(),
                    &cluster_tree,
                    epsilon,
                    allow_single_cluster
                );
            } else {
                clusters = leaves.into_iter().collect();
            }
        }
    }

    let mut clusters: Vec<_> = clusters.into_iter().collect();
    clusters.sort_unstable();
    let clusters: HashMap<_, _> = clusters
        .into_iter()
        .enumerate()
        .map(|(id, c)| (c, id))
        .collect();
    let max_parent = condensed_tree
        .iter()
        .max_by_key(|v| v.0)
        .expect("no maximum parent available")
        .0;
    let min_parent = condensed_tree
        .iter()
        .min_by_key(|v| v.0)
        .expect("no minimum parent available")
        .0;

    let mut uf = TreeUnionFind::new(max_parent + 1);
    for (parent, child, _, _) in condensed_tree {
        if !clusters.contains_key(child) {
            uf.union(*parent, *child);
        }
    }

    let mut res_clusters: HashMap<_, Vec<_>> = HashMap::new();
    let mut outliers = vec![];
    for n in 0..min_parent {
        let cluster = uf.find(n);
        if cluster > min_parent {
            let c = res_clusters.entry(cluster).or_default();
            c.push(n);
        } else {
            outliers.push(n);
        }
    }
    (res_clusters, outliers)
}

fn bfs_tree(tree: &[(usize, usize)], root: usize) -> Vec<usize> {
    let mut result = vec![];
    let mut to_process = HashSet::new();
    to_process.insert(root);
    while !to_process.is_empty() {
        result.extend(to_process.iter());
        to_process = tree
            .iter()
            .filter_map(|(p, c)| {
                if to_process.contains(p) {
                    Some(*c)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();
    }
    result
}

fn bfs_mst<A: Float>(mst: &Array1<&(usize, usize, A, usize)>, start: usize) -> Vec<usize> {
    let n = mst.len() + 1;

    let mut to_process = vec![start];
    let mut result = vec![];

    while !to_process.is_empty() {
        result.extend_from_slice(to_process.as_slice());
        to_process = to_process
            .into_iter()
            .filter_map(|x| {
                if x >= n {
                    Some(vec![mst[x - n].0, mst[x - n].1].into_iter())
                } else {
                    None
                }
            })
            .flatten()
            .collect();
    }
    result
}

#[allow(dead_code)]
#[derive(Debug)]
struct TreeUnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    is_component: BitVector<u64>,
}

#[allow(dead_code)]
impl TreeUnionFind {
    fn new(n: usize) -> Self {
        let parent = (0..n).collect();
        let size = vec![0; n];
        let is_component = BitVector::with_fill(
            u64::try_from(n).expect("fail to build a large enough bit vector"),
            true,
        );
        Self {
            parent,
            size,
            is_component,
        }
    }

    fn find(&mut self, x: usize) -> usize {
        assert!(x < self.parent.len());
        if x != self.parent[x] {
            self.parent[x] = self.find(self.parent[x]);
            self.is_component.set_bit(
                u64::try_from(x).expect("fail to convert usize to u64"),
                false,
            );
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let xx = self.find(x);
        let yy = self.find(y);

        match self.size[xx].cmp(&self.size[yy]) {
            Ordering::Greater => self.parent[yy] = xx,
            Ordering::Equal => {
                self.parent[yy] = xx;
                self.size[xx] += 1;
            }
            Ordering::Less => self.parent[xx] = yy,
        }
    }

    fn components(&self) -> Vec<usize> {
        self.is_component
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| if v { Some(idx) } else { None })
            .collect()
    }

    fn num_components(&self) -> usize {
        self.is_component.iter().filter(|b| *b).count()
    }
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    next_label: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let parent = (0..2 * n).collect();
        let size = vec![1]
            .into_iter()
            .cycle()
            .take(n)
            .chain(vec![0].into_iter().cycle().take(n - 1))
            .collect();
        Self {
            parent,
            size,
            next_label: n,
        }
    }

    fn union(&mut self, m: usize, n: usize) -> usize {
        self.parent[m] = self.next_label;
        self.parent[n] = self.next_label;
        let res = self.size[m] + self.size[n];
        self.size[self.next_label] = res;
        self.next_label += 1;
        res
    }

    fn fast_find(&mut self, mut n: usize) -> usize {
        let mut root = n;
        while self.parent[n] != n {
            n = self.parent[n];
        }
        while self.parent[root] != n {
            let tmp = self.parent[root];
            self.parent[root] = n;
            root = tmp;
        }
        n
    }
}

#[allow(dead_code)]
struct Boruvka<'a, A, M>
where
    A: Float,
    M: Metric<A>,
{
    db: BallTree<'a, A, M>,
    min_samples: usize,
    alpha: A,
    epsilon: A,
    candidates: Candidates<A>,
    components: Components,
    core_distances: Array1<A>,
    bounds: Vec<A>,
    mst: Vec<(usize, usize, A)>,
}

#[allow(dead_code)]
impl<'a, A, M> Boruvka<'a, A, M>
where
    A: Debug + Float + AddAssign + DivAssign + FromPrimitive + Sync + Send,
    M: Metric<A> + Sync + Send,
{
    fn new(db: BallTree<'a, A, M>, min_samples: usize, alpha: A, epsilon: A) -> Self {
        let mut candidates = Candidates::new(db.points.nrows());
        let components = Components::new(db.nodes.len(), db.points.nrows());
        let bounds = vec![A::max_value(); db.nodes.len()];
        // plus one on `min_samples` to account for the point itself
        let core_distances = compute_core_distances(&db, min_samples + 1, &mut candidates);
        let mst = Vec::with_capacity(db.points.nrows() - 1);
        Boruvka {
            db,
            min_samples,
            alpha,
            epsilon,
            candidates,
            components,
            core_distances,
            bounds,
            mst,
        }
    }

    fn min_spanning_tree(mut self) -> Array1<(usize, usize, A)> {
        let mut num_components = self.update_components();

        while num_components > 1 {
            self.traversal(0, 0);
            num_components = self.update_components();
        }
        Array1::from_vec(self.mst)
    }

    fn update_components(&mut self) -> usize {
        let components = self.components.get_current();
        for i in components {
            let (src, sink, dist) = match self.candidates.get(i) {
                Some((src, sink, dist)) => (src, sink, dist),
                None => continue,
            };

            if self.components.add(src, sink).is_none() {
                self.candidates.reset(i);
                continue;
            }

            self.candidates.distances[i] = A::max_value();

            self.mst.push((src, sink, dist));

            if self.mst.len() == self.db.num_points() - 1 {
                return self.components.len();
            }
        }
        self.components.update_points();
        for n in (0..self.db.num_nodes()).rev() {
            match self.db.children_of(n) {
                None => {
                    let mut points = self
                        .db
                        .points_of(n)
                        .iter()
                        .map(|i| self.components.point[*i]);
                    let pivot = points.next().expect("empty node");
                    if points.all(|c| c == pivot) {
                        self.components.node[n] =
                            u32::try_from(pivot).expect("overflow components");
                    }
                }
                Some((left, right)) => {
                    if self.components.node[left] == self.components.node[right]
                        && self.components.node[left] != u32::MAX
                    {
                        self.components.node[n] = self.components.node[left];
                    }
                }
            }
        }
        self.reset_bounds();
        self.components.len()
    }

    fn traversal(&mut self, query: usize, reference: usize) {
        // prune min{||query - ref||} >= bound_query
        let node_dist = self.db.node_distance_lower_bound(query, reference);
        if node_dist >= self.bounds[query] {
            return;
        }
        // prune when query and ref are in the same component
        if self.components.node[query] == self.components.node[reference]
            && self.components.node[query] != u32::MAX
        {
            return;
        }

        let query_children = self.db.children_of(query);
        let ref_children = self.db.children_of(reference);
        match (
            query_children,
            ref_children,
            self.db.compare_nodes(query, reference),
        ) {
            (None, None, _) => {
                let mut lower = A::max_value();
                let mut upper = A::zero();
                for &i in self.db.points_of(query) {
                    let c1 = self.components.point[i];
                    // mreach(i, j) >= core_i > candidate[c1]
                    // i.e. current best candidate for component c1 => prune
                    if self.core_distances[i] > self.candidates.distances[c1] {
                        continue;
                    }
                    for &j in self.db.points_of(reference) {
                        let c2 = self.components.point[j];
                        // mreach(i, j) >= core_j > candidate[c1] => prune
                        // i, j in the same component => prune
                        if self.core_distances[j] > self.candidates.distances[c1] || c1 == c2 {
                            continue;
                        }

                        let mut mreach = self
                            .db
                            .metric
                            .distance(&self.db.points.row(i), &self.db.points.row(j));
                        if self.alpha != A::one() {
                            mreach /= self.alpha;
                        }
                        if self.core_distances[j] > mreach {
                            mreach = self.core_distances[j];
                        }
                        if self.core_distances[i] > mreach {
                            mreach = self.core_distances[i];
                        }

                        if mreach < self.candidates.distances[c1] {
                            self.candidates.update(c1, (i, j, mreach));
                        }
                    }
                    if self.candidates.distances[c1] < lower {
                        lower = self.candidates.distances[c1];
                    }
                    if self.candidates.distances[c1] > upper {
                        upper = self.candidates.distances[c1];
                    }
                }

                let radius = self.db.radius_of(query);
                let mut bound = lower + radius + radius;
                if bound > upper {
                    bound = upper;
                }
                if bound < self.bounds[query] {
                    self.bounds[query] = bound;
                    let mut cur = query;
                    while cur > 0 {
                        let p = (cur - 1) / 2;
                        let new_bound = self.bound(p);
                        if new_bound >= self.bounds[p] {
                            break;
                        }
                        self.bounds[p] = new_bound;
                        cur = p;
                    }
                }
            }
            (None, Some((left, right)), _)
            | (_, Some((left, right)), Some(std::cmp::Ordering::Less)) => {
                let left_bound = self.db.node_distance_lower_bound(query, left);
                let right_bound = self.db.node_distance_lower_bound(query, right);

                if left_bound < right_bound {
                    self.traversal(query, left);
                    self.traversal(query, right);
                } else {
                    self.traversal(query, right);
                    self.traversal(query, left);
                }
            }
            (Some((left, right)), _, _) => {
                let left_bound = self.db.node_distance_lower_bound(reference, left);
                let right_bound = self.db.node_distance_lower_bound(reference, right);
                if left_bound < right_bound {
                    self.traversal(reference, left);
                    self.traversal(reference, right);
                } else {
                    self.traversal(reference, right);
                    self.traversal(reference, left);
                }
            }
        }
    }

    fn reset_bounds(&mut self) {
        self.bounds.iter_mut().for_each(|v| *v = A::max_value());
    }

    #[inline]
    fn lower_bound(&self, node: usize, parent: usize) -> A {
        let diff = self.db.radius_of(parent) - self.db.radius_of(node);
        self.bounds[node] + diff + diff
    }

    #[inline]
    fn bound(&self, parent: usize) -> A {
        let left = 2 * parent + 1;
        let right = left + 1;

        let upper = if self.bounds[left] > self.bounds[right] {
            self.bounds[left]
        } else {
            self.bounds[right]
        };

        let lower_left = self.lower_bound(left, parent);
        let lower_right = self.lower_bound(right, parent);
        let lower = if lower_left > lower_right {
            lower_right
        } else {
            lower_left
        };

        if lower > A::zero() && lower < upper {
            lower
        } else {
            upper
        }
    }
}

// core_distances: distance of center to min_samples' closest point (including the center).
fn compute_core_distances<A, M>(
    db: &BallTree<A, M>,
    min_samples: usize,
    candidates: &mut Candidates<A>,
) -> Array1<A>
where
    A: AddAssign + DivAssign + FromPrimitive + Float + Sync + Send,
    M: Metric<A> + Sync + Send,
{
    let mut knn_indices = vec![0; db.points.nrows() * min_samples];
    let mut core_distances = vec![A::zero(); db.points.nrows()];
    let rows: Vec<(usize, (&mut [usize], &mut A))> = knn_indices
        .chunks_mut(min_samples)
        .zip(core_distances.iter_mut())
        .enumerate()
        .collect();
    rows.into_par_iter().for_each(|(i, (indices, dist))| {
        let row = db.points.row(i);
        let (idx, d) = db.query(&row, min_samples);
        indices.clone_from_slice(&idx);
        *dist = *d.last().expect("ball tree query failed");
    });

    knn_indices
        .chunks_exact(min_samples)
        .enumerate()
        .for_each(|(n, row)| {
            for val in row.iter().skip(1).rev() {
                if core_distances[*val] <= core_distances[n] {
                    candidates.update(n, (n, *val, core_distances[n]));
                }
            }
        });

    Array1::from_vec(core_distances)
}

#[allow(dead_code)]
struct Candidates<A> {
    points: Vec<u32>,
    neighbors: Vec<u32>,
    distances: Vec<A>,
}

#[allow(dead_code)]
impl<A: Float> Candidates<A> {
    fn new(n: usize) -> Self {
        // define max_value as NULL
        let neighbors = vec![u32::max_value(); n];
        // define max_value as NULL
        let points = vec![u32::max_value(); n];
        // define max_value as infinite far
        let distances = vec![A::max_value(); n];
        Self {
            points,
            neighbors,
            distances,
        }
    }

    fn get(&self, i: usize) -> Option<(usize, usize, A)> {
        if self.is_undefined(i) {
            None
        } else {
            Some((
                usize::try_from(self.points[i]).expect("fail to convert points"),
                usize::try_from(self.neighbors[i]).expect("fail to convert neighbor"),
                self.distances[i],
            ))
        }
    }

    fn update(&mut self, i: usize, val: (usize, usize, A)) {
        self.distances[i] = val.2;
        self.points[i] = u32::try_from(val.0).expect("candidate index overflow");
        self.neighbors[i] = u32::try_from(val.1).expect("candidate index overflow");
    }

    fn reset(&mut self, i: usize) {
        self.points[i] = u32::max_value();
        self.neighbors[i] = u32::max_value();
        self.distances[i] = A::max_value();
    }

    fn is_undefined(&self, i: usize) -> bool {
        self.points[i] == u32::max_value() || self.neighbors[i] == u32::max_value()
    }
}

#[allow(dead_code)]
struct Components {
    point: Vec<usize>,
    node: Vec<u32>,
    uf: TreeUnionFind,
}

#[allow(dead_code)]
impl Components {
    fn new(m: usize, n: usize) -> Self {
        // each point started as its own component.
        let point = (0..n).collect();
        // the component of the node is concluded when
        // all the enclosed points are in the same component
        let node = vec![u32::MAX; m];
        let uf = TreeUnionFind::new(n);
        Self { point, node, uf }
    }

    fn add(&mut self, src: usize, sink: usize) -> Option<()> {
        let current_src = self.uf.find(src);
        let current_sink = self.uf.find(sink);
        if current_src == current_sink {
            return None;
        }
        self.uf.union(current_src, current_sink);
        Some(())
    }

    fn update_points(&mut self) {
        for i in 0..self.point.len() {
            self.point[i] = self.uf.find(i);
        }
    }

    fn get_current(&self) -> Vec<usize> {
        self.uf.components()
    }

    fn len(&self) -> usize {
        self.uf.num_components()
    }
}

mod test {
    use std::collections::HashMap;


    #[test]
    fn hdbscan() {
        use crate::Fit;
        use ndarray::array;
        use petal_neighbors::distance::Euclidean;

        let data = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let mut hdbscan = super::HDbscan {
            eps: 0.,
            alpha: 1.,
            min_samples: 1,
            min_cluster_size: 2,
            metric: Euclidean::default(),
            boruvka: true,
            ..Default::default()
        };
        let (clusters, outliers) = hdbscan.fit(&data);
        assert_eq!(clusters.len(), 2);
        assert_eq!(
            outliers.len(),
            data.nrows() - clusters.values().fold(0, |acc, v| acc + v.len())
        );
    }

    #[test]
    fn hdbscan_leaf() {
        use crate::Fit;
        use ndarray::array;
        use petal_neighbors::distance::Euclidean;

        let data = array![
            [ 0.06773077,  9.08570996],
            [ 3.49302242, -7.87654764],
            [-2.43203197,  7.46577602],
            [-0.77757877,  8.73031386],
            [-2.47912623,  8.7266127 ],
            [ 2.15570162, -7.68741574],
            [ 4.4471007 , -5.41556534],
            [ 4.21618717, -6.4725295 ],
            [-1.31953417,  9.76281499],
            [ 1.90393137, -6.8904547 ],
            [ 4.89948794, -6.74059989]
        ];
        let expected_eom = HashMap::from([
            (0, vec![0, 2, 3, 4, 8]),
            (1, vec![1, 5, 6, 7, 9, 10]),
        ]);
        let expected_leaf = HashMap::from([
            (0, vec![0, 2, 3, 4, 8]),
            (1, vec![1, 5]),
            (2, vec![6, 7, 10]),
        ]);
        let mut hdbscan_eom = super::HDbscan {
            eps: 0.,
            alpha: 1.,
            min_samples: 2,
            min_cluster_size: 2,
            metric: Euclidean::default(),
            boruvka: true,
            cluster_selection_method: super::ClusterSelection::Eom,
            ..Default::default()
        };
        let mut hdbscan_leaf = super::HDbscan {
            eps: 0.,
            alpha: 1.,
            min_samples: 2,
            min_cluster_size: 2,
            metric: Euclidean::default(),
            boruvka: true,
            cluster_selection_method: super::ClusterSelection::Leaf,
            ..Default::default()
        };
        let (clusters_eom, _) = hdbscan_eom.fit(&data);
        let (clusters_leaf, _) = hdbscan_leaf.fit(&data);
        assert_eq!(clusters_eom, expected_eom);
        assert_eq!(clusters_leaf, expected_leaf);
    }

    #[test]
    fn recurse_leaf_dfs() {
        use ndarray::array;

        let condensed_tree = array![
            (0, 1, 0.0, 0),
            (0, 2, 0.0, 0),
            (2, 3, 0.0, 0),
            (3, 4, 0.0, 0),
        ];
        let descendants = super::recurse_leaf_dfs(
            &condensed_tree.iter().collect(),
            0
        );
        assert_eq!(descendants, vec![1, 4]);
    }

    #[test]
    fn mst_linkage() {
        use ndarray::{arr1, arr2};
        use petal_neighbors::distance::Euclidean;
        //  0, 1, 2, 3, 4, 5, 6
        // {A, B, C, D, E, F, G}
        // {AB = 7, AD = 5,
        //  BC = 8, BD = 9, BE = 7,
        //  CB = 8, CE = 5,
        //  DB = 9, DE = 15, DF = 6,
        //  EF = 8, EG = 9
        //  FG = 11}
        let input = arr2(&[
            [0., 0.],
            [7., 0.],
            [15., 0.],
            [0., -5.],
            [15., -5.],
            [7., -7.],
            [15., -14.],
        ]);
        let core_distances = arr1(&[5., 7., 5., 5., 5., 6., 9.]);
        let mst = super::mst_linkage(
            input.view(),
            &Euclidean::default(),
            core_distances.view(),
            1.,
        );
        let answer = arr1(&[
            (0, 3, 5.),
            (0, 1, 7.),
            (1, 5, 7.),
            (1, 2, 8.),
            (2, 4, 5.),
            (4, 6, 9.),
        ]);
        assert_eq!(mst, answer);
    }

    #[test]
    fn boruvka() {
        use ndarray::{arr1, arr2};
        use petal_neighbors::{distance::Euclidean, BallTree};

        let input = arr2(&[
            [0., 0.],
            [7., 0.],
            [15., 0.],
            [0., -5.],
            [15., -5.],
            [7., -7.],
            [15., -14.],
        ]);

        let db = BallTree::new(input, Euclidean::default()).unwrap();
        let boruvka = super::Boruvka::new(db, 1, 1., 0.);
        let mst = boruvka.min_spanning_tree();

        let answer = arr1(&[
            (0, 3, 5.0),
            (1, 0, 7.0),
            (2, 4, 5.0),
            (5, 1, 7.0),
            (6, 4, 9.0),
            (1, 2, 8.0),
        ]);
        assert_eq!(answer, mst);
    }

    #[test]
    fn tree_union_find() {
        use succinct::{BitVecMut, BitVector};

        let parent = vec![0, 0, 1, 2, 4];
        let size = vec![0; 5];
        let is_component = BitVector::with_fill(5, true);
        let mut uf = super::TreeUnionFind {
            parent,
            size,
            is_component,
        };
        assert_eq!(0, uf.find(3));
        assert_eq!(vec![0, 0, 0, 0, 4], uf.parent);
        uf.union(4, 0);
        assert_eq!(vec![4, 0, 0, 0, 4], uf.parent);
        assert_eq!(vec![0, 0, 0, 0, 1], uf.size);
        let mut bv = BitVector::with_fill(5, false);
        bv.set_bit(0, true);
        bv.set_bit(4, true);
        assert_eq!(bv, uf.is_component);
        assert_eq!(vec![0, 4], uf.components());

        uf = super::TreeUnionFind::new(3);
        assert_eq!((0..3).collect::<Vec<_>>(), uf.parent);
        assert_eq!(vec![0; 3], uf.size);
    }

    #[test]
    fn union_find() {
        let mut uf = super::UnionFind::new(7);
        let pairs = vec![(0, 3), (4, 2), (3, 5), (0, 1), (1, 4), (4, 6)];
        let uf_res: Vec<_> = pairs
            .into_iter()
            .map(|(l, r)| {
                let ll = uf.fast_find(l);
                let rr = uf.fast_find(r);
                (ll, rr, uf.union(ll, rr))
            })
            .collect();
        assert_eq!(
            uf_res,
            vec![
                (0, 3, 2),
                (4, 2, 2),
                (7, 5, 3),
                (9, 1, 4),
                (10, 8, 6),
                (11, 6, 7)
            ]
        )
    }

    #[test]
    fn label() {
        use ndarray::arr1;
        let mst = arr1(&[
            (0, 3, 5.),
            (4, 2, 5.),
            (3, 5, 6.),
            (0, 1, 7.),
            (1, 4, 7.),
            (4, 6, 9.),
        ]);
        let labeled_mst = super::label(mst);
        assert_eq!(
            labeled_mst,
            arr1(&[
                (0, 3, 5., 2),
                (4, 2, 5., 2),
                (7, 5, 6., 3),
                (9, 1, 7., 4),
                (10, 8, 7., 6),
                (11, 6, 9., 7)
            ])
        );
    }

    #[test]
    fn bfs_mst() {
        use ndarray::arr1;
        let mst = arr1(&[
            (0, 3, 5., 2),
            (4, 2, 5., 2),
            (7, 5, 6., 3),
            (9, 1, 7., 4),
            (10, 8, 7., 6),
            (11, 6, 9., 7),
        ]);
        let root = mst.len() * 2;
        let mst_ref = mst.iter().collect();
        let bfs = super::bfs_mst(&mst_ref, root);
        assert_eq!(bfs, [12, 11, 6, 10, 8, 9, 1, 4, 2, 7, 5, 0, 3]);

        let bfs = super::bfs_mst(&mst_ref, 11);
        assert_eq!(bfs, vec![11, 10, 8, 9, 1, 4, 2, 7, 5, 0, 3]);

        let bfs = super::bfs_mst(&mst_ref, 8);
        assert_eq!(bfs, vec![8, 4, 2]);
    }

    #[test]
    fn condense_mst() {
        use ndarray::arr1;

        let mst = arr1(&[
            (0, 3, 5., 2),
            (4, 2, 5., 2),
            (7, 5, 6., 3),
            (9, 1, 7., 4),
            (10, 8, 7., 6),
            (11, 6, 9., 7),
        ]);

        let condensed_mst = super::condense_mst(mst.iter().collect(), 3);
        assert_eq!(
            condensed_mst,
            vec![
                (7, 6, 1. / 9., 1),
                (7, 4, 1. / 7., 1),
                (7, 2, 1. / 7., 1),
                (7, 1, 1. / 7., 1),
                (7, 5, 1. / 6., 1),
                (7, 0, 1. / 6., 1),
                (7, 3, 1. / 6., 1)
            ],
        );
    }

    #[test]
    fn get_stability() {
        use ndarray::arr1;
        use std::collections::HashMap;

        let condensed = arr1(&[
            (7, 6, 1. / 9., 1),
            (7, 4, 1. / 7., 1),
            (7, 2, 1. / 7., 1),
            (7, 1, 1. / 7., 1),
            (7, 5, 1. / 6., 1),
            (7, 0, 1. / 6., 1),
            (7, 3, 1. / 6., 1),
        ]);
        let stability_map = super::get_stability(condensed.view());
        let mut answer = HashMap::new();
        answer.insert(7, 1. / 9. + 3. / 7. + 3. / 6.);
        assert_eq!(stability_map, answer);
    }
}
