#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic K estimation utilities for AND."""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse.csgraph import laplacian
from scipy.stats import jarque_bera
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

from packages.config import CONFIG as cfg
from packages.loggers.std_logger import STDLogger as logger
from packages.register import REGISTER


def require_args():
    """Register CLI arguments for the Auto-K head."""

    cfg.add_argument('--auto-k-enable', action='store_true',
                     help='Enable automatic estimation of the number of clusters.')
    cfg.add_argument('--auto-k-method', default='consensus', type=str,
                     help='Primary Auto-K heuristic to use. ')
    cfg.add_argument('--auto-k-method-cycle', default='', type=str,
                     help='Comma separated list of heuristics to cycle through during training.')
    cfg.add_argument('--auto-k-cycle-level', default='round', type=str,
                     choices=['round', 'epoch'],
                     help='Whether heuristic cycling happens per round or per epoch.')
    cfg.add_argument('--auto-k-min-k', default=2, type=int,
                     help='Minimum number of clusters considered by Auto-K.')
    cfg.add_argument('--auto-k-max-k', default=50, type=int,
                     help='Maximum number of clusters considered by Auto-K.')
    cfg.add_argument('--auto-k-sample-size', default=2048, type=int,
                     help='Maximum number of feature vectors evaluated by Auto-K heuristics.')
    cfg.add_argument('--auto-k-dp-lambda', default=0.5, type=float,
                     help='Penalty parameter for DP-means heuristic.')
    cfg.add_argument('--auto-k-dp-max-iter', default=50, type=int,
                     help='Maximum DP-means refinement iterations.')
    cfg.add_argument('--auto-k-consensus-methods', default='', type=str,
                     help='Subset of heuristics participating in consensus voting.')
    cfg.add_argument('--auto-k-eigengap-neighbors', default=10, type=int,
                     help='Number of neighbours used to build the graph for eigengap heuristic.')
    cfg.add_argument('--auto-k-xmeans-bic-threshold', default=0.0, type=float,
                     help='Minimum BIC gain required to accept a split in the X-means heuristic.')
    cfg.add_argument('--auto-k-max-depth', default=8, type=int,
                     help='Maximum recursion depth for hierarchical heuristics.')
    cfg.add_argument('--auto-k-gmeans-threshold', default=5.991, type=float,
                     help='Jarque-Bera statistic threshold used by the G-means heuristic.')
    cfg.add_argument('--auto-k-log-embeddings', action='store_true',
                     help='Log sampled embeddings and cluster assignments to TensorBoard.')


class AutoKHead:
    """Collection of heuristics for estimating the number of clusters."""

    SUPPORTED_METHODS = (
        'consensus', 'silhouette', 'eigengap', 'dpmeans', 'xmeans', 'gmeans'
    )

    def __init__(
        self,
        min_k: int = 2,
        max_k: int = 50,
        method: str = 'consensus',
        method_cycle: Optional[Iterable[str]] = None,
        cycle_level: str = 'round',
        consensus_methods: Optional[Iterable[str]] = None,
        sample_size: Optional[int] = None,
        dp_lambda: float = 0.5,
        dp_max_iter: int = 50,
        eigengap_neighbors: int = 10,
        xmeans_bic_threshold: float = 0.0,
        max_depth: int = 8,
        gmeans_threshold: float = 5.991,
        log_embeddings: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.min_k = max(1, min_k)
        self.max_k = max(self.min_k, max_k)
        if isinstance(method_cycle, str):
            method_cycle = [m.strip() for m in method_cycle.split(',') if m.strip()]
        self.default_method = method if method in self.SUPPORTED_METHODS else 'consensus'
        self.method_cycle = [m for m in (method_cycle or []) if m in self.SUPPORTED_METHODS]
        self.cycle_level = cycle_level
        self.sample_size = sample_size
        self.dp_lambda = max(1e-6, dp_lambda)
        self.dp_max_iter = max(1, dp_max_iter)
        self.eigengap_neighbors = max(1, eigengap_neighbors)
        self.xmeans_bic_threshold = xmeans_bic_threshold
        self.max_depth = max(1, max_depth)
        self.gmeans_threshold = gmeans_threshold
        self.log_embeddings = log_embeddings
        self.random_state = np.random.RandomState(random_state or cfg.seed)
        self.consensus_methods = self._prepare_consensus_methods(consensus_methods)
        self.method_map = {
            'consensus': self._consensus,
            'silhouette': self._silhouette,
            'eigengap': self._eigengap,
            'dpmeans': self._dpmeans,
            'xmeans': self._xmeans,
            'gmeans': self._gmeans,
        }

    def _prepare_consensus_methods(
        self, methods: Optional[Iterable[str]]
    ) -> List[str]:
        if methods is None:
            return [m for m in self.SUPPORTED_METHODS if m != 'consensus']
        if isinstance(methods, str):
            methods = [m.strip() for m in methods.split(',') if m.strip()]
        return [m for m in methods if m in self.SUPPORTED_METHODS and m != 'consensus']

    def resolve_method(
        self,
        round_idx: Optional[int] = None,
        epoch_idx: Optional[int] = None,
        requested: Optional[str] = None,
    ) -> str:
        """Determine which heuristic should be executed for the current step."""

        if requested in self.SUPPORTED_METHODS:
            return requested
        if self.method_cycle:
            if self.cycle_level == 'epoch' and epoch_idx is not None:
                index = epoch_idx
            else:
                index = round_idx if round_idx is not None else 0
            return self.method_cycle[index % len(self.method_cycle)]
        return self.default_method

    def estimate(
        self,
        features: torch.Tensor,
        round_idx: Optional[int] = None,
        epoch_idx: Optional[int] = None,
        method: Optional[str] = None,
    ) -> Tuple[int, Dict[str, object]]:
        method_name = self.resolve_method(round_idx, epoch_idx, method)
        feats, indices = self._prepare_features(features)
        heuristic = self.method_map[method_name]
        est_k, info = heuristic(feats, indices)
        info.setdefault('assignments', None)
        info.setdefault('features', feats)
        info.setdefault('indices', indices)
        info['method'] = method_name
        info['sample_size'] = feats.shape[0]
        info['estimated_k'] = int(est_k)
        return int(est_k), info

    # ------------------------------------------------------------------
    # Heuristic implementations
    # ------------------------------------------------------------------
    def _silhouette(self, features: np.ndarray, indices: np.ndarray) -> Tuple[int, Dict[str, object]]:
        n_samples = features.shape[0]
        if n_samples < 2:
            return 1, {'score': float('nan')}

        upper = min(self.max_k, n_samples - 1)
        best_score = -np.inf
        best_labels = np.zeros(n_samples, dtype=np.int32)
        best_centers = np.mean(features, axis=0, keepdims=True)
        scores = {}
        for k in range(self.min_k, upper + 1):
            if k <= 1:
                continue
            try:
                kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300,
                                 random_state=self.random_state)
                labels = kmeans.fit_predict(features)
            except Exception as err:
                logger.debug('Silhouette heuristic failed for K=%d with error: %s', k, err)
                continue
            unique_labels = np.unique(labels)
            if unique_labels.size <= 1:
                continue
            score = silhouette_score(features, labels, metric='euclidean')
            scores[k] = score
            if score > best_score:
                best_score = score
                best_labels = labels
                best_centers = kmeans.cluster_centers_
        if not scores:
            return max(self.min_k, 1), {
                'score': float('nan'),
                'assignments': best_labels,
                'centers': best_centers,
            }
        best_k = max(scores.items(), key=lambda kv: kv[1])[0]
        return best_k, {
            'score': scores[best_k],
            'assignments': best_labels,
            'centers': best_centers,
        }

    def _eigengap(self, features: np.ndarray, indices: np.ndarray) -> Tuple[int, Dict[str, object]]:
        n_samples = features.shape[0]
        if n_samples <= 1:
            return 1, {}
        neighbours = min(self.eigengap_neighbors, n_samples - 1)
        if neighbours <= 0:
            return 1, {}
        try:
            knn_graph = kneighbors_graph(features, neighbours, mode='connectivity', include_self=False)
            adjacency = 0.5 * (knn_graph + knn_graph.T)
            laplacian_matrix = laplacian(adjacency, normed=True)
            if hasattr(laplacian_matrix, 'toarray'):
                laplacian_matrix = laplacian_matrix.toarray()
            eigvals = np.linalg.eigvalsh(laplacian_matrix)
        except Exception as err:
            logger.debug('Eigengap heuristic failed: %s', err)
            return self.min_k, {}
        eigvals = np.sort(np.real(eigvals))
        max_rank = min(self.max_k + 1, eigvals.size)
        if max_rank < 2:
            return self.min_k, {'spectrum': eigvals[:max_rank]}
        gaps = np.diff(eigvals[:max_rank])
        best_k = np.argmax(gaps) + 1
        best_k = int(np.clip(best_k, self.min_k, self.max_k))
        return best_k, {
            'spectrum': eigvals[:max_rank],
            'largest_gap': gaps[best_k - 1] if gaps.size >= best_k else float('nan'),
        }

    def _dpmeans(self, features: np.ndarray, indices: np.ndarray) -> Tuple[int, Dict[str, object]]:
        centers = [features.mean(axis=0)]
        assignments = np.zeros(features.shape[0], dtype=np.int32)
        objective = 0.0
        for _ in range(self.dp_max_iter):
            changed = False
            for idx, vector in enumerate(features):
                distances = np.array([
                    np.linalg.norm(vector - center) ** 2 for center in centers
                ])
                min_idx = np.argmin(distances)
                if distances[min_idx] > self.dp_lambda and len(centers) < self.max_k:
                    centers.append(vector.copy())
                    assignments[idx] = len(centers) - 1
                    changed = True
                else:
                    if assignments[idx] != min_idx:
                        assignments[idx] = min_idx
                        changed = True
            centers = self._update_centers(features, assignments, len(centers))
            if not changed:
                break
        k = int(len(centers))
        for idx, vector in enumerate(features):
            objective += np.linalg.norm(vector - centers[assignments[idx]]) ** 2
        objective += self.dp_lambda * k
        return k, {
            'assignments': assignments,
            'centers': np.asarray(centers),
            'objective': objective,
        }

    def _xmeans(self, features: np.ndarray, indices: np.ndarray) -> Tuple[int, Dict[str, object]]:
        n_samples = features.shape[0]
        base_k = min(max(self.min_k, 1), min(self.max_k, n_samples))
        kmeans = KMeans(n_clusters=base_k, n_init=10, max_iter=300,
                        random_state=self.random_state)
        assignments = kmeans.fit_predict(features)
        centers = [center for center in kmeans.cluster_centers_]
        queue: deque[Tuple[int, np.ndarray, int]] = deque()
        for cid in range(base_k):
            idxs = np.where(assignments == cid)[0]
            queue.append((cid, idxs, 0))
        next_label = base_k
        while queue:
            cid, idxs, depth = queue.popleft()
            if depth >= self.max_depth or idxs.size <= 2 or next_label >= self.max_k:
                continue
            data = features[idxs]
            child_kmeans = KMeans(n_clusters=2, n_init=5, max_iter=200,
                                  random_state=self.random_state)
            child_labels = child_kmeans.fit_predict(data)
            parent_bic = self._bic_score(data, np.zeros_like(child_labels),
                                         [data.mean(axis=0)])
            split_bic = self._bic_score(data, child_labels,
                                        child_kmeans.cluster_centers_)
            if split_bic - parent_bic > self.xmeans_bic_threshold:
                zero_indices = idxs[child_labels == 0]
                one_indices = idxs[child_labels == 1]
                assignments[zero_indices] = cid
                assignments[one_indices] = next_label
                centers[cid] = child_kmeans.cluster_centers_[0]
                centers.append(child_kmeans.cluster_centers_[1])
                queue.append((cid, zero_indices, depth + 1))
                queue.append((next_label, one_indices, depth + 1))
                next_label += 1
        assignments, centers = self._reindex(assignments, centers)
        return int(len(centers)), {
            'assignments': assignments,
            'centers': np.asarray(centers),
        }

    def _gmeans(self, features: np.ndarray, indices: np.ndarray) -> Tuple[int, Dict[str, object]]:
        n_samples = features.shape[0]
        assignments = np.zeros(n_samples, dtype=np.int32)
        centers = [features.mean(axis=0)]
        queue: deque[Tuple[int, np.ndarray, int]] = deque([(0, np.arange(n_samples), 0)])
        next_label = 1
        while queue:
            cid, idxs, depth = queue.popleft()
            if depth >= self.max_depth or idxs.size <= 2 or next_label >= self.max_k:
                assignments[idxs] = cid
                continue
            data = features[idxs]
            kmeans = KMeans(n_clusters=2, n_init=5, max_iter=200,
                            random_state=self.random_state)
            split_labels = kmeans.fit_predict(data)
            direction = kmeans.cluster_centers_[1] - kmeans.cluster_centers_[0]
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                assignments[idxs] = cid
                continue
            unit_direction = direction / norm
            projections = np.dot(data, unit_direction)
            jb_stat, jb_p = jarque_bera(projections)
            if jb_stat > self.gmeans_threshold and next_label < self.max_k:
                zero_indices = idxs[split_labels == 0]
                one_indices = idxs[split_labels == 1]
                assignments[zero_indices] = cid
                assignments[one_indices] = next_label
                if cid >= len(centers):
                    centers.append(kmeans.cluster_centers_[0])
                else:
                    centers[cid] = kmeans.cluster_centers_[0]
                centers.append(kmeans.cluster_centers_[1])
                queue.append((cid, zero_indices, depth + 1))
                queue.append((next_label, one_indices, depth + 1))
                next_label += 1
            else:
                assignments[idxs] = cid
                if cid >= len(centers):
                    centers.append(kmeans.cluster_centers_[0])
                else:
                    centers[cid] = kmeans.cluster_centers_[0]
        assignments, centers = self._reindex(assignments, centers)
        return int(len(centers)), {
            'assignments': assignments,
            'centers': np.asarray(centers),
        }

    def _consensus(self, features: np.ndarray, indices: np.ndarray) -> Tuple[int, Dict[str, object]]:
        votes: Dict[str, int] = {}
        metrics: Dict[str, object] = {}
        for method in self.consensus_methods:
            estimator = self.method_map[method]
            est_k, info = estimator(features, indices)
            votes[method] = int(est_k)
            for key, value in info.items():
                metrics[f'{method}_{key}'] = value
        if not votes:
            return self.min_k, {'votes': votes}
        counter = Counter(votes.values())
        best_k = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        best_k = int(np.clip(best_k, self.min_k, self.max_k))
        assignments = None
        if best_k > 0:
            try:
                kmeans = KMeans(n_clusters=best_k, n_init=10, max_iter=300,
                                random_state=self.random_state)
                assignments = kmeans.fit_predict(features)
                metrics['centers'] = kmeans.cluster_centers_
            except Exception as err:
                logger.debug('Consensus refinement failed: %s', err)
        metrics['votes'] = votes
        metrics['assignments'] = assignments
        return best_k, metrics

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _prepare_features(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(features, torch.Tensor):
            feats = features.detach().cpu().numpy()
        else:
            feats = np.asarray(features)
        feats = np.nan_to_num(feats, copy=False)
        n_samples = feats.shape[0]
        if self.sample_size and n_samples > self.sample_size:
            indices = self.random_state.choice(n_samples, self.sample_size, replace=False)
            feats = feats[indices]
        else:
            indices = np.arange(n_samples)
        return feats.astype(np.float32), indices

    @staticmethod
    def _update_centers(features: np.ndarray, assignments: np.ndarray, k: int) -> List[np.ndarray]:
        centers: List[np.ndarray] = []
        for idx in range(k):
            members = features[assignments == idx]
            if members.size == 0:
                continue
            centers.append(np.mean(members, axis=0))
        if not centers:
            centers.append(np.mean(features, axis=0))
        return centers

    @staticmethod
    def _reindex(assignments: np.ndarray, centers: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        unique_labels = np.unique(assignments)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        new_assignments = np.vectorize(mapping.get)(assignments)
        new_centers = [centers[label] for label in unique_labels if label < len(centers)]
        return new_assignments.astype(np.int32), new_centers

    @staticmethod
    def _bic_score(features: np.ndarray, labels: np.ndarray,
                   centers: Iterable[np.ndarray]) -> float:
        centers = list(centers)
        n_clusters = len(centers)
        n_samples, n_features = features.shape
        if n_clusters == 0:
            return -math.inf
        variance = 0.0
        for idx, center in enumerate(centers):
            members = features[labels == idx]
            if members.size == 0:
                continue
            variance += ((members - center) ** 2).sum()
        variance = max(variance, 1e-6)
        variance /= max(n_samples - n_clusters, 1)
        log_likelihood = -0.5 * n_samples * (
            n_features * math.log(2 * math.pi * variance) + 1
        )
        num_params = n_clusters * (n_features + 1)
        bic = log_likelihood - 0.5 * num_params * math.log(n_samples)
        return bic


REGISTER.set_package(__name__)
REGISTER.set_class(__name__, 'auto_k', AutoKHead)
