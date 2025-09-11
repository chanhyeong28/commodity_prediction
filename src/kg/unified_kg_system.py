"""
Unified Knowledge Graph System

This module consolidates all KG functionality into a single, production-ready system
that combines graph construction, retrieval, and optimization for the commodity
forecasting pipeline.
"""

import sqlite3
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import pickle
import logging
import time
from tqdm import tqdm
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class KGConfig:
    """Unified configuration for the knowledge graph system"""
    # Construction settings
    window_sizes: List[int] = None
    strides: List[int] = None
    max_nodes_per_series: int = 100
    embedding_dim: int = 64
    
    # Correlation settings
    correlation_threshold: float = 0.3
    p_value_threshold: float = 0.05
    max_correlations_per_node: int = 50
    
    # Retrieval settings
    max_retrieval_nodes: int = 50
    max_retrieval_edges: int = 100
    similarity_threshold: float = 0.2
    cache_size: int = 1000
    
    # Performance settings
    batch_size: int = 100
    db_path: str = "database/commodity_kg.db"
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [7, 14]
        if self.strides is None:
            self.strides = [7]


@dataclass
class RetrievalResult:
    """Result of KG retrieval"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    query_embedding: np.ndarray
    retrieval_stats: Dict[str, Any]


class UnifiedKGSystem:
    """
    Unified Knowledge Graph System that combines construction, retrieval, and optimization.
    
    This class provides a single interface for all KG operations, optimized for
    Kaggle commodity forecasting constraints.
    """
    
    def __init__(self, config: KGConfig = None):
        self.config = config or KGConfig()
        self.db_path = Path(self.config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Cache for fast retrieval
        self.cache = {}
        self._init_cache()
        
        # Statistics
        self.stats = {
            'nodes_created': 0,
            'edges_created': 0,
            'retrievals_performed': 0,
            'total_retrieval_time': 0
        }
    
    def _init_database(self):
        """Initialize the SQLite database with proper schema and indexes"""
        with sqlite3.connect(self.db_path) as conn:
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ts_nodes (
                    node_id TEXT PRIMARY KEY,
                    series_id TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    window_start INTEGER NOT NULL,
                    window_end INTEGER NOT NULL,
                    window_size INTEGER NOT NULL,
                    stride INTEGER NOT NULL,
                    embedding BLOB,
                    stats_json TEXT,
                    regime_label TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ts_edges (
                    source_node TEXT NOT NULL,
                    target_node TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL,
                    lag INTEGER,
                    p_value REAL,
                    test_stat REAL,
                    window_size INTEGER,
                    gap_days INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_node, target_node, relation_type)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    attrs_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create performance indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_nodes_series ON ts_nodes(series_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_nodes_end ON ts_nodes(window_end)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_nodes_exchange ON ts_nodes(exchange)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON ts_edges(relation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON ts_edges(source_node)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_tgt ON ts_edges(target_node)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
            
            conn.commit()
    
    def _init_cache(self):
        """Initialize retrieval cache with recent nodes"""
        if not self.db_path.exists():
            return
        
        logger.info("Initializing KG retrieval cache...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Cache recent nodes for fast access
            recent_nodes = conn.execute("""
                SELECT node_id, series_id, exchange, instrument, window_end, embedding
                FROM ts_nodes 
                WHERE window_end > (SELECT MAX(window_end) - 100 FROM ts_nodes)
                ORDER BY window_end DESC
                LIMIT ?
            """, (self.config.cache_size,)).fetchall()
            
            for node_id, series_id, exchange, instrument, window_end, embedding_blob in recent_nodes:
                embedding = pickle.loads(embedding_blob)
                self.cache[node_id] = {
                    'series_id': series_id,
                    'exchange': exchange,
                    'instrument': instrument,
                    'window_end': window_end,
                    'embedding': embedding
                }
        
        logger.info(f"Cached {len(self.cache)} recent nodes")
    
    def build_from_dataframe(self, data: pd.DataFrame, target_pairs: pd.DataFrame):
        """Build knowledge graph from training data"""
        start_time = time.time()
        logger.info(f"Building unified KG with {len(data)} rows and {len(data.columns)-1} series")
        
        # Parse series information
        series_info = self._parse_series_info(data.columns[1:])
        logger.info(f"Parsed {len(series_info)} series from {len(set(s['exchange'] for s in series_info))} exchanges")
        
        # Create patches
        self._create_patches(data, series_info)
        
        # Compute correlations
        self._compute_correlations()
        
        # Create entities
        self._create_entities(series_info, data)
        
        # Create temporal edges
        self._create_temporal_edges()
        
        # Finalize
        self._finalize_construction()
        
        construction_time = time.time() - start_time
        logger.info(f"KG construction completed in {construction_time:.2f} seconds")
        
        # Reinitialize cache
        self._init_cache()
    
    def retrieve_context(self, target_series_id: str, series_data: pd.Series, 
                        forecast_date: int) -> RetrievalResult:
        """Retrieve relevant context for forecasting"""
        start_time = time.time()
        
        # Compute query embedding
        query_embedding = self._compute_query_embedding(series_data)
        
        # Get candidates from cache first
        candidates = self._get_cached_candidates(query_embedding, target_series_id, forecast_date)
        
        # If not enough candidates, query database
        if len(candidates) < self.config.max_retrieval_nodes:
            db_candidates = self._get_db_candidates(query_embedding, target_series_id, forecast_date)
            candidates.extend(db_candidates)
        
        # Limit candidates
        candidates = candidates[:self.config.max_retrieval_nodes]
        
        # Get edges for candidates
        edges = self._get_candidate_edges(candidates)
        
        # Create result
        retrieval_time = time.time() - start_time
        result = RetrievalResult(
            nodes=candidates,
            edges=edges,
            query_embedding=query_embedding,
            retrieval_stats={
                'retrieval_time': retrieval_time,
                'candidates_found': len(candidates),
                'edges_found': len(edges),
                'cache_hit_rate': len([c for c in candidates if c['node_id'] in self.cache]) / len(candidates) if candidates else 0
            }
        )
        
        # Update statistics
        self.stats['retrievals_performed'] += 1
        self.stats['total_retrieval_time'] += retrieval_time
        
        logger.info(f"Retrieved {len(candidates)} nodes, {len(edges)} edges in {retrieval_time:.3f}s")
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        with sqlite3.connect(self.db_path) as conn:
            node_count = conn.execute("SELECT COUNT(*) FROM ts_nodes").fetchone()[0]
            edge_count = conn.execute("SELECT COUNT(*) FROM ts_edges").fetchone()[0]
            entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        
        avg_retrieval_time = (self.stats['total_retrieval_time'] / self.stats['retrievals_performed'] 
                            if self.stats['retrievals_performed'] > 0 else 0)
        
        return {
            'nodes': node_count,
            'edges': edge_count,
            'entities': entity_count,
            'cache_size': len(self.cache),
            'retrievals_performed': self.stats['retrievals_performed'],
            'avg_retrieval_time': avg_retrieval_time,
            'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }
    
    # Private methods for construction
    def _parse_series_info(self, columns: List[str]) -> List[Dict[str, str]]:
        """Parse series information from column names"""
        series_info = []
        
        for col in columns:
            if col.startswith('LME_'):
                exchange = 'LME'
                instrument = col.replace('LME_', '').split('_')[0]
            elif col.startswith('JPX_'):
                exchange = 'JPX'
                instrument = col.replace('JPX_', '').split('_')[0]
            elif col.startswith('US_Stock_'):
                exchange = 'US'
                instrument = col.replace('US_Stock_', '').split('_')[0]
            elif col.startswith('FX_'):
                exchange = 'FX'
                instrument = col.replace('FX_', '')
            else:
                continue
            
            series_info.append({
                'series_id': col,
                'exchange': exchange,
                'instrument': instrument,
                'field': col.split('_')[-1] if '_' in col else 'price'
            })
        
        return series_info
    
    def _create_patches(self, data: pd.DataFrame, series_info: List[Dict[str, str]]):
        """Create time series patches"""
        logger.info("Creating time series patches...")
        
        date_col = data.columns[0]
        series_data = data.set_index(date_col)
        
        for series in tqdm(series_info, desc="Creating patches"):
            series_id = series['series_id']
            if series_id not in series_data.columns:
                continue
            
            values = series_data[series_id].dropna()
            if len(values) < max(self.config.window_sizes):
                continue
            
            for window_size in self.config.window_sizes:
                for stride in self.config.strides:
                    self._create_patches_for_series(series_id, values, window_size, stride, series)
    
    def _create_patches_for_series(self, series_id: str, values: pd.Series, 
                                 window_size: int, stride: int, series_info: Dict[str, str]):
        """Create patches for a specific series"""
        max_patches = min(self.config.max_nodes_per_series, len(values) // stride)
        
        for i in range(0, len(values) - window_size + 1, stride):
            if i // stride >= max_patches:
                break
            
            window_data = values.iloc[i:i + window_size]
            if len(window_data) < window_size:
                continue
            
            node_id = f"{series_id}_{window_size}d_{stride}s_{i}_{i+window_size-1}"
            embedding = self._compute_patch_embedding(window_data)
            stats_dict = self._compute_patch_statistics(window_data)
            regime_label = self._classify_regime(stats_dict)
            
            self._store_patch_node(
                node_id, series_id, series_info, i, i + window_size - 1,
                window_size, stride, embedding, stats_dict, regime_label
            )
            
            self.stats['nodes_created'] += 1
    
    def _compute_patch_embedding(self, window_data: pd.Series) -> np.ndarray:
        """Compute embedding for a time series patch"""
        features = [
            window_data.mean(),
            window_data.std(),
            window_data.skew(),
            window_data.kurtosis(),
            np.corrcoef(window_data[:-1], window_data[1:])[0, 1] if len(window_data) > 1 else 0,
            (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0] if window_data.iloc[0] != 0 else 0
        ]
        
        embedding = np.zeros(self.config.embedding_dim)
        embedding[:len(features)] = features
        
        return embedding.astype(np.float32)
    
    def _compute_patch_statistics(self, window_data: pd.Series) -> Dict[str, float]:
        """Compute statistical features for a patch"""
        return {
            'mean': float(window_data.mean()),
            'std': float(window_data.std()),
            'skew': float(window_data.skew()),
            'kurtosis': float(window_data.kurtosis()),
            'trend_slope': float(np.polyfit(range(len(window_data)), window_data, 1)[0]),
            'volatility': float(window_data.std() / window_data.mean()) if window_data.mean() != 0 else 0,
            'min': float(window_data.min()),
            'max': float(window_data.max()),
            'range': float(window_data.max() - window_data.min())
        }
    
    def _classify_regime(self, stats: Dict[str, float]) -> str:
        """Classify market regime based on statistics"""
        if stats['volatility'] > 0.1:
            return 'HIGH_VOL'
        elif stats['trend_slope'] > 0.01:
            return 'TREND_UP'
        elif stats['trend_slope'] < -0.01:
            return 'TREND_DOWN'
        else:
            return 'NORMAL'
    
    def _store_patch_node(self, node_id: str, series_id: str, series_info: Dict[str, str],
                         window_start: int, window_end: int, window_size: int, stride: int,
                         embedding: np.ndarray, stats: Dict[str, float], regime: str):
        """Store a patch node in the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ts_nodes 
                (node_id, series_id, exchange, instrument, window_start, window_end, 
                 window_size, stride, embedding, stats_json, regime_label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id, series_id, series_info['exchange'], series_info['instrument'],
                window_start, window_end, window_size, stride,
                pickle.dumps(embedding), str(stats), regime
            ))
            conn.commit()
    
    def _compute_correlations(self):
        """Compute correlations between patches"""
        logger.info("Computing correlations...")
        
        with sqlite3.connect(self.db_path) as conn:
            nodes_by_exchange = {}
            for row in conn.execute("SELECT node_id, series_id, exchange, embedding FROM ts_nodes"):
                exchange = row[2]
                if exchange not in nodes_by_exchange:
                    nodes_by_exchange[exchange] = []
                nodes_by_exchange[exchange].append(row)
            
            for exchange, nodes in tqdm(nodes_by_exchange.items(), desc="Computing correlations"):
                if len(nodes) > 1:
                    self._compute_correlations_for_exchange(nodes)
    
    def _compute_correlations_for_exchange(self, nodes: List[Tuple]):
        """Compute correlations within an exchange"""
        if len(nodes) < 2:
            return
        
        batch_size = min(self.config.batch_size, len(nodes))
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            self._compute_correlation_batch(batch)
    
    def _compute_correlation_batch(self, nodes: List[Tuple]):
        """Compute correlations for a batch of nodes"""
        embeddings = []
        node_ids = []
        
        for node_id, series_id, exchange, embedding_blob in nodes:
            try:
                embedding = pickle.loads(embedding_blob)
                embeddings.append(embedding)
                node_ids.append(node_id)
            except:
                continue
        
        if len(embeddings) < 2:
            return
        
        embeddings = np.array(embeddings)
        corr_matrix = np.corrcoef(embeddings)
        
        for i in range(len(node_ids)):
            correlations = []
            for j in range(len(node_ids)):
                if i != j:
                    corr = corr_matrix[i, j]
                    if abs(corr) >= self.config.correlation_threshold:
                        correlations.append((j, abs(corr), corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_correlations = correlations[:self.config.max_correlations_per_node]
            
            for j, abs_corr, corr in top_correlations:
                p_value = self._compute_correlation_p_value(corr, len(embeddings[0]))
                
                if p_value <= self.config.p_value_threshold:
                    relation_type = 'POSITIVE_CORR' if corr > 0 else 'NEGATIVE_CORR'
                    self._store_edge(
                        node_ids[i], node_ids[j], relation_type,
                        weight=abs_corr, p_value=p_value
                    )
                    self.stats['edges_created'] += 1
    
    def _compute_correlation_p_value(self, corr: float, n: int) -> float:
        """Compute p-value for correlation coefficient"""
        if n <= 2 or abs(corr) >= 0.999:
            return 1.0
        
        try:
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            return p_value
        except:
            return 1.0
    
    def _create_entities(self, series_info: List[Dict[str, str]], data: pd.DataFrame):
        """Create entity nodes"""
        logger.info("Creating entity nodes...")
        
        with sqlite3.connect(self.db_path) as conn:
            exchanges = set(s['exchange'] for s in series_info)
            for exchange in exchanges:
                entity_id = f"MARKET_{exchange}"
                attrs = {
                    'market_id': exchange,
                    'timezone': self._get_exchange_timezone(exchange)
                }
                conn.execute("""
                    INSERT OR REPLACE INTO entities (entity_id, entity_type, attrs_json)
                    VALUES (?, ?, ?)
                """, (entity_id, 'MARKET', str(attrs)))
            
            instruments = set((s['exchange'], s['instrument']) for s in series_info)
            for exchange, instrument in instruments:
                entity_id = f"INSTRUMENT_{exchange}_{instrument}"
                attrs = {
                    'instrument_id': instrument,
                    'exchange': exchange,
                    'sector': self._get_instrument_sector(instrument),
                    'commodity_family': self._get_commodity_family(instrument)
                }
                conn.execute("""
                    INSERT OR REPLACE INTO entities (entity_id, entity_type, attrs_json)
                    VALUES (?, ?, ?)
                """, (entity_id, 'INSTRUMENT', str(attrs)))
            
            conn.commit()
    
    def _create_temporal_edges(self):
        """Create temporal edges between consecutive patches"""
        logger.info("Creating temporal edges...")
        
        with sqlite3.connect(self.db_path) as conn:
            series_nodes = {}
            for row in conn.execute("""
                SELECT node_id, series_id, window_start, window_end 
                FROM ts_nodes ORDER BY series_id, window_start
            """):
                series_id = row[1]
                if series_id not in series_nodes:
                    series_nodes[series_id] = []
                series_nodes[series_id].append(row)
            
            for series_id, nodes in tqdm(series_nodes.items(), desc="Creating temporal edges"):
                for i in range(len(nodes) - 1):
                    current_node = nodes[i]
                    next_node = nodes[i + 1]
                    
                    gap_days = next_node[2] - current_node[3] - 1
                    if gap_days >= 0:
                        self._store_edge(
                            current_node[0], next_node[0], 'TEMPORAL_NEXT',
                            gap_days=gap_days
                        )
    
    def _store_edge(self, source: str, target: str, relation_type: str, 
                   weight: float = None, lag: int = None, p_value: float = None,
                   test_stat: float = None, window_size: int = None, gap_days: int = None):
        """Store an edge in the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ts_edges 
                (source_node, target_node, relation_type, weight, lag, p_value, 
                 test_stat, window_size, gap_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (source, target, relation_type, weight, lag, p_value, 
                  test_stat, window_size, gap_days))
            conn.commit()
    
    def _finalize_construction(self):
        """Finalize KG construction with optimizations"""
        logger.info("Finalizing KG construction...")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("ANALYZE")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_nodes_regime ON ts_nodes(regime_label)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_edges_weight ON ts_edges(weight)")
            conn.commit()
    
    # Private methods for retrieval
    def _compute_query_embedding(self, series_data: pd.Series) -> np.ndarray:
        """Compute embedding for query"""
        if len(series_data) < 7:
            padded_data = pd.Series([series_data.iloc[-1]] * (7 - len(series_data)) + series_data.tolist())
        else:
            padded_data = series_data.tail(7)
        
        features = [
            padded_data.mean(),
            padded_data.std(),
            padded_data.skew(),
            padded_data.kurtosis(),
            np.corrcoef(padded_data.values[:-1], padded_data.values[1:])[0, 1] if len(padded_data) > 1 else 0,
            (padded_data.iloc[-1] - padded_data.iloc[0]) / padded_data.iloc[0] if padded_data.iloc[0] != 0 else 0
        ]
        
        embedding = np.zeros(self.config.embedding_dim)
        embedding[:len(features)] = features
        
        return embedding.astype(np.float32)
    
    def _get_cached_candidates(self, query_embedding: np.ndarray, target_series_id: str, 
                              forecast_date: int) -> List[Dict[str, Any]]:
        """Get candidates from cache"""
        if not self.cache:
            return []
        
        exchange, instrument, _ = self._parse_series_identifier(target_series_id)
        
        cached_embeddings = []
        cached_nodes = []
        
        for node_id, node_data in self.cache.items():
            if node_data['window_end'] <= forecast_date:
                cached_embeddings.append(node_data['embedding'])
                cached_nodes.append({
                    'node_id': node_id,
                    'series_id': node_data['series_id'],
                    'exchange': node_data['exchange'],
                    'instrument': node_data['instrument'],
                    'window_end': node_data['window_end'],
                    'type': 'TS_PATCH'
                })
        
        if not cached_embeddings:
            return []
        
        similarities = cosine_similarity([query_embedding], cached_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        
        candidates = []
        for idx in sorted_indices:
            if similarities[idx] >= self.config.similarity_threshold:
                candidates.append(cached_nodes[idx])
                if len(candidates) >= self.config.max_retrieval_nodes:
                    break
        
        return candidates
    
    def _get_db_candidates(self, query_embedding: np.ndarray, target_series_id: str, 
                          forecast_date: int) -> List[Dict[str, Any]]:
        """Get candidates from database"""
        exchange, instrument, _ = self._parse_series_identifier(target_series_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, series_id, exchange, instrument, window_start, window_end,
                       window_size, stride, embedding, stats_json, regime_label
                FROM ts_nodes 
                WHERE (exchange = ? OR instrument = ?) 
                AND window_end <= ?
                ORDER BY window_end DESC
                LIMIT ?
            """, (exchange, instrument, forecast_date, self.config.batch_size))
            
            candidates = []
            embeddings = []
            
            for row in cursor.fetchall():
                embedding = pickle.loads(row[8])
                candidates.append({
                    'node_id': row[0],
                    'series_id': row[1],
                    'exchange': row[2],
                    'instrument': row[3],
                    'window_start': row[4],
                    'window_end': row[5],
                    'window_size': row[6],
                    'stride': row[7],
                    'stats_json': row[9],
                    'regime_label': row[10],
                    'type': 'TS_PATCH'
                })
                embeddings.append(embedding)
            
            if not embeddings:
                return []
            
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            
            filtered_candidates = []
            for idx in sorted_indices:
                if similarities[idx] >= self.config.similarity_threshold:
                    filtered_candidates.append(candidates[idx])
                    if len(filtered_candidates) >= self.config.max_retrieval_nodes:
                        break
            
            return filtered_candidates
    
    def _get_candidate_edges(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get edges for candidates"""
        if not candidates:
            return []
        
        node_ids = [c['node_id'] for c in candidates]
        
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join(['?' for _ in node_ids])
            cursor = conn.execute(f"""
                SELECT source_node, target_node, relation_type, weight, p_value
                FROM ts_edges 
                WHERE source_node IN ({placeholders}) AND target_node IN ({placeholders})
                LIMIT ?
            """, node_ids + node_ids + [self.config.max_retrieval_edges])
            
            edges = []
            for row in cursor.fetchall():
                edges.append({
                    'source': row[0],
                    'target': row[1],
                    'relation_type': row[2],
                    'weight': row[3],
                    'p_value': row[4]
                })
            
            return edges
    
    def _parse_series_identifier(self, series_id: str) -> Tuple[str, str, str]:
        """Parse series identifier into components"""
        if series_id.startswith('LME_'):
            exchange = 'LME'
            instrument = series_id.replace('LME_', '').split('_')[0]
            field = series_id.split('_')[-1]
        elif series_id.startswith('JPX_'):
            exchange = 'JPX'
            instrument = series_id.replace('JPX_', '').split('_')[0]
            field = series_id.split('_')[-1]
        elif series_id.startswith('US_Stock_'):
            exchange = 'US'
            instrument = series_id.replace('US_Stock_', '').split('_')[0]
            field = series_id.split('_')[-1]
        elif series_id.startswith('FX_'):
            exchange = 'FX'
            instrument = series_id.replace('FX_', '')
            field = 'rate'
        else:
            exchange = 'UNKNOWN'
            instrument = series_id
            field = 'unknown'
        
        return exchange, instrument, field
    
    # Utility methods
    def _get_exchange_timezone(self, exchange: str) -> str:
        """Get timezone for exchange"""
        timezones = {
            'LME': 'Europe/London',
            'JPX': 'Asia/Tokyo',
            'US': 'America/New_York',
            'FX': 'UTC'
        }
        return timezones.get(exchange, 'UTC')
    
    def _get_instrument_sector(self, instrument: str) -> str:
        """Get sector for instrument"""
        if any(metal in instrument.upper() for metal in ['GOLD', 'SILVER', 'PLATINUM']):
            return 'PRECIOUS_METALS'
        elif any(metal in instrument.upper() for metal in ['COPPER', 'ALUMINUM', 'LEAD', 'ZINC']):
            return 'BASE_METALS'
        elif 'OIL' in instrument.upper() or 'CRUDE' in instrument.upper():
            return 'ENERGY'
        else:
            return 'OTHER'
    
    def _get_commodity_family(self, instrument: str) -> str:
        """Get commodity family for instrument"""
        return instrument.upper()
