"""
GraphRAG Retrieval System for Commodity Time Series Forecasting

This module implements the retrieval system that queries the SQLite knowledge graph
to find relevant time series patches and relationships for forecasting tasks.
"""

import sqlite3
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pickle
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """Configuration for GraphRAG retrieval"""
    max_nodes: int = 256
    max_edges: int = 1024
    ts_patch_ratio: float = 0.7  # 70% TS_PATCH nodes, 30% entities
    similarity_threshold: float = 0.1
    max_hops: int = 2
    recency_decay: float = 90.0  # Days for recency boost
    market_match_weight: float = 0.2
    recency_weight: float = 0.1
    edge_strength_weight: float = 0.2
    similarity_weight: float = 0.5

@dataclass
class RetrievalResult:
    """Result of GraphRAG retrieval"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    query_embedding: np.ndarray
    retrieval_stats: Dict[str, Any]

class GraphRAGRetriever:
    """
    Retrieves relevant subgraphs from the knowledge graph for forecasting tasks.
    
    This class implements the retrieval policy defined in the rules:
    1. Seed embedding computation for the latest patch
    2. Candidate selection via kNN and time filtering
    3. Graph expansion via correlation edges
    4. Ranking and budgeting
    """
    
    def __init__(self, db_path: str, config: RetrievalConfig = None):
        self.db_path = db_path
        self.config = config or RetrievalConfig()
        self.pca = None
        self.embedding_dim = 128
        
    def _load_connection(self) -> sqlite3.Connection:
        """Load SQLite connection"""
        return sqlite3.connect(self.db_path)
    
    def compute_query_embedding(self, series_data: pd.Series, window_size: int = 7) -> np.ndarray:
        """
        Compute embedding for the query (latest patch of target series).
        
        Args:
            series_data: Recent time series data
            window_size: Size of the query window
            
        Returns:
            Embedding vector for the query
        """
        if len(series_data) < window_size:
            # Pad with last value if insufficient data
            padded_data = pd.Series([series_data.iloc[-1]] * (window_size - len(series_data)) + series_data.tolist())
        else:
            padded_data = series_data.tail(window_size)
        
        # Use the same embedding method as in graph builder
        features = [
            padded_data.mean(),
            padded_data.std(),
            padded_data.skew(),
            padded_data.kurtosis(),
            np.corrcoef(padded_data.values[:-1], padded_data.values[1:])[0, 1] if len(padded_data) > 1 else 0,
            (padded_data.iloc[-1] - padded_data.iloc[0]) / padded_data.iloc[0] if padded_data.iloc[0] != 0 else 0
        ]
        
        # Pad or truncate to fixed size
        if len(features) < self.embedding_dim:
            features.extend([0.0] * (self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]
        
        return np.array(features, dtype=np.float32)
    
    def get_candidate_nodes(self, query_embedding: np.ndarray, target_series_id: str, 
                          forecast_date: int, max_candidates: int = 1000) -> List[Dict[str, Any]]:
        """
        Get candidate nodes using indexed filters and similarity.
        
        Args:
            query_embedding: Query embedding vector
            target_series_id: Target series identifier
            forecast_date: Date for forecasting
            max_candidates: Maximum number of candidates to consider
            
        Returns:
            List of candidate node dictionaries
        """
        conn = self._load_connection()
        cursor = conn.cursor()
        
        # Parse target series info
        exchange, instrument, _ = self._parse_series_identifier(target_series_id)
        
        # Get candidates with indexed filters first
        # 1. Same exchange/instrument family (higher priority)
        cursor.execute("""
            SELECT node_id, series_id, exchange, instrument, window_start, window_end,
                   window_size, stride, embedding, stats_json, regime_label
            FROM ts_nodes 
            WHERE (exchange = ? OR instrument = ?) 
            AND window_end <= ?
            ORDER BY window_end DESC
            LIMIT ?
        """, (exchange, instrument, forecast_date, max_candidates // 2))
        
        same_market_candidates = []
        for row in cursor.fetchall():
            embedding = pickle.loads(row[8])
            same_market_candidates.append({
                'node_id': row[0],
                'series_id': row[1],
                'exchange': row[2],
                'instrument': row[3],
                'window_start': row[4],
                'window_end': row[5],
                'window_size': row[6],
                'stride': row[7],
                'embedding': embedding,
                'stats_json': row[9],
                'regime_label': row[10],
                'market_match': 1.0
            })
        
        # 2. Global candidates (different markets)
        remaining_candidates = max_candidates - len(same_market_candidates)
        cursor.execute("""
            SELECT node_id, series_id, exchange, instrument, window_start, window_end,
                   window_size, stride, embedding, stats_json, regime_label
            FROM ts_nodes 
            WHERE window_end <= ?
            ORDER BY window_end DESC
            LIMIT ?
        """, (forecast_date, remaining_candidates))
        
        global_candidates = []
        for row in cursor.fetchall():
            embedding = pickle.loads(row[8])
            global_candidates.append({
                'node_id': row[0],
                'series_id': row[1],
                'exchange': row[2],
                'instrument': row[3],
                'window_start': row[4],
                'window_end': row[5],
                'window_size': row[6],
                'stride': row[7],
                'embedding': embedding,
                'stats_json': row[9],
                'regime_label': row[10],
                'market_match': 0.0
            })
        
        conn.close()
        
        # Combine and compute similarities
        all_candidates = same_market_candidates + global_candidates
        
        # Check if we have any candidates
        if not all_candidates:
            logger.warning(f"No candidate nodes found for {target_series_id}")
            return []
        
        # Compute cosine similarities
        candidate_embeddings = np.array([c['embedding'] for c in all_candidates])
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        for i, candidate in enumerate(all_candidates):
            candidate['similarity'] = similarities[i]
        
        # Filter by similarity threshold
        filtered_candidates = [
            c for c in all_candidates 
            if c['similarity'] > self.config.similarity_threshold
        ]
        
        return filtered_candidates
    
    def expand_graph(self, seed_nodes: List[Dict[str, Any]], max_hops: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Expand the graph by following edges from seed nodes.
        
        Args:
            seed_nodes: Initial seed nodes
            max_hops: Maximum number of hops for expansion
            
        Returns:
            Tuple of (expanded_nodes, edges)
        """
        conn = self._load_connection()
        cursor = conn.cursor()
        
        # Get node IDs for querying
        seed_node_ids = [node['node_id'] for node in seed_nodes]
        
        # Get edges connected to seed nodes
        placeholders = ','.join(['?' for _ in seed_node_ids])
        cursor.execute(f"""
            SELECT source_node, target_node, relation_type, weight, lag, p_value,
                   test_stat, window_size, gap_days
            FROM ts_edges 
            WHERE source_node IN ({placeholders}) OR target_node IN ({placeholders})
            ORDER BY ABS(weight) DESC
        """, seed_node_ids + seed_node_ids)
        
        edges = []
        connected_node_ids = set(seed_node_ids)
        
        for row in cursor.fetchall():
            edge = {
                'source_node': row[0],
                'target_node': row[1],
                'relation_type': row[2],
                'weight': row[3],
                'lag': row[4],
                'p_value': row[5],
                'test_stat': row[6],
                'window_size': row[7],
                'gap_days': row[8]
            }
            edges.append(edge)
            
            # Add connected nodes to set
            connected_node_ids.add(row[0])
            connected_node_ids.add(row[1])
        
        # Get full node information for connected nodes
        if connected_node_ids:
            placeholders = ','.join(['?' for _ in connected_node_ids])
            cursor.execute(f"""
                SELECT node_id, series_id, exchange, instrument, window_start, window_end,
                       window_size, stride, embedding, stats_json, regime_label
                FROM ts_nodes 
                WHERE node_id IN ({placeholders})
            """, list(connected_node_ids))
            
            expanded_nodes = []
            for row in cursor.fetchall():
                embedding = pickle.loads(row[8])
                expanded_nodes.append({
                    'node_id': row[0],
                    'series_id': row[1],
                    'exchange': row[2],
                    'instrument': row[3],
                    'window_start': row[4],
                    'window_end': row[5],
                    'window_size': row[6],
                    'stride': row[7],
                    'embedding': embedding,
                    'stats_json': row[9],
                    'regime_label': row[10],
                    'market_match': 1.0 if row[2] == seed_nodes[0]['exchange'] else 0.0
                })
        else:
            expanded_nodes = seed_nodes
        
        conn.close()
        
        return expanded_nodes, edges
    
    def rank_and_budget(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], 
                       query_embedding: np.ndarray, target_series_id: str, 
                       forecast_date: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Rank nodes and apply budgeting constraints.
        
        Args:
            nodes: List of candidate nodes
            edges: List of edges
            query_embedding: Query embedding
            target_series_id: Target series
            forecast_date: Forecast date
            
        Returns:
            Tuple of (ranked_nodes, filtered_edges)
        """
        # Compute ranking scores
        for node in nodes:
            # Compute recency boost
            days_diff = forecast_date - node['window_end']
            recency_boost = np.exp(-days_diff / self.config.recency_decay)
            
            # Compute edge strength (sum of connected edge weights)
            connected_edges = [e for e in edges if e['source_node'] == node['node_id'] or e['target_node'] == node['node_id']]
            edge_strength = sum(abs(e['weight']) for e in connected_edges) / max(len(connected_edges), 1)
            
            # Compute final score
            score = (
                self.config.similarity_weight * node.get('similarity', 0.0) +
                self.config.market_match_weight * node.get('market_match', 0.0) +
                self.config.recency_weight * recency_boost +
                self.config.edge_strength_weight * edge_strength
            )
            
            node['ranking_score'] = score
            node['recency_boost'] = recency_boost
            node['edge_strength'] = edge_strength
        
        # Sort by ranking score
        nodes.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Apply budgeting
        max_ts_patches = int(self.config.max_nodes * self.config.ts_patch_ratio)
        max_entities = self.config.max_nodes - max_ts_patches
        
        # Separate TS_PATCH nodes and entities
        ts_patch_nodes = [n for n in nodes if 'embedding' in n]
        entity_nodes = [n for n in nodes if 'embedding' not in n]
        
        # Select top nodes
        selected_ts_patches = ts_patch_nodes[:max_ts_patches]
        selected_entities = entity_nodes[:max_entities]
        selected_nodes = selected_ts_patches + selected_entities
        
        # Filter edges to only include those between selected nodes
        selected_node_ids = {n['node_id'] for n in selected_nodes}
        filtered_edges = [
            e for e in edges 
            if e['source_node'] in selected_node_ids and e['target_node'] in selected_node_ids
        ]
        
        # Limit edges
        filtered_edges = filtered_edges[:self.config.max_edges]
        
        return selected_nodes, filtered_edges
    
    def retrieve(self, target_series_id: str, series_data: pd.Series, 
                forecast_date: int, window_size: int = 7) -> RetrievalResult:
        """
        Main retrieval method that orchestrates the GraphRAG process.
        
        Args:
            target_series_id: Target series identifier
            series_data: Recent time series data for query embedding
            forecast_date: Date for forecasting
            window_size: Size of query window
            
        Returns:
            RetrievalResult with nodes, edges, and metadata
        """
        logger.info(f"Retrieving context for {target_series_id} at date {forecast_date}")
        
        # 1. Compute query embedding
        query_embedding = self.compute_query_embedding(series_data, window_size)
        
        # 2. Get candidate nodes
        candidates = self.get_candidate_nodes(query_embedding, target_series_id, forecast_date)
        
        if not candidates:
            logger.warning(f"No candidates found for {target_series_id}")
            return RetrievalResult(
                nodes=[],
                edges=[],
                query_embedding=query_embedding,
                retrieval_stats={'error': 'No candidates found'}
            )
        
        # 3. Expand graph
        expanded_nodes, edges = self.expand_graph(candidates, self.config.max_hops)
        
        # 4. Rank and budget
        ranked_nodes, filtered_edges = self.rank_and_budget(
            expanded_nodes, edges, query_embedding, target_series_id, forecast_date
        )
        
        # 5. Prepare result
        retrieval_stats = {
            'num_candidates': len(candidates),
            'num_expanded_nodes': len(expanded_nodes),
            'num_final_nodes': len(ranked_nodes),
            'num_edges': len(filtered_edges),
            'avg_similarity': np.mean([n.get('similarity', 0) for n in ranked_nodes]),
            'avg_ranking_score': np.mean([n.get('ranking_score', 0) for n in ranked_nodes])
        }
        
        logger.info(f"Retrieved {len(ranked_nodes)} nodes and {len(filtered_edges)} edges")
        
        return RetrievalResult(
            nodes=ranked_nodes,
            edges=filtered_edges,
            query_embedding=query_embedding,
            retrieval_stats=retrieval_stats
        )
    
    def _parse_series_identifier(self, series_id: str) -> Tuple[str, str, str]:
        """Parse series identifier (same as in graph builder)"""
        exchange_prefixes = {
            'LME': 'LME_',
            'JPX': 'JPX_', 
            'US': 'US_',
            'FX': 'FX_'
        }
        
        for exchange, prefix in exchange_prefixes.items():
            if series_id.startswith(prefix):
                instrument_part = series_id[len(prefix):]
                
                # Remove common suffixes
                suffixes_to_remove = ['_Close', '_adj_close', '_Volume', '_Open', '_High', '_Low']
                for suffix in suffixes_to_remove:
                    if instrument_part.endswith(suffix):
                        instrument_part = instrument_part[:-len(suffix)]
                        break
                
                return exchange, instrument_part, series_id
        
        return 'UNKNOWN', series_id, series_id
    
    def prepare_context_for_model(self, result: RetrievalResult, 
                                 max_context_tokens: int = 256) -> Dict[str, Any]:
        """
        Prepare the retrieval result for the Time-LlaMA model.
        
        Args:
            result: RetrievalResult from retrieve()
            max_context_tokens: Maximum number of context tokens
            
        Returns:
            Dictionary with context ready for the model
        """
        # Down-project embeddings if needed
        if self.pca is None and result.nodes:
            embeddings = np.array([n['embedding'] for n in result.nodes])
            if embeddings.shape[1] > 64:  # Down-project to 64 dims
                self.pca = PCA(n_components=64)
                self.pca.fit(embeddings)
        
        # Prepare context
        context = {
            'nodes': [],
            'edges': result.edges[:max_context_tokens // 2],  # Limit edges
            'query_embedding': result.query_embedding,
            'stats': result.retrieval_stats
        }
        
        # Add nodes with down-projected embeddings
        for i, node in enumerate(result.nodes[:max_context_tokens // 2]):
            node_context = {
                'node_id': node['node_id'],
                'series_id': node['series_id'],
                'exchange': node['exchange'],
                'instrument': node['instrument'],
                'window_end': node['window_end'],
                'window_size': node['window_size'],
                'regime_label': node['regime_label'],
                'ranking_score': node.get('ranking_score', 0.0),
                'similarity': node.get('similarity', 0.0)
            }
            
            # Add down-projected embedding
            if self.pca is not None:
                node_context['embedding'] = self.pca.transform([node['embedding']])[0].tolist()
            else:
                node_context['embedding'] = node['embedding'].tolist()
            
            context['nodes'].append(node_context)
        
        return context


def main():
    """Example usage of the GraphRAGRetriever"""
    import pandas as pd
    
    # Load data
    train_df = pd.read_csv('/data/kaggle_projects/commodity_prediction/kaggle_data/train.csv')
    
    # Initialize retriever
    config = RetrievalConfig(
        max_nodes=128,
        max_edges=256,
        similarity_threshold=0.1
    )
    
    retriever = GraphRAGRetriever(
        db_path='/data/kaggle_projects/commodity_prediction/database/commodity_kg.db',
        config=config
    )
    
    # Example retrieval
    target_series = 'LME_AL_Close'
    series_data = train_df[target_series].dropna().tail(30)  # Last 30 days
    forecast_date = train_df['date_id'].max() + 1
    
    result = retriever.retrieve(target_series, series_data, forecast_date)
    
    print(f"Retrieved {len(result.nodes)} nodes and {len(result.edges)} edges")
    print(f"Retrieval stats: {result.retrieval_stats}")
    
    # Prepare context for model
    context = retriever.prepare_context_for_model(result)
    print(f"Context prepared with {len(context['nodes'])} nodes")


if __name__ == "__main__":
    main()
