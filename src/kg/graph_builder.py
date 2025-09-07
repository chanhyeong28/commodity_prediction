"""
Knowledge Graph Builder for Commodity Time Series Forecasting

This module implements the offline knowledge graph construction that will be
shipped to Kaggle as a SQLite database. It creates TS_PATCH nodes from time
series windows and establishes relationships between them.
"""

import sqlite3
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PatchConfig:
    """Configuration for patch generation"""
    window_sizes: List[int] = None  # [7, 14, 28] days
    strides: List[int] = None        # [1, 3, 7] days overlap
    min_patch_length: int = 5       # Minimum days for valid patch
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [7, 14, 28]
        if self.strides is None:
            self.strides = [1, 3, 7]

@dataclass
class TSNode:
    """Time series patch node"""
    node_id: str
    series_id: str
    exchange: str
    instrument: str
    window_start: int
    window_end: int
    window_size: int
    stride: int
    embedding: np.ndarray
    stats_json: str
    regime_label: str
    created_at: str

@dataclass
class TSEdge:
    """Time series relationship edge"""
    source_node: str
    target_node: str
    relation_type: str
    weight: float
    lag: int
    p_value: float
    test_stat: float
    window_size: int
    gap_days: int

class KnowledgeGraphBuilder:
    """
    Builds the knowledge graph from commodity time series data.
    
    This class handles:
    1. Parsing time series identifiers from column names
    2. Creating overlapping patches with different window sizes
    3. Computing embeddings for each patch
    4. Establishing correlation and cointegration relationships
    5. Storing everything in SQLite format for Kaggle
    """
    
    def __init__(self, db_path: str, patch_config: PatchConfig = None):
        self.db_path = db_path
        self.patch_config = patch_config or PatchConfig()
        self.exchange_prefixes = {
            'LME': 'LME_',
            'JPX': 'JPX_', 
            'US': 'US_',
            'FX': 'FX_'
        }
        
    def create_database_schema(self):
        """Create the SQLite database schema as defined in the rules"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # TS_PATCH nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ts_nodes (
                node_id TEXT PRIMARY KEY,
                series_id TEXT,
                exchange TEXT,
                instrument TEXT,
                window_start INTEGER,
                window_end INTEGER,
                window_size INTEGER,
                stride INTEGER,
                embedding BLOB,
                stats_json TEXT,
                regime_label TEXT,
                created_at TEXT
            )
        """)
        
        # TS edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ts_edges (
                source_node TEXT,
                target_node TEXT,
                relation_type TEXT,
                weight REAL,
                lag INTEGER,
                p_value REAL,
                test_stat REAL,
                window_size INTEGER,
                gap_days INTEGER
            )
        """)
        
        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT,
                attrs_json TEXT
            )
        """)
        
        # Create indexes for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts_nodes_series ON ts_nodes(series_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts_nodes_end ON ts_nodes(window_end)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON ts_edges(relation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON ts_edges(source_node)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_tgt ON ts_edges(target_node)")
        
        conn.commit()
        conn.close()
        logger.info(f"Created database schema at {self.db_path}")
    
    def parse_series_identifier(self, column_name: str) -> Tuple[str, str, str]:
        """
        Parse series identifier from column name.
        
        Examples:
        - 'LME_AL_Close' -> ('LME', 'AL', 'LME_AL_Close')
        - 'JPX_Gold_Close' -> ('JPX', 'Gold', 'JPX_Gold_Close')
        - 'US_Stock_VT_adj_close' -> ('US', 'Stock_VT', 'US_Stock_VT_adj_close')
        - 'FX_USDJPY' -> ('FX', 'USDJPY', 'FX_USDJPY')
        """
        for exchange, prefix in self.exchange_prefixes.items():
            if column_name.startswith(prefix):
                # Remove prefix and common suffixes
                instrument_part = column_name[len(prefix):]
                
                # Remove common suffixes
                suffixes_to_remove = ['_Close', '_adj_close', '_Volume', '_Open', '_High', '_Low']
                for suffix in suffixes_to_remove:
                    if instrument_part.endswith(suffix):
                        instrument_part = instrument_part[:-len(suffix)]
                        break
                
                return exchange, instrument_part, column_name
        
        # Fallback for unrecognized format
        logger.warning(f"Could not parse series identifier: {column_name}")
        return 'UNKNOWN', column_name, column_name
    
    def create_patches(self, series_data: pd.Series, series_id: str) -> List[TSNode]:
        """
        Create overlapping patches from a time series.
        
        Args:
            series_data: Time series data with date_id as index
            series_id: Identifier for the series
            
        Returns:
            List of TSNode objects representing patches
        """
        exchange, instrument, _ = self.parse_series_identifier(series_id)
        patches = []
        
        for window_size in self.patch_config.window_sizes:
            for stride in self.patch_config.strides:
                # Create overlapping windows
                for start_idx in range(0, len(series_data) - window_size + 1, stride):
                    end_idx = start_idx + window_size
                    window_data = series_data.iloc[start_idx:end_idx]
                    
                    # Skip if window is too short or has too many NaNs
                    if len(window_data) < self.patch_config.min_patch_length:
                        continue
                    
                    if window_data.isna().sum() > len(window_data) * 0.5:  # >50% NaN
                        continue
                    
                    # Create node ID
                    window_start = start_idx  # Use integer index
                    window_end = end_idx - 1  # Use integer index
                    node_id = f"{series_id}_{window_size}d_{stride}s_{window_start}_{window_end}"
                    
                    # Compute embedding (placeholder - will be replaced with actual embedding)
                    embedding = self._compute_patch_embedding(window_data)
                    
                    # Compute statistics
                    stats = self._compute_patch_stats(window_data)
                    
                    # Determine regime label
                    regime_label = self._classify_regime(window_data, stats)
                    
                    patch = TSNode(
                        node_id=node_id,
                        series_id=series_id,
                        exchange=exchange,
                        instrument=instrument,
                        window_start=window_start,
                        window_end=window_end,
                        window_size=window_size,
                        stride=stride,
                        embedding=embedding,
                        stats_json=json.dumps(stats),
                        regime_label=regime_label,
                        created_at=pd.Timestamp.now().isoformat()
                    )
                    
                    patches.append(patch)
        
        return patches
    
    def _compute_patch_embedding(self, window_data: pd.Series) -> np.ndarray:
        """
        Compute embedding for a time series patch.
        
        This is a placeholder implementation. In practice, this would use
        a trained encoder (CNN, Transformer, etc.) to create meaningful embeddings.
        """
        # Simple statistical features as embedding (will be replaced)
        features = [
            window_data.mean(),
            window_data.std(),
            window_data.skew(),
            window_data.kurtosis(),
            np.corrcoef(window_data.values[:-1], window_data.values[1:])[0, 1] if len(window_data) > 1 else 0,
            (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0] if window_data.iloc[0] != 0 else 0
        ]
        
        # Pad or truncate to fixed size
        embedding_size = 128
        if len(features) < embedding_size:
            features.extend([0.0] * (embedding_size - len(features)))
        else:
            features = features[:embedding_size]
        
        return np.array(features, dtype=np.float32)
    
    def _compute_patch_stats(self, window_data: pd.Series) -> Dict[str, float]:
        """Compute statistical features for a patch"""
        return {
            'mean': float(window_data.mean()),
            'std': float(window_data.std()),
            'skew': float(window_data.skew()),
            'kurtosis': float(window_data.kurtosis()),
            'trend_slope': float(np.polyfit(range(len(window_data)), window_data.values, 1)[0]),
            'volatility': float(window_data.std() / window_data.mean()) if window_data.mean() != 0 else 0.0,
            'min': float(window_data.min()),
            'max': float(window_data.max()),
            'range': float(window_data.max() - window_data.min())
        }
    
    def _classify_regime(self, window_data: pd.Series, stats: Dict[str, float]) -> str:
        """Classify the regime of a time series patch"""
        volatility = stats['volatility']
        trend_slope = stats['trend_slope']
        
        if volatility > 0.1:  # High volatility threshold
            return 'HIGH_VOL'
        elif trend_slope > 0.01:  # Strong upward trend
            return 'TREND_UP'
        elif trend_slope < -0.01:  # Strong downward trend
            return 'TREND_DOWN'
        else:
            return 'NORMAL'
    
    def compute_correlations(self, patches: List[TSNode], correlation_threshold: float = 0.1) -> List[TSEdge]:
        """
        Compute correlation relationships between patches.
        
        Args:
            patches: List of TSNode objects
            correlation_threshold: Minimum absolute correlation to create edge
            
        Returns:
            List of TSEdge objects representing correlations
        """
        edges = []
        
        # Group patches by series for temporal relationships
        series_patches = {}
        for patch in patches:
            if patch.series_id not in series_patches:
                series_patches[patch.series_id] = []
            series_patches[patch.series_id].append(patch)
        
        # Create temporal edges within series
        for series_id, series_patch_list in series_patches.items():
            # Sort by window_end
            series_patch_list.sort(key=lambda x: x.window_end)
            
            for i in range(len(series_patch_list) - 1):
                current_patch = series_patch_list[i]
                next_patch = series_patch_list[i + 1]
                
                gap_days = next_patch.window_start - current_patch.window_end
                
                edge = TSEdge(
                    source_node=current_patch.node_id,
                    target_node=next_patch.node_id,
                    relation_type='TEMPORAL_NEXT',
                    weight=1.0,
                    lag=0,
                    p_value=0.0,
                    test_stat=0.0,
                    window_size=0,
                    gap_days=gap_days
                )
                edges.append(edge)
        
        # Compute cross-correlations between different series
        for i, patch1 in enumerate(patches):
            for j, patch2 in enumerate(patches[i+1:], i+1):
                # Skip if same series
                if patch1.series_id == patch2.series_id:
                    continue
                
                # Compute correlation between embeddings
                correlation = np.corrcoef(patch1.embedding, patch2.embedding)[0, 1]
                
                if abs(correlation) > correlation_threshold:
                    relation_type = 'POSITIVE_CORR' if correlation > 0 else 'NEGATIVE_CORR'
                    
                    edge = TSEdge(
                        source_node=patch1.node_id,
                        target_node=patch2.node_id,
                        relation_type=relation_type,
                        weight=abs(correlation),
                        lag=0,  # Will be computed properly with actual time series
                        p_value=0.0,  # Will be computed with proper statistical test
                        test_stat=0.0,
                        window_size=min(patch1.window_size, patch2.window_size),
                        gap_days=0
                    )
                    edges.append(edge)
        
        return edges
    
    def build_from_dataframe(self, df: pd.DataFrame, target_pairs_df: pd.DataFrame = None):
        """
        Build the knowledge graph from the training dataframe.
        
        Args:
            df: Training dataframe with time series columns
            target_pairs_df: Target pairs dataframe for additional context
        """
        logger.info("Starting knowledge graph construction...")
        
        # Create database schema
        self.create_database_schema()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        all_patches = []
        
        # Process each time series column
        for column in df.columns:
            if column == 'date_id':
                continue
                
            logger.info(f"Processing series: {column}")
            series_data = df[column].dropna()
            
            if len(series_data) < self.patch_config.min_patch_length:
                logger.warning(f"Skipping {column}: insufficient data")
                continue
            
            # Create patches for this series
            patches = self.create_patches(series_data, column)
            all_patches.extend(patches)
            
            # Store patches in database
            for patch in patches:
                cursor.execute("""
                    INSERT OR REPLACE INTO ts_nodes 
                    (node_id, series_id, exchange, instrument, window_start, window_end, 
                     window_size, stride, embedding, stats_json, regime_label, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patch.node_id, patch.series_id, patch.exchange, patch.instrument,
                    patch.window_start, patch.window_end, patch.window_size, patch.stride,
                    pickle.dumps(patch.embedding), patch.stats_json, patch.regime_label, patch.created_at
                ))
        
        logger.info(f"Created {len(all_patches)} patches")
        
        # Compute relationships
        logger.info("Computing relationships...")
        edges = self.compute_correlations(all_patches)
        
        # Store edges in database
        for edge in edges:
            cursor.execute("""
                INSERT INTO ts_edges 
                (source_node, target_node, relation_type, weight, lag, p_value, 
                 test_stat, window_size, gap_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.source_node, edge.target_node, edge.relation_type, edge.weight,
                edge.lag, edge.p_value, edge.test_stat, edge.window_size, edge.gap_days
            ))
        
        logger.info(f"Created {len(edges)} edges")
        
        # Create entity nodes
        self._create_entity_nodes(cursor, df, target_pairs_df)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Knowledge graph construction complete. Database saved to {self.db_path}")
    
    def _create_entity_nodes(self, cursor, df: pd.DataFrame, target_pairs_df: pd.DataFrame = None):
        """Create entity nodes for dates, markets, instruments, etc."""
        
        # Create date entities
        unique_dates = df['date_id'].unique()
        for date_id in unique_dates:
            date_obj = pd.to_datetime(date_id)
            attrs = {
                'weekday': date_obj.weekday(),
                'month': date_obj.month,
                'quarter': date_obj.quarter,
                'year': date_obj.year,
                'is_holiday_by_exchange': False  # Would need holiday calendar
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO entities (entity_id, entity_type, attrs_json)
                VALUES (?, ?, ?)
            """, (f"DATE_{date_id}", "ENTITY_DATE", json.dumps(attrs)))
        
        # Create market entities
        for exchange in ['LME', 'JPX', 'US', 'FX']:
            attrs = {
                'timezone': self._get_exchange_timezone(exchange),
                'trading_hours': self._get_trading_hours(exchange)
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO entities (entity_id, entity_type, attrs_json)
                VALUES (?, ?, ?)
            """, (f"MARKET_{exchange}", "ENTITY_MARKET", json.dumps(attrs)))
        
        # Create instrument entities
        for column in df.columns:
            if column == 'date_id':
                continue
                
            exchange, instrument, _ = self.parse_series_identifier(column)
            attrs = {
                'sector': self._get_instrument_sector(instrument),
                'commodity_family': self._get_commodity_family(instrument),
                'currency': self._get_instrument_currency(instrument, exchange)
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO entities (entity_id, entity_type, attrs_json)
                VALUES (?, ?, ?)
            """, (f"INSTRUMENT_{instrument}", "ENTITY_INSTRUMENT", json.dumps(attrs)))
    
    def _get_exchange_timezone(self, exchange: str) -> str:
        """Get timezone for exchange"""
        timezones = {
            'LME': 'Europe/London',
            'JPX': 'Asia/Tokyo', 
            'US': 'America/New_York',
            'FX': 'UTC'
        }
        return timezones.get(exchange, 'UTC')
    
    def _get_trading_hours(self, exchange: str) -> str:
        """Get trading hours for exchange"""
        hours = {
            'LME': '08:00-17:00',
            'JPX': '09:00-15:00',
            'US': '09:30-16:00', 
            'FX': '24/7'
        }
        return hours.get(exchange, '24/7')
    
    def _get_instrument_sector(self, instrument: str) -> str:
        """Get sector for instrument"""
        if any(metal in instrument.upper() for metal in ['AL', 'CU', 'PB', 'ZN', 'NI']):
            return 'METALS'
        elif any(precious in instrument.upper() for precious in ['GOLD', 'SILVER', 'PLATINUM']):
            return 'PRECIOUS_METALS'
        elif 'STOCK' in instrument.upper():
            return 'EQUITIES'
        elif any(fx in instrument.upper() for fx in ['USD', 'JPY', 'EUR', 'GBP']):
            return 'FOREX'
        else:
            return 'OTHER'
    
    def _get_commodity_family(self, instrument: str) -> str:
        """Get commodity family for instrument"""
        return self._get_instrument_sector(instrument)  # Simplified mapping
    
    def _get_instrument_currency(self, instrument: str, exchange: str) -> str:
        """Get currency for instrument"""
        if exchange == 'LME':
            return 'USD'
        elif exchange == 'JPX':
            return 'JPY'
        elif exchange == 'US':
            return 'USD'
        elif exchange == 'FX':
            # Extract currency from FX pair
            if 'USD' in instrument:
                return 'USD'
            elif 'JPY' in instrument:
                return 'JPY'
            else:
                return 'USD'  # Default
        else:
            return 'USD'


def main():
    """Example usage of the KnowledgeGraphBuilder"""
    import pandas as pd
    
    # Load data
    train_df = pd.read_csv('/data/kaggle_projects/commodity_prediction/kaggle_data/train.csv')
    target_pairs_df = pd.read_csv('/data/kaggle_projects/commodity_prediction/kaggle_data/target_pairs.csv')
    
    # Configure patches
    patch_config = PatchConfig(
        window_sizes=[7, 14, 28],
        strides=[1, 3, 7]
    )
    
    # Build knowledge graph
    builder = KnowledgeGraphBuilder(
        db_path='/data/kaggle_projects/commodity_prediction/database/commodity_kg.db',
        patch_config=patch_config
    )
    
    builder.build_from_dataframe(train_df, target_pairs_df)
    
    print("Knowledge graph construction complete!")


if __name__ == "__main__":
    main()
