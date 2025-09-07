#!/usr/bin/env python3
"""
Test NetworkX and SQLite Database Components

This script tests the knowledge graph construction and GraphRAG retrieval
with NetworkX and SQLite to identify and fix any issues.
"""

import sys
import pandas as pd
import numpy as np
import sqlite3
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_sqlite_basic():
    """Test basic SQLite functionality"""
    print("🗄️  Testing SQLite basic functionality...")
    
    try:
        # Create test database
        test_db = "test_kg.db"
        if Path(test_db).exists():
            Path(test_db).unlink()
        
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        # Test table creation
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
        
        # Test data insertion
        test_data = {
            'node_id': 'test_node_1',
            'series_id': 'LME_AL_Close',
            'exchange': 'LME',
            'instrument': 'AL',
            'window_start': 1,
            'window_end': 7,
            'window_size': 7,
            'stride': 1,
            'embedding': np.random.randn(128).astype(np.float32).tobytes(),
            'stats_json': '{"mean": 100.0, "std": 10.0}',
            'regime_label': 'NORMAL',
            'created_at': '2023-01-01'
        }
        
        cursor.execute("""
            INSERT INTO ts_nodes 
            (node_id, series_id, exchange, instrument, window_start, window_end, 
             window_size, stride, embedding, stats_json, regime_label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(test_data.values()))
        
        # Test data retrieval
        cursor.execute("SELECT * FROM ts_nodes WHERE node_id = ?", ('test_node_1',))
        result = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        # Cleanup
        Path(test_db).unlink()
        
        print("✅ SQLite basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ SQLite test failed: {e}")
        return False

def test_networkx_basic():
    """Test basic NetworkX functionality"""
    print("\n🕸️  Testing NetworkX basic functionality...")
    
    try:
        # Create test graph
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node('node1', type='TS_PATCH', series_id='LME_AL_Close')
        G.add_node('node2', type='TS_PATCH', series_id='JPX_Gold_Close')
        G.add_node('node3', type='ENTITY_MARKET', market_id='LME')
        
        # Add edges
        G.add_edge('node1', 'node2', relation='POSITIVE_CORR', weight=0.8)
        G.add_edge('node1', 'node3', relation='BELONGS_TO')
        
        # Test graph operations
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert 'node1' in G.nodes()
        assert ('node1', 'node2') in G.edges()
        
        # Test graph traversal
        neighbors = list(G.neighbors('node1'))
        assert len(neighbors) == 2
        
        # Test centrality
        centrality = nx.degree_centrality(G)
        assert len(centrality) == 3
        
        print("✅ NetworkX basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ NetworkX test failed: {e}")
        return False

def test_kg_builder_small():
    """Test knowledge graph builder with small dataset"""
    print("\n🏗️  Testing Knowledge Graph Builder with small dataset...")
    
    try:
        from src.kg.graph_builder import KnowledgeGraphBuilder, PatchConfig
        
        # Create small test data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        test_data = pd.DataFrame({
            'date_id': dates,
            'LME_AL_Close': np.random.randn(30).cumsum() + 100,
            'JPX_Gold_Close': np.random.randn(30).cumsum() + 2000
        })
        
        target_pairs = pd.DataFrame({
            'target': ['LME_AL_Close - JPX_Gold_Close'],
            'pair': ['LME_AL_Close - JPX_Gold_Close'],
            'lag': [1]
        })
        
        # Configure patches
        patch_config = PatchConfig(
            window_sizes=[7],
            strides=[1]
        )
        
        # Build knowledge graph
        test_db = "test_kg_small.db"
        if Path(test_db).exists():
            Path(test_db).unlink()
        
        builder = KnowledgeGraphBuilder(
            db_path=test_db,
            patch_config=patch_config
        )
        
        builder.build_from_dataframe(test_data, target_pairs)
        
        # Verify database was created
        assert Path(test_db).exists()
        
        # Test database contents
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        # Check ts_nodes table
        cursor.execute("SELECT COUNT(*) FROM ts_nodes")
        node_count = cursor.fetchone()[0]
        print(f"   - Created {node_count} TS_PATCH nodes")
        
        # Check edges table
        cursor.execute("SELECT COUNT(*) FROM ts_edges")
        edge_count = cursor.fetchone()[0]
        print(f"   - Created {edge_count} edges")
        
        conn.close()
        
        # Cleanup
        Path(test_db).unlink()
        
        print("✅ Knowledge Graph Builder works with small dataset")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge Graph Builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_rag_retrieval():
    """Test GraphRAG retrieval system"""
    print("\n🔍 Testing GraphRAG Retrieval System...")
    
    try:
        from src.kg.graph_rag import GraphRAGRetriever, RetrievalConfig
        
        # First create a small knowledge graph
        from src.kg.graph_builder import KnowledgeGraphBuilder, PatchConfig
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'date_id': dates,
            'LME_AL_Close': np.random.randn(50).cumsum() + 100,
            'JPX_Gold_Close': np.random.randn(50).cumsum() + 2000
        })
        
        target_pairs = pd.DataFrame({
            'target': ['LME_AL_Close'],
            'pair': ['LME_AL_Close'],
            'lag': [1]
        })
        
        # Build small KG
        test_db = "test_kg_rag.db"
        if Path(test_db).exists():
            Path(test_db).unlink()
        
        patch_config = PatchConfig(window_sizes=[7], strides=[1])
        builder = KnowledgeGraphBuilder(db_path=test_db, patch_config=patch_config)
        builder.build_from_dataframe(test_data, target_pairs)
        
        # Test GraphRAG retrieval
        retrieval_config = RetrievalConfig(
            max_nodes=10,
            max_edges=20,
            similarity_threshold=0.1
        )
        
        retriever = GraphRAGRetriever(
            db_path=test_db,
            config=retrieval_config
        )
        
        # Test query embedding computation
        series_data = pd.Series(np.random.randn(20).cumsum() + 100)
        query_embedding = retriever.compute_query_embedding(series_data, window_size=7)
        
        assert query_embedding.shape[0] == 128  # embedding_dim
        print(f"   - Query embedding shape: {query_embedding.shape}")
        
        # Test retrieval
        kg_result = retriever.retrieve('LME_AL_Close', series_data, 30)
        
        print(f"   - Retrieved {len(kg_result.nodes)} nodes")
        print(f"   - Retrieved {len(kg_result.edges)} edges")
        
        # Cleanup
        Path(test_db).unlink()
        
        print("✅ GraphRAG Retrieval System works")
        return True
        
    except Exception as e:
        print(f"❌ GraphRAG Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_networkx_integration():
    """Test NetworkX integration with SQLite data"""
    print("\n🔗 Testing NetworkX integration with SQLite...")
    
    try:
        # Create test database with some data
        test_db = "test_networkx_integration.db"
        if Path(test_db).exists():
            Path(test_db).unlink()
        
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        # Create schema
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
        
        # Insert test data
        nodes_data = [
            ('node1', 'LME_AL_Close', 'LME', 'AL', 1, 7, 7, 1, 
             np.random.randn(128).astype(np.float32).tobytes(), '{"mean": 100.0}', 'NORMAL', '2023-01-01'),
            ('node2', 'JPX_Gold_Close', 'JPX', 'Gold', 1, 7, 7, 1,
             np.random.randn(128).astype(np.float32).tobytes(), '{"mean": 2000.0}', 'NORMAL', '2023-01-01'),
            ('node3', 'LME_AL_Close', 'LME', 'AL', 8, 14, 7, 1,
             np.random.randn(128).astype(np.float32).tobytes(), '{"mean": 105.0}', 'NORMAL', '2023-01-02')
        ]
        
        for node_data in nodes_data:
            cursor.execute("""
                INSERT INTO ts_nodes 
                (node_id, series_id, exchange, instrument, window_start, window_end, 
                 window_size, stride, embedding, stats_json, regime_label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, node_data)
        
        # Insert edges
        edges_data = [
            ('node1', 'node2', 'POSITIVE_CORR', 0.8, 0, 0.01, 0.0, 7, 0),
            ('node1', 'node3', 'TEMPORAL_NEXT', 0.0, 0, 0.0, 0.0, 7, 1),
            ('node2', 'node3', 'NEGATIVE_CORR', -0.6, 0, 0.05, 0.0, 7, 0)
        ]
        
        for edge_data in edges_data:
            cursor.execute("""
                INSERT INTO ts_edges (source_node, target_node, relation_type, weight, lag, p_value, test_stat, window_size, gap_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, edge_data)
        
        conn.commit()
        conn.close()
        
        # Load into NetworkX
        G = nx.DiGraph()
        
        # Load nodes
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ts_nodes")
        nodes = cursor.fetchall()
        
        for node in nodes:
            node_id, series_id, exchange, instrument, window_start, window_end, window_size, stride, embedding, stats_json, regime_label, created_at = node
            G.add_node(node_id, 
                      series_id=series_id,
                      exchange=exchange,
                      instrument=instrument,
                      window_start=window_start,
                      window_end=window_end,
                      window_size=window_size,
                      stride=stride,
                      regime_label=regime_label,
                      created_at=created_at)
        
        # Load edges
        cursor.execute("SELECT * FROM ts_edges")
        edges = cursor.fetchall()
        
        for edge in edges:
            source, target, relation_type, weight, lag, p_value, test_stat, window_size, gap_days = edge
            G.add_edge(source, target, relation_type=relation_type, weight=weight, lag=lag, p_value=p_value, test_stat=test_stat, window_size=window_size, gap_days=gap_days)
        
        conn.close()
        
        # Test NetworkX operations
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3
        
        # Test graph traversal
        neighbors = list(G.neighbors('node1'))
        assert len(neighbors) == 2
        
        # Test centrality
        centrality = nx.degree_centrality(G)
        assert len(centrality) == 3
        
        # Test subgraph extraction
        subgraph = G.subgraph(['node1', 'node2'])
        assert subgraph.number_of_nodes() == 2
        assert subgraph.number_of_edges() == 1
        
        # Cleanup
        Path(test_db).unlink()
        
        print("✅ NetworkX integration with SQLite works")
        print(f"   - Loaded {G.number_of_nodes()} nodes")
        print(f"   - Loaded {G.number_of_edges()} edges")
        return True
        
    except Exception as e:
        print(f"❌ NetworkX integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all NetworkX and SQLite tests"""
    print("🧪 Testing NetworkX and SQLite Database Components\n")
    
    tests = [
        test_sqlite_basic,
        test_networkx_basic,
        test_kg_builder_small,
        test_graph_rag_retrieval,
        test_networkx_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 NetworkX & SQLite Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All NetworkX and SQLite tests passed!")
        print("✅ Database components are working correctly")
    else:
        print("⚠️  Some tests failed. Issues need to be fixed.")

if __name__ == "__main__":
    main()
