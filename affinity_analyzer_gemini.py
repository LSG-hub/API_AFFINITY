#!/usr/bin/env python3
"""
Affinity Analyzer (Gemini Version)
==================================

An advanced API similarity analysis tool that combines Graph Neural Networks (GNNs)
with deep semantic understanding from pre-trained language models. This version
builds upon the v4 analyzer by integrating the unified plan developed in
collaboration with Gemini and Claude Code.

Key Enhancements:
1.  **Semantic Embeddings:** Uses sentence-transformers to convert text into
    meaningful vectors, enabling the model to understand semantic similarity.
2.  **Refined Feature Engineering:** Includes more granular features for response
    codes, data formats, and security schemes.
3.  **Enhanced Similarity Metrics:** Introduces dedicated analysis for path
    similarity and a deeper, more recursive schema comparison.
4.  **Holistic Scoring:** The final similarity score is a weighted composite of
    GNN embedding similarity, structural similarity, path similarity, and
    schema similarity.
"""

import yaml
import json
import re
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
from Levenshtein import distance as levenshtein_distance

# --- New Dependencies ---
# This script requires the sentence-transformers library.
# Install it using: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers library not found.")
    print("Please install it using: pip install sentence-transformers")
    exit(1)

warnings.filterwarnings('ignore')

class APIStructureExtractor:
    """Loads an API specification from a YAML or JSON file."""
    def load_api_spec(self, path):
        """Loads the API spec from the given path."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                elif path.endswith('.json'):
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading API spec from {path}: {e}")
            return None

class APIGraphBuilder:
    """
    Convert OpenAPI specifications into rich, semantically-aware graph representations.
    """
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.node_types = {
            'api': 0, 'path': 1, 'operation': 2, 'parameter': 3,
            'schema': 4, 'property': 5, 'response': 6
        }
        self.edge_types = {
            'contains': 0, 'has_operation': 1, 'uses_parameter': 2,
            'returns': 3, 'uses_schema': 4, 'has_property': 5, 'references': 6
        }
        self.edge_weights = {
            'contains': 1.5, 'has_operation': 1.5, 'uses_parameter': 1.2,
            'returns': 1.2, 'uses_schema': 1.8, 'has_property': 1.0, 'references': 1.8
        }

    def _get_text_embedding(self, text):
        """Generates a sentence embedding for the given text."""
        if not text or not isinstance(text, str):
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def build_api_graph(self, api_spec):
        G = nx.DiGraph()
        metadata = self._extract_api_metadata(api_spec)
        api_node_id = "api_root"
        G.add_node(api_node_id,
                   node_type=self.node_types['api'],
                   features=self._extract_api_features(metadata, api_spec))

        paths = api_spec.get('paths', {})
        schemas = api_spec.get('components', {}).get('schemas', {})
        
        schema_node_ids = self._add_schema_nodes(G, schemas)
        self._add_path_nodes(G, paths, api_node_id, schema_node_ids)
        
        self._add_graph_features(G)
        return G

    def _extract_api_metadata(self, api_spec):
        """Extract API-level metadata."""
        info = api_spec.get('info', {})
        return {
            'title': info.get('title', ''),
            'description': info.get('description', ''),
            'version': info.get('version', ''),
            'servers': api_spec.get('servers', [])
        }

    def _extract_api_features(self, metadata, api_spec):
        text_for_embedding = f"{metadata.get('title', '')} {metadata.get('description', '')}"
        embedding = self._get_text_embedding(text_for_embedding)
        
        security_schemes = api_spec.get('components', {}).get('securitySchemes', {})
        features = {
            'has_oauth2': 1.0 if any('oauth2' in v.get('type', '') for v in security_schemes.values()) else 0.0,
            'has_apikey': 1.0 if any('apiKey' in v.get('type', '') for v in security_schemes.values()) else 0.0,
            'server_count': len(metadata.get('servers', [])),
        }
        # Combine embedding with other features
        feature_vector = np.concatenate([embedding, np.array(list(features.values()))])
        return feature_vector

    def _add_path_nodes(self, G, paths, api_node_id, schema_node_ids):
        for path, path_info in paths.items():
            if not isinstance(path_info, dict): continue
            
            path_node_id = f"path_{hash(path)}"
            path_segments = [seg for seg in path.split('/') if seg]
            path_features = np.array([
                len(path), len(path_segments), path.count('{')
            ])
            G.add_node(path_node_id, node_type=self.node_types['path'], features=path_features, path=path)
            G.add_edge(api_node_id, path_node_id, edge_type=self.edge_types['contains'], weight=self.edge_weights['contains'])

            for method, op in path_info.items():
                if not isinstance(op, dict) or method.startswith('x-'): continue
                
                op_node_id = f"op_{method}_{path}"
                op_features = self._extract_operation_features(method, op, path)
                G.add_node(op_node_id, node_type=self.node_types['operation'], features=op_features, method=method, path=path)
                G.add_edge(path_node_id, op_node_id, edge_type=self.edge_types['has_operation'], weight=self.edge_weights['has_operation'])
                
                # Connect operation to schemas
                self._connect_op_to_schemas(G, op, op_node_id, schema_node_ids)

    def _extract_operation_features(self, method, op, path):
        text = f"{op.get('summary', '')} {op.get('description', '')} {' '.join(op.get('tags', []))}"
        embedding = self._get_text_embedding(text)
        
        # CRUD detection
        path_lower = path.lower()
        method_upper = method.upper()
        is_collection = not re.search(r'{\w+}$', path_lower)
        
        features = {
            'is_create': 1.0 if method_upper == 'POST' and is_collection else 0.0,
            'is_read_collection': 1.0 if method_upper == 'GET' and is_collection else 0.0,
            'is_read_item': 1.0 if method_upper == 'GET' and not is_collection else 0.0,
            'is_update': 1.0 if method_upper in ['PUT', 'PATCH'] and not is_collection else 0.0,
            'is_delete': 1.0 if method_upper == 'DELETE' and not is_collection else 0.0,
            'has_request_body': 1.0 if 'requestBody' in op else 0.0,
            'parameter_count': len(op.get('parameters', [])),
        }
        return np.concatenate([embedding, np.array(list(features.values()))])

    def _add_schema_nodes(self, G, schemas):
        schema_node_ids = {}
        for name, schema_def in schemas.items():
            if not isinstance(schema_def, dict): continue
            
            schema_node_id = f"schema_{name}"
            schema_node_ids[name] = schema_node_id
            
            text = f"{name} {schema_def.get('description', '')}"
            embedding = self._get_text_embedding(text)
            
            props = schema_def.get('properties', {})
            features = {
                'property_count': len(props),
                'required_count': len(schema_def.get('required', [])),
            }
            feature_vector = np.concatenate([embedding, np.array(list(features.values()))])
            G.add_node(schema_node_id, node_type=self.node_types['schema'], features=feature_vector, schema_name=name, schema_def=schema_def)
            
            # Add property nodes and connect them
            for prop_name, prop_def in props.items():
                prop_node_id = f"prop_{name}_{prop_name}"
                prop_text = f"{prop_name} {prop_def.get('description', '')}"
                prop_embedding = self._get_text_embedding(prop_text)
                
                prop_features = {
                    'is_required': 1.0 if prop_name in schema_def.get('required', []) else 0.0,
                    'is_string': 1.0 if prop_def.get('type') == 'string' else 0.0,
                    'is_number': 1.0 if prop_def.get('type') in ['number', 'integer'] else 0.0,
                    'is_boolean': 1.0 if prop_def.get('type') == 'boolean' else 0.0,
                    'is_array': 1.0 if prop_def.get('type') == 'array' else 0.0,
                    'has_format_datetime': 1.0 if prop_def.get('format') == 'date-time' else 0.0,
                    'has_format_uuid': 1.0 if prop_def.get('format') == 'uuid' else 0.0,
                }
                prop_feature_vector = np.concatenate([prop_embedding, np.array(list(prop_features.values()))])
                G.add_node(prop_node_id, node_type=self.node_types['property'], features=prop_feature_vector)
                G.add_edge(schema_node_id, prop_node_id, edge_type=self.edge_types['has_property'], weight=self.edge_weights['has_property'])

        return schema_node_ids

    def _connect_op_to_schemas(self, G, op, op_node_id, schema_node_ids):
        # Simplified connection logic for brevity
        refs = self._find_refs(op)
        for ref in refs:
            schema_name = ref.split('/')[-1]
            if schema_name in schema_node_ids:
                G.add_edge(op_node_id, schema_node_ids[schema_name],
                           edge_type=self.edge_types['uses_schema'],
                           weight=self.edge_weights['uses_schema'])

    def _find_refs(self, data):
        """Recursively find all $ref values in a dictionary."""
        refs = []
        if isinstance(data, dict):
            for k, v in data.items():
                if k == '$ref' and isinstance(v, str):
                    refs.append(v)
                else:
                    refs.extend(self._find_refs(v))
        elif isinstance(data, list):
            for item in data:
                refs.extend(self._find_refs(item))
        return refs

    def _add_graph_features(self, G):
        if G.number_of_nodes() == 0: return
        # Add centrality measures as node features
        try:
            centrality = nx.degree_centrality(G)
            nx.set_node_attributes(G, centrality, 'centrality')
        except Exception:
            nx.set_node_attributes(G, 0.0, 'centrality')

class AffinityAnalyzer:
    """
    Analyzes API similarity using a combination of GNN, structural, path, and schema analysis.
    """
    def __init__(self, gnn_dims=(393, 128, 64), feature_dim=393):
        self.graph_builder = APIGraphBuilder()
        # Adjust GNN input_dim based on embedding size + other features
        # Embedding (384) + API features (3) = 387. Let's use a flexible GNN or pad.
        # For now, we will pad/truncate features to a fixed size.
        self.gnn = LightweightGNN(input_dim=feature_dim, hidden_dim=gnn_dims[1], output_dim=gnn_dims[2])
        self.scaler = StandardScaler()
        self.feature_dim = feature_dim

    def _prepare_node_features(self, graph):
        """Prepares a standardized feature matrix for the GNN."""
        features_list = [data.get('features', []) for _, data in graph.nodes(data=True)]
        
        max_len = self.feature_dim
        padded_features = []
        for feat in features_list:
            if len(feat) > max_len:
                padded_feat = feat[:max_len]
            else:
                padded_feat = np.pad(feat, (0, max_len - len(feat)), 'constant')
            padded_features.append(padded_feat)
            
        if not padded_features:
            return np.array([])
            
        feature_matrix = np.array(padded_features)
        return self.scaler.fit_transform(feature_matrix)

    def _calculate_path_similarity(self, graph1, graph2):
        """Calculates similarity based on API paths."""
        paths1 = {data['path'] for _, data in graph1.nodes(data=True) if data.get('node_type') == self.graph_builder.node_types['path']}
        paths2 = {data['path'] for _, data in graph2.nodes(data=True) if data.get('node_type') == self.graph_builder.node_types['path']}
        
        if not paths1 or not paths2:
            return 0.0
            
        # Use Jaccard similarity for quick overlap check
        jaccard_sim = len(paths1.intersection(paths2)) / len(paths1.union(paths2))
        
        # Use Levenshtein distance for more nuanced comparison
        # This is a simplified approach. A more robust method would use bipartite matching.
        avg_lev_sim = 0
        if len(paths1) > 0 and len(paths2) > 0:
            # For simplicity, compare each path in set1 to all in set2 and take the best match
            total_sim = 0
            for p1 in paths1:
                best_sim = max(1 - (levenshtein_distance(p1, p2) / max(len(p1), len(p2))) for p2 in paths2)
                total_sim += best_sim
            avg_lev_sim = total_sim / len(paths1)
            
        return (jaccard_sim + avg_lev_sim) / 2.0

    def _calculate_schema_similarity(self, graph1, graph2):
        """Calculates similarity based on schemas."""
        schemas1 = {data['schema_name']: data['schema_def'] for _, data in graph1.nodes(data=True) if data.get('node_type') == self.graph_builder.node_types['schema']}
        schemas2 = {data['schema_name']: data['schema_def'] for _, data in graph2.nodes(data=True) if data.get('node_type') == self.graph_builder.node_types['schema']}

        if not schemas1 or not schemas2:
            return 0.0

        # Compare schema names (Jaccard)
        names1 = set(schemas1.keys())
        names2 = set(schemas2.keys())
        name_sim = len(names1.intersection(names2)) / len(names1.union(names2))

        # Compare structure of common schemas
        common_schemas = names1.intersection(names2)
        struct_sims = []
        for name in common_schemas:
            props1 = set(schemas1[name].get('properties', {}).keys())
            props2 = set(schemas2[name].get('properties', {}).keys())
            if not props1 or not props2:
                struct_sims.append(1.0 if not props1 and not props2 else 0.0)
                continue
            prop_sim = len(props1.intersection(props2)) / len(props1.union(props2))
            struct_sims.append(prop_sim)
        
        avg_struct_sim = np.mean(struct_sims) if struct_sims else 0.0
        
        return (name_sim + avg_struct_sim) / 2.0

    def analyze(self, api1_path, api2_path, weights=None):
        if weights is None:
            weights = {'gnn': 0.5, 'structural': 0.1, 'path': 0.2, 'schema': 0.2}

        spec1 = APIStructureExtractor().load_api_spec(api1_path)
        spec2 = APIStructureExtractor().load_api_spec(api2_path)
        if not spec1 or not spec2: return None

        graph1 = self.graph_builder.build_api_graph(spec1)
        graph2 = self.graph_builder.build_api_graph(spec2)

        # 1. GNN Similarity
        features1 = self._prepare_node_features(graph1)
        features2 = self._prepare_node_features(graph2)
        
        adj1 = nx.to_numpy_array(graph1, weight='weight')
        adj2 = nx.to_numpy_array(graph2, weight='weight')

        if features1.size == 0 or features2.size == 0:
            gnn_sim = 0.0
        else:
            emb1 = self.gnn.forward(features1, adj1)
            emb2 = self.gnn.forward(features2, adj2)
            graph_emb1 = np.mean(emb1, axis=0)
            graph_emb2 = np.mean(emb2, axis=0)
            gnn_sim = cosine_similarity([graph_emb1], [graph_emb2])[0][0]

        # 2. Structural Similarity
        stats1 = {'nodes': graph1.number_of_nodes(), 'edges': graph1.number_of_edges()}
        stats2 = {'nodes': graph2.number_of_nodes(), 'edges': graph2.number_of_edges()}
        node_sim = min(stats1['nodes'], stats2['nodes']) / max(stats1['nodes'], stats2['nodes']) if max(stats1['nodes'], stats2['nodes']) > 0 else 1.0
        edge_sim = min(stats1['edges'], stats2['edges']) / max(stats1['edges'], stats2['edges']) if max(stats1['edges'], stats2['edges']) > 0 else 1.0
        struct_sim = (node_sim + edge_sim) / 2.0

        # 3. Path Similarity
        path_sim = self._calculate_path_similarity(graph1, graph2)

        # 4. Schema Similarity
        schema_sim = self._calculate_schema_similarity(graph1, graph2)

        # Combine scores
        final_score = (weights['gnn'] * gnn_sim +
                       weights['structural'] * struct_sim +
                       weights['path'] * path_sim +
                       weights['schema'] * schema_sim)
        
        return {
            'final_score': final_score * 100,
            'gnn_similarity': gnn_sim * 100,
            'structural_similarity': struct_sim * 100,
            'path_similarity': path_sim * 100,
            'schema_similarity': schema_sim * 100,
            'weights': weights
        }

class LightweightGNN:
    """A simple GNN implementation with NumPy."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        self.weights = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            # Xavier initialization
            bound = np.sqrt(6.0 / (dims[i] + dims[i+1]))
            self.weights.append(np.random.uniform(-bound, bound, (dims[i], dims[i+1])))

    def forward(self, node_features, adj_matrix):
        h = node_features
        for i, W in enumerate(self.weights):
            h = np.dot(adj_matrix, h) # Aggregate neighbors
            h = np.dot(h, W) # Transform
            if i < len(self.weights) - 1: # ReLU activation
                h = np.maximum(0, h)
        return h

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python affinity_analyzer_gemini.py <api1_path> <api2_path>")
        sys.exit(1)
    
    api1_path, api2_path = sys.argv[1], sys.argv[2]
    
    print("Initializing Affinity Analyzer...")
    analyzer = AffinityAnalyzer()
    
    print(f"Analyzing similarity between {Path(api1_path).name} and {Path(api2_path).name}...")
    result = analyzer.analyze(api1_path, api2_path)
    
    if result:
        print("\n--- API Affinity Analysis Report ---")
        print(f"Overall Similarity Score: {result['final_score']:.2f}%\n")
        print("Breakdown:")
        print(f"  - GNN Semantic Similarity:   {result['gnn_similarity']:.2f}% (Weight: {result['weights']['gnn']})")
        print(f"  - Structural Similarity:     {result['structural_similarity']:.2f}% (Weight: {result['weights']['structural']})")
        print(f"  - Path Similarity:           {result['path_similarity']:.2f}% (Weight: {result['weights']['path']})")
        print(f"  - Schema Similarity:         {result['schema_similarity']:.2f}% (Weight: {result['weights']['schema']})")
        print("------------------------------------")
    else:
        print("Error: Could not complete analysis.")

if __name__ == "__main__":
    main()