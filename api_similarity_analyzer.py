#!/usr/bin/env python3
"""
API Similarity Analyzer - Zero Cost Implementation
==================================================

A comprehensive tool for analyzing similarity between OpenAPI specifications
without using any paid services or LLMs. Uses only free, open-source libraries.

Implements multi-dimensional similarity analysis:
1. Structural Similarity (paths, methods, parameters)
2. Semantic Similarity (TF-IDF, keyword matching)
3. Schema Similarity (data types, field matching)
4. Functional Similarity (operations, business logic)
5. Composite Scoring (weighted final score)
"""

import yaml
import json
import re
import difflib
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class APIStructureExtractor:
    """Extract structured information from OpenAPI specifications."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_api_spec(self, file_path):
        """Load and parse OpenAPI specification file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_metadata(self, spec):
        """Extract basic API metadata."""
        info = spec.get('info', {})
        return {
            'title': info.get('title', 'Unknown'),
            'version': info.get('version', 'Unknown'),
            'description': info.get('description', ''),
            'openapi_version': spec.get('openapi', spec.get('swagger', 'Unknown')),
            'servers': spec.get('servers', [])
        }
    
    def extract_paths_structure(self, spec):
        """Extract path structure and operations."""
        paths = spec.get('paths', {})
        path_info = {}
        
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
                
            path_info[path] = {
                'methods': [],
                'parameters': [],
                'operations': {},
                'tags': set()
            }
            
            for method, operation in methods.items():
                if method.startswith('x-') or not isinstance(operation, dict):
                    continue
                    
                path_info[path]['methods'].append(method.upper())
                
                # Extract operation details
                op_info = {
                    'operationId': operation.get('operationId', ''),
                    'summary': operation.get('summary', ''),
                    'description': operation.get('description', ''),
                    'tags': operation.get('tags', []),
                    'parameters': operation.get('parameters', []),
                    'requestBody': operation.get('requestBody', {}),
                    'responses': operation.get('responses', {})
                }
                
                path_info[path]['operations'][method] = op_info
                path_info[path]['tags'].update(operation.get('tags', []))
                
                # Extract parameters
                for param in operation.get('parameters', []):
                    if isinstance(param, dict):
                        path_info[path]['parameters'].append({
                            'name': param.get('name', ''),
                            'in': param.get('in', ''),
                            'required': param.get('required', False),
                            'type': self._get_param_type(param)
                        })
        
        return path_info
    
    def extract_schemas(self, spec):
        """Extract schema definitions."""
        components = spec.get('components', {})
        schemas = components.get('schemas', {})
        
        schema_info = {}
        for schema_name, schema_def in schemas.items():
            schema_info[schema_name] = self._parse_schema(schema_def)
        
        return schema_info
    
    def _get_param_type(self, param):
        """Extract parameter type from parameter definition."""
        schema = param.get('schema', {})
        if isinstance(schema, dict):
            return schema.get('type', 'unknown')
        return 'unknown'
    
    def _parse_schema(self, schema):
        """Parse schema definition recursively."""
        if not isinstance(schema, dict):
            return {'type': 'unknown'}
        
        result = {
            'type': schema.get('type', 'unknown'),
            'properties': {},
            'required': schema.get('required', []),
            'items': None
        }
        
        # Handle object properties
        if 'properties' in schema:
            for prop_name, prop_def in schema['properties'].items():
                result['properties'][prop_name] = self._parse_schema(prop_def)
        
        # Handle array items
        if 'items' in schema:
            result['items'] = self._parse_schema(schema['items'])
        
        return result
    
    def extract_text_content(self, spec):
        """Extract all textual content for semantic analysis."""
        text_parts = []
        
        def extract_text_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ['description', 'summary', 'title', 'operationId']:
                        if isinstance(value, str) and value.strip():
                            text_parts.append(value.strip())
                    elif isinstance(value, (dict, list)):
                        extract_text_recursive(value, f"{prefix}.{key}")
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursive(item, prefix)
        
        extract_text_recursive(spec)
        return ' '.join(text_parts)

class StructuralSimilarityAnalyzer:
    """Analyze structural similarity between APIs."""
    
    def calculate_path_similarity(self, paths1, paths2):
        """Calculate similarity between path structures."""
        if not paths1 or not paths2:
            return 0.0
        
        # Normalize paths for comparison
        normalized_paths1 = [self._normalize_path(p) for p in paths1.keys()]
        normalized_paths2 = [self._normalize_path(p) for p in paths2.keys()]
        
        # Calculate Jaccard similarity
        set1, set2 = set(normalized_paths1), set(normalized_paths2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Calculate fuzzy similarity for paths
        fuzzy_scores = []
        for p1 in normalized_paths1:
            best_match = max([fuzz.ratio(p1, p2) for p2 in normalized_paths2] or [0])
            fuzzy_scores.append(best_match / 100.0)
        
        fuzzy_similarity = np.mean(fuzzy_scores) if fuzzy_scores else 0.0
        
        return (jaccard + fuzzy_similarity) / 2
    
    def calculate_method_similarity(self, paths1, paths2):
        """Calculate similarity between HTTP methods used."""
        methods1 = set()
        methods2 = set()
        
        for path_info in paths1.values():
            methods1.update(path_info.get('methods', []))
        
        for path_info in paths2.values():
            methods2.update(path_info.get('methods', []))
        
        if not methods1 and not methods2:
            return 1.0
        if not methods1 or not methods2:
            return 0.0
        
        intersection = len(methods1.intersection(methods2))
        union = len(methods1.union(methods2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_parameter_similarity(self, paths1, paths2):
        """Calculate similarity between parameters."""
        params1 = self._extract_all_parameters(paths1)
        params2 = self._extract_all_parameters(paths2)
        
        if not params1 and not params2:
            return 1.0
        if not params1 or not params2:
            return 0.0
        
        # Compare parameter names
        names1 = set(p['name'] for p in params1)
        names2 = set(p['name'] for p in params2)
        
        name_intersection = len(names1.intersection(names2))
        name_union = len(names1.union(names2))
        name_similarity = name_intersection / name_union if name_union > 0 else 0.0
        
        # Compare parameter types
        types1 = Counter(p['type'] for p in params1)
        types2 = Counter(p['type'] for p in params2)
        
        type_similarity = self._calculate_counter_similarity(types1, types2)
        
        return (name_similarity + type_similarity) / 2
    
    def _normalize_path(self, path):
        """Normalize path by removing specific IDs and parameters."""
        # Replace path parameters with placeholders
        normalized = re.sub(r'\{[^}]+\}', '{id}', path)
        # Remove version numbers
        normalized = re.sub(r'/v\d+/', '/v{n}/', normalized)
        return normalized.lower()
    
    def _extract_all_parameters(self, paths):
        """Extract all parameters from all paths."""
        all_params = []
        for path_info in paths.values():
            all_params.extend(path_info.get('parameters', []))
        return all_params
    
    def _calculate_counter_similarity(self, counter1, counter2):
        """Calculate similarity between two Counter objects."""
        if not counter1 and not counter2:
            return 1.0
        if not counter1 or not counter2:
            return 0.0
        
        keys = set(counter1.keys()).union(set(counter2.keys()))
        similarity = 0.0
        
        for key in keys:
            val1 = counter1.get(key, 0)
            val2 = counter2.get(key, 0)
            similarity += min(val1, val2) / max(val1, val2, 1)
        
        return similarity / len(keys) if keys else 0.0

class SemanticSimilarityAnalyzer:
    """Analyze semantic similarity using NLP techniques."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.business_domains = {
            'banking': ['account', 'transaction', 'balance', 'payment', 'transfer', 'deposit', 'withdrawal', 'loan', 'credit', 'debit', 'bank', 'financial'],
            'authentication': ['auth', 'login', 'logout', 'token', 'oauth', 'session', 'credential', 'password', 'verify'],
            'user_management': ['user', 'profile', 'customer', 'member', 'registration', 'signup', 'account'],
            'ecommerce': ['product', 'order', 'cart', 'checkout', 'payment', 'shipping', 'invoice', 'purchase'],
            'content': ['content', 'article', 'post', 'media', 'document', 'file', 'upload', 'download'],
            'notification': ['notification', 'alert', 'message', 'email', 'sms', 'push'],
            'kyc': ['kyc', 'verification', 'identity', 'document', 'compliance', 'validation', 'lead', 'customer'],
            'data': ['data', 'record', 'table', 'database', 'query', 'filter', 'sort', 'pagination']
        }
    
    def calculate_tfidf_similarity(self, text1, text2):
        """Calculate TF-IDF based similarity."""
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # Preprocess texts
        processed_text1 = self._preprocess_text(text1)
        processed_text2 = self._preprocess_text(text2)
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def calculate_domain_similarity(self, spec1, spec2):
        """Calculate domain-based similarity."""
        domain1 = self._classify_domain(spec1)
        domain2 = self._classify_domain(spec2)
        
        if domain1 == domain2:
            return 1.0
        
        # Check for related domains
        related_domains = {
            'banking': ['authentication', 'user_management'],
            'ecommerce': ['user_management', 'notification'],
            'kyc': ['user_management', 'data', 'authentication']
        }
        
        if domain1 in related_domains and domain2 in related_domains[domain1]:
            return 0.7
        if domain2 in related_domains and domain1 in related_domains[domain2]:
            return 0.7
        
        return 0.0
    
    def _preprocess_text(self, text):
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters except spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and stem
        processed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def _classify_domain(self, spec):
        """Classify API into business domain."""
        text_content = json.dumps(spec).lower()
        
        domain_scores = {}
        for domain, keywords in self.business_domains.items():
            score = sum(text_content.count(keyword) for keyword in keywords)
            domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'unknown'

class SchemaSimilarityAnalyzer:
    """Analyze similarity between data schemas."""
    
    def calculate_schema_similarity(self, schemas1, schemas2):
        """Calculate overall schema similarity."""
        if not schemas1 and not schemas2:
            return 1.0
        if not schemas1 or not schemas2:
            return 0.0
        
        # Compare schema names
        names1 = set(schemas1.keys())
        names2 = set(schemas2.keys())
        
        name_similarity = len(names1.intersection(names2)) / len(names1.union(names2))
        
        # Compare schema structures
        structure_similarities = []
        for name1, schema1 in schemas1.items():
            best_match = 0.0
            for name2, schema2 in schemas2.items():
                similarity = self._compare_schemas(schema1, schema2)
                best_match = max(best_match, similarity)
            structure_similarities.append(best_match)
        
        structure_similarity = np.mean(structure_similarities) if structure_similarities else 0.0
        
        return (name_similarity + structure_similarity) / 2
    
    def _compare_schemas(self, schema1, schema2):
        """Compare two individual schemas."""
        if not isinstance(schema1, dict) or not isinstance(schema2, dict):
            return 0.0
        
        # Compare types
        type1 = schema1.get('type', 'unknown')
        type2 = schema2.get('type', 'unknown')
        type_match = 1.0 if type1 == type2 else 0.0
        
        # Compare properties
        props1 = schema1.get('properties', {})
        props2 = schema2.get('properties', {})
        
        if not props1 and not props2:
            prop_similarity = 1.0
        elif not props1 or not props2:
            prop_similarity = 0.0
        else:
            prop_names1 = set(props1.keys())
            prop_names2 = set(props2.keys())
            prop_similarity = len(prop_names1.intersection(prop_names2)) / len(prop_names1.union(prop_names2))
        
        # Compare required fields
        req1 = set(schema1.get('required', []))
        req2 = set(schema2.get('required', []))
        
        if not req1 and not req2:
            req_similarity = 1.0
        elif not req1 or not req2:
            req_similarity = 0.0
        else:
            req_similarity = len(req1.intersection(req2)) / len(req1.union(req2))
        
        return (type_match + prop_similarity + req_similarity) / 3

class APISimilarityAnalyzer:
    """Main analyzer that combines all similarity metrics."""
    
    def __init__(self):
        self.extractor = APIStructureExtractor()
        self.structural_analyzer = StructuralSimilarityAnalyzer()
        self.semantic_analyzer = SemanticSimilarityAnalyzer()
        self.schema_analyzer = SchemaSimilarityAnalyzer()
        
        # Weights for different similarity components
        self.weights = {
            'structural': 0.25,
            'semantic': 0.25,
            'schema': 0.25,
            'functional': 0.25
        }
    
    def analyze_similarity(self, api1_path, api2_path):
        """Perform comprehensive similarity analysis."""
        # Load API specifications
        spec1 = self.extractor.load_api_spec(api1_path)
        spec2 = self.extractor.load_api_spec(api2_path)
        
        if not spec1 or not spec2:
            return None
        
        # Extract structured information
        metadata1 = self.extractor.extract_metadata(spec1)
        metadata2 = self.extractor.extract_metadata(spec2)
        
        paths1 = self.extractor.extract_paths_structure(spec1)
        paths2 = self.extractor.extract_paths_structure(spec2)
        
        schemas1 = self.extractor.extract_schemas(spec1)
        schemas2 = self.extractor.extract_schemas(spec2)
        
        text1 = self.extractor.extract_text_content(spec1)
        text2 = self.extractor.extract_text_content(spec2)
        
        # Calculate similarity scores
        structural_score = self._calculate_structural_similarity(paths1, paths2)
        semantic_score = self._calculate_semantic_similarity(spec1, spec2, text1, text2)
        schema_score = self.schema_analyzer.calculate_schema_similarity(schemas1, schemas2)
        functional_score = self._calculate_functional_similarity(paths1, paths2, metadata1, metadata2)
        
        # Calculate weighted final score
        final_score = (
            structural_score * self.weights['structural'] +
            semantic_score * self.weights['semantic'] +
            schema_score * self.weights['schema'] +
            functional_score * self.weights['functional']
        ) * 100  # Convert to percentage
        
        return {
            'final_score': final_score,
            'detailed_scores': {
                'structural': structural_score * 100,
                'semantic': semantic_score * 100,
                'schema': schema_score * 100,
                'functional': functional_score * 100
            },
            'metadata': {
                'api1': metadata1,
                'api2': metadata2
            },
            'analysis': self._generate_analysis_report(
                final_score, structural_score, semantic_score, 
                schema_score, functional_score, metadata1, metadata2
            )
        }
    
    def _calculate_structural_similarity(self, paths1, paths2):
        """Calculate overall structural similarity."""
        path_sim = self.structural_analyzer.calculate_path_similarity(paths1, paths2)
        method_sim = self.structural_analyzer.calculate_method_similarity(paths1, paths2)
        param_sim = self.structural_analyzer.calculate_parameter_similarity(paths1, paths2)
        
        return (path_sim + method_sim + param_sim) / 3
    
    def _calculate_semantic_similarity(self, spec1, spec2, text1, text2):
        """Calculate overall semantic similarity."""
        tfidf_sim = self.semantic_analyzer.calculate_tfidf_similarity(text1, text2)
        domain_sim = self.semantic_analyzer.calculate_domain_similarity(spec1, spec2)
        
        return (tfidf_sim + domain_sim) / 2
    
    def _calculate_functional_similarity(self, paths1, paths2, metadata1, metadata2):
        """Calculate functional similarity based on operations and business logic."""
        # Analyze CRUD operations
        crud1 = self._analyze_crud_operations(paths1)
        crud2 = self._analyze_crud_operations(paths2)
        
        crud_similarity = self._compare_crud_patterns(crud1, crud2)
        
        # Analyze authentication patterns
        auth_similarity = self._compare_auth_patterns(metadata1, metadata2)
        
        return (crud_similarity + auth_similarity) / 2
    
    def _analyze_crud_operations(self, paths):
        """Analyze CRUD operation patterns."""
        crud_patterns = {'create': 0, 'read': 0, 'update': 0, 'delete': 0}
        
        for path_info in paths.values():
            methods = path_info.get('methods', [])
            if 'POST' in methods:
                crud_patterns['create'] += 1
            if 'GET' in methods:
                crud_patterns['read'] += 1
            if 'PUT' in methods or 'PATCH' in methods:
                crud_patterns['update'] += 1
            if 'DELETE' in methods:
                crud_patterns['delete'] += 1
        
        return crud_patterns
    
    def _compare_crud_patterns(self, crud1, crud2):
        """Compare CRUD operation patterns."""
        if not any(crud1.values()) and not any(crud2.values()):
            return 1.0
        
        total1 = sum(crud1.values())
        total2 = sum(crud2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        # Normalize and compare distributions
        norm1 = {k: v/total1 for k, v in crud1.items()}
        norm2 = {k: v/total2 for k, v in crud2.items()}
        
        similarity = 0.0
        for operation in crud1.keys():
            similarity += 1 - abs(norm1[operation] - norm2[operation])
        
        return similarity / len(crud1)
    
    def _compare_auth_patterns(self, metadata1, metadata2):
        """Compare authentication patterns (simplified)."""
        # This is a simplified comparison - in a real implementation,
        # you would analyze security schemes from the OpenAPI specs
        return 0.5  # Neutral score for now
    
    def _generate_analysis_report(self, final_score, structural, semantic, schema, functional, metadata1, metadata2):
        """Generate detailed analysis report."""
        category = self._categorize_similarity(final_score)
        
        report = {
            'similarity_category': category,
            'recommendation': self._get_recommendation(final_score),
            'key_differences': [],
            'consolidation_potential': self._assess_consolidation_potential(final_score),
            'domain_analysis': {
                'api1_domain': self.semantic_analyzer._classify_domain(metadata1),
                'api2_domain': self.semantic_analyzer._classify_domain(metadata2)
            }
        }
        
        return report
    
    def _categorize_similarity(self, score):
        """Categorize similarity score according to prompt template."""
        if score >= 95:
            return "Near-identical APIs"
        elif score >= 85:
            return "High similarity"
        elif score >= 70:
            return "Moderate similarity"
        elif score >= 50:
            return "Some overlap"
        else:
            return "Low similarity"
    
    def _get_recommendation(self, score):
        """Get recommendation based on score."""
        if score >= 95:
            return "Immediate consolidation candidate"
        elif score >= 85:
            return "Strong consolidation potential with minor adaptations"
        elif score >= 70:
            return "Evaluate for extension or partial consolidation"
        elif score >= 50:
            return "Monitor for potential future consolidation"
        else:
            return "Likely legitimate separate APIs"
    
    def _assess_consolidation_potential(self, score):
        """Assess consolidation potential."""
        if score >= 85:
            return "High"
        elif score >= 70:
            return "Medium"
        elif score >= 50:
            return "Low"
        else:
            return "Very Low"

def format_similarity_report(result, api1_name, api2_name):
    """Format the similarity analysis result into a readable report."""
    if not result:
        return "Error: Could not analyze API specifications."
    
    final_score = result['final_score']
    detailed = result['detailed_scores']
    metadata = result['metadata']
    analysis = result['analysis']
    
    report = f"""
# API Similarity Analysis Report

## APIs Compared
- **Source API**: {metadata['api1']['title']} (v{metadata['api1']['version']})
- **Target API**: {metadata['api2']['title']} (v{metadata['api2']['version']})

## Similarity Score: {final_score:.1f}%

### Category: {analysis['similarity_category']}
**Recommendation**: {analysis['recommendation']}

## Detailed Analysis

### Similarity Breakdown
- **Structural Similarity**: {detailed['structural']:.1f}%
  - Path structure, HTTP methods, parameters
- **Semantic Similarity**: {detailed['semantic']:.1f}%
  - Content analysis, domain classification
- **Schema Similarity**: {detailed['schema']:.1f}%
  - Data models, field structures
- **Functional Similarity**: {detailed['functional']:.1f}%
  - CRUD operations, business logic patterns

### Domain Analysis
- **API 1 Domain**: {analysis['domain_analysis']['api1_domain'].title()}
- **API 2 Domain**: {analysis['domain_analysis']['api2_domain'].title()}

### Consolidation Assessment
- **Potential**: {analysis['consolidation_potential']}
- **Risk Level**: {"Low" if final_score >= 70 else "Medium" if final_score >= 50 else "High"}

## Summary
Based on the multi-dimensional analysis, these APIs show **{analysis['similarity_category'].lower()}** 
with a composite score of **{final_score:.1f}%**. {analysis['recommendation']}.

---
*Analysis performed using zero-cost, open-source similarity detection framework*
"""
    
    return report

def main():
    """Main function to run the API similarity analysis."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python api_similarity_analyzer.py <api1_path> <api2_path>")
        sys.exit(1)
    
    api1_path = sys.argv[1]
    api2_path = sys.argv[2]
    
    analyzer = APISimilarityAnalyzer()
    result = analyzer.analyze_similarity(api1_path, api2_path)
    
    if result:
        report = format_similarity_report(result, api1_path, api2_path)
        print(report)
        
        # Save report to file
        output_file = "api_similarity_report.md"
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nDetailed report saved to: {output_file}")
    else:
        print("Error: Could not analyze the provided API specifications.")

if __name__ == "__main__":
    main()