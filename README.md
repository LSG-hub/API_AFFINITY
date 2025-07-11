# Zero-Cost API Similarity Analyzer

A comprehensive tool for analyzing similarity between OpenAPI specifications without using any paid services or LLMs. This implementation uses only free, open-source libraries and provides detailed similarity analysis across multiple dimensions.

## Features

### 🔍 Multi-Dimensional Analysis
- **Structural Similarity**: Path structures, HTTP methods, parameters
- **Semantic Similarity**: TF-IDF analysis, domain classification  
- **Schema Similarity**: Data models, field structures, types
- **Functional Similarity**: CRUD operations, business logic patterns

### 📊 Comprehensive Scoring
- Weighted composite scoring (0-100%)
- Category classification based on prompt template criteria:
  - 95-100%: Near-identical APIs (immediate consolidation)
  - 85-94%: High similarity (strong consolidation potential)
  - 70-84%: Moderate similarity (evaluate for extension)
  - 50-69%: Some overlap (monitor for future consolidation)
  - 0-49%: Low similarity (likely legitimate separate APIs)

### 🆓 Zero Cost
- Uses only free, open-source libraries
- No API calls to paid services
- No LLM dependencies
- Runs completely offline

## Installation

1. **Create Virtual Environment**
   ```bash
   python3 -m venv api_similarity_env
   source api_similarity_env/bin/activate  # On Windows: api_similarity_env\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install pyyaml scikit-learn nltk pandas numpy fuzzywuzzy python-levenshtein
   ```

3. **Download NLTK Data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
   ```

## Usage

### Basic Usage
```bash
python api_similarity_analyzer.py <path_to_api1.yaml> <path_to_api2.yaml>
```

### Example
```bash
python api_similarity_analyzer.py "684a73d5d272fe65eaa65f70/contract/AMH_Accounts_Swagger (3).yaml" "6852a02080b2ce09ba28586c/contract/spec.yaml"
```

### Output
The tool generates:
1. **Console Output**: Summary report with similarity score and category
2. **Detailed Report**: `api_similarity_report.md` with comprehensive analysis

## Technical Implementation

### Libraries Used
- **PyYAML**: OpenAPI specification parsing
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **NLTK**: Natural language processing and tokenization
- **fuzzywuzzy**: Fuzzy string matching for path similarity
- **pandas/numpy**: Data manipulation and numerical operations

### Architecture

```
APIStructureExtractor
├── Metadata extraction (title, version, domain)
├── Path structure analysis
├── Schema parsing
└── Text content extraction

StructuralSimilarityAnalyzer
├── Path pattern matching
├── HTTP method comparison
└── Parameter analysis

SemanticSimilarityAnalyzer
├── TF-IDF similarity calculation
├── Domain classification
└── Keyword matching

SchemaSimilarityAnalyzer
├── Schema structure comparison
├── Field type analysis
└── Required field matching

APISimilarityAnalyzer (Main)
├── Weighted composite scoring
├── Report generation
└── Recommendation engine
```

### Similarity Calculation

The final similarity score is calculated as a weighted average:

```
Final Score = (Structural × 0.25) + (Semantic × 0.25) + (Schema × 0.25) + (Functional × 0.25)
```

Where each component uses specific algorithms:
- **Structural**: Jaccard similarity + fuzzy matching
- **Semantic**: TF-IDF cosine similarity + domain classification
- **Schema**: Field overlap + type matching
- **Functional**: CRUD pattern analysis + authentication comparison

## Advantages Over Alternative Methods

### vs. Graph Neural Networks (GNN, GraphSAGE, etc.)
- ✅ **Simpler**: No complex graph construction required
- ✅ **Faster**: Linear time complexity vs. quadratic for GNNs
- ✅ **Zero Cost**: No GPU requirements or training data needed
- ✅ **Interpretable**: Clear contribution of each similarity dimension

### vs. GraphCodeBert/LLM-based Methods
- ✅ **No LLM Dependency**: Completely offline and free
- ✅ **No API Costs**: Zero ongoing operational costs
- ✅ **Privacy**: All processing happens locally
- ✅ **Consistent**: No variability from model updates

### vs. Multimodal Embeddings
- ✅ **No Pre-trained Models**: Uses classical ML techniques
- ✅ **Lightweight**: Minimal memory and compute requirements
- ✅ **Customizable**: Easy to adjust weights and add domain-specific rules

## Sample Analysis Result

```
# API Similarity Analysis Report

## Similarity Score: 32.4%
### Category: Low similarity
**Recommendation**: Likely legitimate separate APIs

## Detailed Analysis
### Similarity Breakdown
- **Structural Similarity**: 33.8%
- **Semantic Similarity**: 5.7%
- **Schema Similarity**: 20.4%
- **Functional Similarity**: 69.8%

### Domain Analysis
- **API 1 Domain**: Banking
- **API 2 Domain**: KYC

### Consolidation Assessment
- **Potential**: Very Low
- **Risk Level**: High
```

## Customization

### Adjusting Weights
Modify the weights in `APISimilarityAnalyzer.__init__()`:
```python
self.weights = {
    'structural': 0.25,    # Adjust based on your priorities
    'semantic': 0.25,      # Higher for domain-specific analysis
    'schema': 0.25,        # Higher for data-focused APIs
    'functional': 0.25     # Higher for operation-focused analysis
}
```

### Adding Domain Keywords
Extend domain classification in `SemanticSimilarityAnalyzer`:
```python
self.business_domains = {
    'your_domain': ['keyword1', 'keyword2', ...],
    # ... existing domains
}
```

### Custom Scoring Thresholds
Modify similarity categorization in `_categorize_similarity()` method.

## Future Enhancements

1. **Authentication Pattern Analysis**: Deep analysis of security schemes
2. **API Versioning Similarity**: Version compatibility assessment
3. **Performance Impact Analysis**: Response time and payload size comparison
4. **Business Rule Extraction**: Enhanced functional similarity detection
5. **Batch Processing**: Multiple API comparison support

## Contributing

This is a zero-cost, open-source implementation. Feel free to:
- Add new similarity metrics
- Improve domain classification
- Enhance schema parsing
- Optimize performance

## License

Open-source implementation using only free libraries. No licensing restrictions from paid services.