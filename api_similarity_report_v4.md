
# Graph Neural Network API Similarity Analysis Report (v4)

## APIs Compared
- **Source API**: AMH_Accounts_Swagger (3)
- **Target API**: spec

## Similarity Score: 78.3%

### Category: Moderate similarity
**Recommendation**: Evaluate for consolidation

## Graph Neural Network Analysis

### Advanced Similarity Breakdown
- **GNN Embedding Similarity**: 97.2%
  - Deep graph structure and semantic understanding through neural networks
- **Structural Graph Similarity**: 59.4%
  - Graph topology and statistical properties comparison

### Graph Structure Analysis

#### API 1 Graph Statistics
- **Nodes**: 249 (API components)
- **Edges**: 214 (relationships)
- **Average Degree**: 1.72
- **Graph Density**: 0.003

#### API 2 Graph Statistics  
- **Nodes**: 120 (API components)
- **Edges**: 111 (relationships)
- **Average Degree**: 1.85
- **Graph Density**: 0.008

### Graph Neural Network Architecture

#### Node Types Modeled
- **API Root**: Overall API characteristics
- **Paths**: Endpoint structure and organization
- **Operations**: HTTP methods and business logic
- **Parameters**: Input specifications and constraints
- **Schemas**: Data model definitions
- **Properties**: Individual field characteristics
- **Responses**: Output specifications

#### Edge Types Modeled
- **Contains**: Hierarchical containment relationships
- **Uses**: Operational dependencies and references
- **Returns**: Response relationships
- **References**: Schema and component references

#### GNN Processing Pipeline
1. **Graph Construction**: Convert OpenAPI spec to rich graph representation
2. **Feature Extraction**: Extract numerical features for all nodes and edges  
3. **Graph Neural Network**: Multi-layer message passing for node embeddings
4. **Graph Pooling**: Aggregate node embeddings to graph-level representation
5. **Similarity Calculation**: Compare graph embeddings using cosine similarity

### Key Innovations in v4

#### Advanced Graph Representation
- **Rich Node Features**: 45+ numerical features per node type
- **Semantic Edge Types**: 7 different relationship categories
- **Hierarchical Structure**: Captures API component relationships
- **Centrality Measures**: Graph topology analysis

#### Lightweight GNN Implementation
- **Pure NumPy**: No PyTorch dependency, CPU-friendly
- **Custom Architecture**: 2-layer GNN with attention pooling
- **Message Passing**: Neighbor aggregation with feature combination
- **Graph Pooling**: Multiple pooling strategies for graph-level embeddings

#### Zero-Cost Approach
- **NetworkX**: Free graph construction and analysis
- **Scikit-network**: Lightweight graph processing
- **Custom GNN**: Implemented from scratch using NumPy
- **No External APIs**: Completely offline processing

## Summary
Based on advanced Graph Neural Network analysis with comprehensive graph representation,
these APIs show **moderate similarity** with a composite score of **78.3%**.

The v4 analyzer provides state-of-the-art similarity analysis through:
- Complete API-to-graph conversion with rich features
- Custom Graph Neural Network for deep structural understanding
- Advanced pooling techniques for graph-level embeddings
- Comprehensive similarity analysis combining multiple graph perspectives

**Evaluate for consolidation**.

---
*Analysis performed using state-of-the-art Graph Neural Network framework v4*
*Enhanced with complete graph representation and deep learning techniques*
