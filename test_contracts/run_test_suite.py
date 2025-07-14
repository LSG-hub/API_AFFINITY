#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Affinity Analyzer
=====================================================

This script runs the unified analyzer on various API contract combinations
to test its robustness and accuracy across different scenarios:

1. Very Similar APIs (ecommerce vs similar_ecommerce)
2. Completely Different APIs (ecommerce vs weather)
3. Different Domain APIs (social_media vs weather)
4. Complex vs Simple APIs (ecommerce vs minimal)
5. Similar Domain, Different Complexity (social_media vs ecommerce)

Test Categories:
- High Similarity Expected: APIs that should score 70%+
- Low Similarity Expected: APIs that should score 30% or less
- Medium Similarity Expected: APIs that should score 30-70%
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def run_analyzer(api1_path, api2_path):
    """Run the unified analyzer on two API files."""
    try:
        # Ensure we're in the right directory
        base_dir = Path(__file__).parent.parent
        os.chdir(base_dir)
        
        # Run the analyzer with virtual environment
        cmd = [
            "bash", "-c",
            f"source api_similarity_env_py312/bin/activate && python unified_affinity_analyzer.py '{api1_path}' '{api2_path}'"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Analysis timed out"
    except Exception as e:
        return f"Error: {e}"

def extract_similarity_score(output):
    """Extract the overall similarity score from analyzer output."""
    try:
        lines = output.split('\n')
        for line in lines:
            if 'Overall Similarity Score:' in line:
                # Extract percentage
                score_str = line.split(':')[1].strip().replace('%', '')
                return float(score_str)
        return None
    except:
        return None

def extract_component_scores(output):
    """Extract component scores from analyzer output."""
    scores = {}
    try:
        lines = output.split('\n')
        in_breakdown = False
        
        for line in lines:
            if '| Component' in line and 'Score' in line:
                in_breakdown = True
                continue
            elif in_breakdown and '|' in line and 'GNN Functional Similarity' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    scores['gnn'] = float(parts[2].replace('%', ''))
            elif in_breakdown and '|' in line and 'High-Level Semantics' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    scores['semantic'] = float(parts[2].replace('%', ''))
            elif in_breakdown and '|' in line and 'Structural Similarity' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    scores['structural'] = float(parts[2].replace('%', ''))
            elif '- **Path Similarity' in line:
                score_str = line.split(':')[1].strip().replace('%', '')
                scores['path'] = float(score_str)
            elif '- **Deep Schema Similarity' in line:
                score_str = line.split(':')[1].strip().replace('%', '')
                scores['schema'] = float(score_str)
                
        return scores
    except:
        return {}

def run_test_suite():
    """Run comprehensive test suite."""
    print("🧪 Starting Comprehensive API Affinity Analyzer Test Suite")
    print("=" * 70)
    
    # Test cases: (api1, api2, expected_category, description)
    test_cases = [
        # High Similarity Expected (70%+)
        ("test_contracts/ecommerce_api.yaml", 
         "test_contracts/similar_ecommerce_api.yaml", 
         "HIGH", 
         "Very Similar E-commerce APIs (should be 70%+)"),
        
        # Low Similarity Expected (30% or less)
        ("test_contracts/ecommerce_api.yaml", 
         "test_contracts/weather_api.yaml", 
         "LOW", 
         "E-commerce vs Weather APIs (should be 30% or less)"),
        
        ("test_contracts/social_media_api.yaml", 
         "test_contracts/weather_api.yaml", 
         "LOW", 
         "Social Media vs Weather APIs (should be 30% or less)"),
        
        ("test_contracts/ecommerce_api.yaml", 
         "test_contracts/minimal_api.yaml", 
         "LOW", 
         "Complex E-commerce vs Simple Health API (should be 30% or less)"),
        
        # Medium Similarity Expected (30-70%)
        ("test_contracts/social_media_api.yaml", 
         "test_contracts/ecommerce_api.yaml", 
         "MEDIUM", 
         "Social Media vs E-commerce APIs (should be 30-70%)"),
        
        ("test_contracts/weather_api.yaml", 
         "test_contracts/minimal_api.yaml", 
         "MEDIUM", 
         "Weather vs Simple Health API (should be 30-70%)"),
        
        # Edge cases
        ("test_contracts/minimal_api.yaml", 
         "test_contracts/minimal_api.yaml", 
         "IDENTICAL", 
         "Identical APIs (should be ~100%)"),
    ]
    
    results = []
    
    for i, (api1, api2, expected, description) in enumerate(test_cases, 1):
        print(f"\n🔬 Test Case {i}: {description}")
        print("-" * 50)
        print(f"API 1: {api1}")
        print(f"API 2: {api2}")
        print(f"Expected Similarity: {expected}")
        
        # Run analysis
        print("⏳ Running analysis...")
        output = run_analyzer(api1, api2)
        
        # Extract scores
        overall_score = extract_similarity_score(output)
        component_scores = extract_component_scores(output)
        
        if overall_score is not None:
            print(f"📊 Overall Similarity: {overall_score:.2f}%")
            
            # Validate expectations
            if expected == "HIGH" and overall_score >= 70:
                result = "✅ PASS"
            elif expected == "LOW" and overall_score <= 30:
                result = "✅ PASS"
            elif expected == "MEDIUM" and 30 < overall_score < 70:
                result = "✅ PASS"
            elif expected == "IDENTICAL" and overall_score >= 95:
                result = "✅ PASS"
            else:
                result = "❌ UNEXPECTED"
            
            print(f"🎯 Result: {result}")
            
            # Show component breakdown
            if component_scores:
                print("📈 Component Scores:")
                for component, score in component_scores.items():
                    print(f"   - {component.upper()}: {score:.1f}%")
            
        else:
            result = "❌ ERROR"
            print(f"🎯 Result: {result}")
            print("⚠️  Could not extract similarity score")
            print("📄 Raw output:")
            print(output[:500] + "..." if len(output) > 500 else output)
        
        # Store result
        results.append({
            'test_case': i,
            'description': description,
            'api1': api1,
            'api2': api2,
            'expected': expected,
            'overall_score': overall_score,
            'component_scores': component_scores,
            'result': result,
            'output_sample': output[:200] if output else None
        })
        
        print()
    
    # Summary
    print("=" * 70)
    print("📋 TEST SUITE SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if '✅ PASS' in r['result'])
    total = len(results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    
    print("\n📈 DETAILED RESULTS:")
    print("-" * 50)
    
    for result in results:
        status_icon = "✅" if "PASS" in result['result'] else "❌"
        score = f"{result['overall_score']:.1f}%" if result['overall_score'] else "ERROR"
        print(f"{status_icon} Test {result['test_case']}: {score} - {result['description']}")
    
    # Robustness Analysis
    print("\n🔍 ROBUSTNESS ANALYSIS:")
    print("-" * 50)
    
    # Check if analyzer handles different API complexities
    scores_by_complexity = {}
    for result in results:
        if result['overall_score'] is not None:
            if 'minimal' in result['api1'] or 'minimal' in result['api2']:
                complexity = 'simple'
            elif 'ecommerce' in result['api1'] or 'ecommerce' in result['api2']:
                complexity = 'complex'
            else:
                complexity = 'medium'
            
            if complexity not in scores_by_complexity:
                scores_by_complexity[complexity] = []
            scores_by_complexity[complexity].append(result['overall_score'])
    
    for complexity, scores in scores_by_complexity.items():
        avg_score = sum(scores) / len(scores)
        print(f"📊 {complexity.upper()} APIs: Avg similarity {avg_score:.1f}% ({len(scores)} tests)")
    
    # Component Analysis
    print("\n🧩 COMPONENT PERFORMANCE:")
    print("-" * 50)
    
    component_totals = {'gnn': [], 'semantic': [], 'structural': [], 'path': [], 'schema': []}
    
    for result in results:
        if result['component_scores']:
            for comp, score in result['component_scores'].items():
                if comp in component_totals:
                    component_totals[comp].append(score)
    
    for component, scores in component_totals.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"🔧 {component.upper()}: Avg {avg_score:.1f}% (range: {min(scores):.1f}%-{max(scores):.1f}%)")
    
    print("\n" + "=" * 70)
    print("🎉 Test Suite Complete!")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    try:
        results = run_test_suite()
        
        # Save detailed results to file
        import json
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("📁 Detailed results saved to test_results.json")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        sys.exit(1)