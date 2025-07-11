#!/usr/bin/env python3
"""
API Similarity Analyzer v2 - Enhanced Business Context Understanding
===================================================================

An improved version of the zero-cost API similarity analyzer with comprehensive 
business domain vocabularies and enhanced functional similarity detection.

Key Improvements in v2:
1. Extensive industry-standard business domain vocabularies
2. Enhanced semantic operation classification  
3. Business process flow analysis
4. Intent-based operation grouping
5. Field semantic analysis
6. Domain-specific pattern recognition
7. Weighted functional similarity calculation

Uses only free, open-source libraries - completely zero cost.
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

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class EnhancedDomainClassifier:
    """Enhanced domain classification with comprehensive industry vocabularies."""
    
    def __init__(self):
        # Comprehensive business domain vocabularies based on industry standards
        self.business_domains = {
            'banking_financial': {
                'keywords': [
                    # Core Banking
                    'account', 'balance', 'transaction', 'payment', 'transfer', 'deposit', 
                    'withdrawal', 'loan', 'credit', 'debit', 'bank', 'financial', 'currency',
                    'amount', 'fund', 'asset', 'liability', 'equity', 'portfolio',
                    
                    # PSD2 / Open Banking
                    'consent', 'aisp', 'pisp', 'cbpii', 'psd2', 'open-banking', 'tpp',
                    'strong-customer-auth', 'sca', 'authorization', 'fapi',
                    
                    # Payment Processing  
                    'acquirer', 'issuer', 'merchant', 'cardholder', 'settlement', 'clearing',
                    'interchange', 'chargeback', 'refund', 'void', 'capture', 'authorize',
                    
                    # Investment & Trading
                    'securities', 'stocks', 'bonds', 'derivatives', 'forex', 'trading',
                    'order', 'execution', 'position', 'risk', 'compliance', 'kyc'
                ],
                'operations': [
                    'consent-setup', 'consent-retrieval', 'consent-deletion', 'account-status',
                    'balance-inquiry', 'transaction-history', 'payment-initiate', 'transfer-funds',
                    'authorize-payment', 'settle-transaction', 'verify-account'
                ],
                'paths': [
                    '/aisp/', '/pisp/', '/cbpii/', '/accounts/', '/transactions/', '/balances/',
                    '/payments/', '/consents/', '/transfers/', '/cards/', '/loans/'
                ]
            },
            
            'ecommerce_retail': {
                'keywords': [
                    # Core eCommerce
                    'product', 'catalog', 'inventory', 'stock', 'sku', 'variant', 'price',
                    'discount', 'promotion', 'coupon', 'cart', 'checkout', 'order', 'purchase',
                    'customer', 'shipping', 'delivery', 'fulfillment', 'warehouse',
                    
                    # Order Management
                    'order-status', 'tracking', 'shipment', 'invoice', 'receipt', 'return',
                    'refund', 'exchange', 'cancellation', 'backorder', 'preorder',
                    
                    # Customer Experience
                    'wishlist', 'favorites', 'recommendation', 'review', 'rating', 'feedback',
                    'loyalty', 'rewards', 'points', 'tier', 'membership',
                    
                    # Marketplace
                    'vendor', 'seller', 'merchant', 'commission', 'marketplace', 'listing',
                    'category', 'subcategory', 'brand', 'manufacturer'
                ],
                'operations': [
                    'product-list', 'product-create', 'product-update', 'inventory-check',
                    'cart-add', 'cart-update', 'checkout-process', 'order-create', 'order-status',
                    'shipping-calculate', 'payment-process', 'return-initiate'
                ],
                'paths': [
                    '/products/', '/catalog/', '/inventory/', '/cart/', '/checkout/', '/orders/',
                    '/customers/', '/shipping/', '/payments/', '/reviews/', '/categories/'
                ]
            },
            
            'healthcare_medical': {
                'keywords': [
                    # FHIR Resources
                    'patient', 'practitioner', 'encounter', 'observation', 'medication',
                    'immunization', 'diagnosis', 'condition', 'procedure', 'appointment',
                    'organization', 'location', 'device', 'specimen', 'diagnostic-report',
                    
                    # Clinical Terms
                    'clinical', 'medical', 'health', 'healthcare', 'fhir', 'hl7', 'emr', 'ehr',
                    'hipaa', 'phi', 'treatment', 'therapy', 'prescription', 'dosage',
                    'allergy', 'vital-signs', 'symptoms', 'laboratory', 'radiology',
                    
                    # Terminology Standards
                    'loinc', 'snomed', 'icd', 'cpt', 'rxnorm', 'terminology', 'coding',
                    'value-set', 'code-system', 'concept-map'
                ],
                'operations': [
                    'patient-search', 'patient-create', 'patient-update', 'encounter-create',
                    'observation-record', 'medication-dispense', 'appointment-schedule',
                    'diagnosis-record', 'procedure-perform', 'result-publish'
                ],
                'paths': [
                    '/fhir/', '/patients/', '/practitioners/', '/encounters/', '/observations/',
                    '/medications/', '/appointments/', '/conditions/', '/procedures/'
                ]
            },
            
            'logistics_supply_chain': {
                'keywords': [
                    # Shipping & Logistics
                    'shipment', 'tracking', 'delivery', 'pickup', 'carrier', 'freight',
                    'logistics', 'transport', 'route', 'dispatch', 'manifest', 'bill-of-lading',
                    'warehouse', 'fulfillment', 'distribution', 'consolidation',
                    
                    # Supply Chain
                    'supplier', 'vendor', 'procurement', 'sourcing', 'purchasing', 'inbound',
                    'outbound', 'inventory', 'stock', 'demand', 'forecast', 'planning',
                    'replenishment', 'allocation', 'backorder',
                    
                    # Delivery & Tracking
                    'package', 'parcel', 'container', 'pallet', 'label', 'barcode', 'scan',
                    'status', 'location', 'gps', 'geofence', 'eta', 'proof-of-delivery'
                ],
                'operations': [
                    'shipment-create', 'shipment-track', 'delivery-schedule', 'route-optimize',
                    'inventory-update', 'stock-check', 'warehouse-receive', 'order-fulfill',
                    'carrier-assign', 'label-generate'
                ],
                'paths': [
                    '/shipments/', '/tracking/', '/deliveries/', '/warehouses/', '/inventory/',
                    '/carriers/', '/routes/', '/manifests/', '/labels/', '/packages/'
                ]
            },
            
            'user_management_auth': {
                'keywords': [
                    # Authentication
                    'auth', 'authentication', 'authorize', 'login', 'logout', 'signin', 'signout',
                    'token', 'jwt', 'oauth', 'oauth2', 'openid', 'saml', 'sso', 'mfa',
                    'password', 'credential', 'session', 'refresh', 'verify', 'validate',
                    
                    # User Management
                    'user', 'profile', 'account', 'customer', 'member', 'registration', 'signup',
                    'identity', 'persona', 'role', 'permission', 'access', 'privilege',
                    'group', 'team', 'organization', 'tenant', 'domain',
                    
                    # Security
                    'security', 'encryption', 'hash', 'salt', 'cipher', 'certificate',
                    'public-key', 'private-key', 'signature', 'audit', 'compliance'
                ],
                'operations': [
                    'user-register', 'user-login', 'user-logout', 'token-generate', 'token-refresh',
                    'password-reset', 'profile-update', 'role-assign', 'permission-grant',
                    'session-validate', 'auth-verify'
                ],
                'paths': [
                    '/auth/', '/users/', '/login/', '/logout/', '/register/', '/profiles/',
                    '/tokens/', '/sessions/', '/roles/', '/permissions/', '/oauth/'
                ]
            },
            
            'data_management_crud': {
                'keywords': [
                    # Database Operations
                    'data', 'database', 'table', 'record', 'row', 'column', 'field', 'schema',
                    'query', 'filter', 'sort', 'search', 'index', 'primary-key', 'foreign-key',
                    'relation', 'join', 'aggregate', 'group', 'count', 'sum', 'average',
                    
                    # CRUD Operations
                    'create', 'read', 'update', 'delete', 'insert', 'select', 'modify',
                    'remove', 'add', 'edit', 'save', 'retrieve', 'fetch', 'list',
                    
                    # Data Processing
                    'import', 'export', 'sync', 'replicate', 'backup', 'restore', 'migrate',
                    'transform', 'validate', 'cleanse', 'normalize', 'format'
                ],
                'operations': [
                    'record-create', 'record-list', 'record-update', 'record-delete',
                    'data-import', 'data-export', 'query-execute', 'search-perform',
                    'table-create', 'schema-update'
                ],
                'paths': [
                    '/data/', '/records/', '/tables/', '/queries/', '/search/', '/import/',
                    '/export/', '/sync/', '/api/v2/tables/', '/database/'
                ]
            },
            
            'content_media_management': {
                'keywords': [
                    # Content Types
                    'content', 'document', 'file', 'media', 'image', 'video', 'audio',
                    'text', 'article', 'post', 'page', 'blog', 'news', 'story',
                    'attachment', 'asset', 'resource', 'library', 'gallery',
                    
                    # Content Operations
                    'upload', 'download', 'stream', 'transcode', 'compress', 'resize',
                    'crop', 'filter', 'edit', 'publish', 'draft', 'review', 'approve',
                    'archive', 'version', 'revision', 'metadata', 'tag', 'category',
                    
                    # Digital Asset Management
                    'dam', 'cms', 'repository', 'storage', 'cdn', 'cache', 'thumbnail',
                    'preview', 'watermark', 'license', 'copyright', 'usage-rights'
                ],
                'operations': [
                    'content-upload', 'content-download', 'media-process', 'file-convert',
                    'image-resize', 'video-transcode', 'content-publish', 'asset-organize',
                    'metadata-extract', 'thumbnail-generate'
                ],
                'paths': [
                    '/content/', '/media/', '/files/', '/documents/', '/images/', '/videos/',
                    '/upload/', '/download/', '/assets/', '/cms/', '/dam/'
                ]
            },
            
            'communication_notification': {
                'keywords': [
                    # Communication Channels
                    'notification', 'message', 'email', 'sms', 'push', 'webhook', 'alert',
                    'reminder', 'broadcast', 'campaign', 'newsletter', 'announcement',
                    'chat', 'messaging', 'conversation', 'thread', 'channel',
                    
                    # Communication Management
                    'template', 'personalization', 'segmentation', 'targeting', 'delivery',
                    'bounce', 'unsubscribe', 'opt-in', 'opt-out', 'preference',
                    'schedule', 'queue', 'batch', 'bulk', 'automation',
                    
                    # Event-Driven
                    'event', 'trigger', 'subscription', 'publisher', 'subscriber',
                    'topic', 'feed', 'stream', 'real-time', 'live'
                ],
                'operations': [
                    'message-send', 'notification-create', 'email-send', 'sms-send',
                    'push-notify', 'webhook-trigger', 'subscription-manage', 'template-create',
                    'campaign-launch', 'event-publish'
                ],
                'paths': [
                    '/notifications/', '/messages/', '/email/', '/sms/', '/push/', '/webhooks/',
                    '/alerts/', '/events/', '/campaigns/', '/subscriptions/'
                ]
            },
            
            'kyc_compliance_onboarding': {
                'keywords': [
                    # KYC & Compliance
                    'kyc', 'know-your-customer', 'compliance', 'verification', 'identity',
                    'document', 'validation', 'screening', 'aml', 'anti-money-laundering',
                    'sanctions', 'pep', 'politically-exposed-person', 'risk-assessment',
                    'due-diligence', 'background-check', 'watchlist',
                    
                    # Onboarding Process
                    'onboarding', 'registration', 'application', 'form', 'questionnaire',
                    'step', 'workflow', 'process', 'status', 'approval', 'rejection',
                    'pending', 'review', 'manual-review', 'auto-approve',
                    
                    # Lead Management
                    'lead', 'prospect', 'applicant', 'candidate', 'contact', 'inquiry',
                    'interest', 'qualification', 'scoring', 'nurturing', 'conversion',
                    'business-name', 'business-email', 'phone-number', 'country-code'
                ],
                'operations': [
                    'lead-create', 'lead-update', 'identity-verify', 'document-upload',
                    'compliance-check', 'risk-assess', 'kyc-complete', 'onboard-customer',
                    'application-submit', 'status-check'
                ],
                'paths': [
                    '/kyc/', '/compliance/', '/verification/', '/onboarding/', '/leads/',
                    '/applications/', '/documents/', '/identity/', '/screening/'
                ]
            }
        }
        
        # Operation intent patterns
        self.operation_intents = {
            'data_access': [
                'get', 'list', 'retrieve', 'fetch', 'read', 'view', 'show', 'find',
                'search', 'query', 'lookup', 'browse', 'explore'
            ],
            'data_modification': [
                'create', 'post', 'add', 'insert', 'new', 'register', 'submit',
                'update', 'put', 'patch', 'edit', 'modify', 'change', 'set',
                'delete', 'remove', 'destroy', 'cancel', 'revoke', 'void'
            ],
            'business_process': [
                'process', 'execute', 'perform', 'run', 'initiate', 'trigger',
                'approve', 'reject', 'confirm', 'verify', 'validate', 'check',
                'authorize', 'authenticate', 'sign', 'complete', 'finish'
            ],
            'system_operation': [
                'setup', 'configure', 'install', 'deploy', 'start', 'stop',
                'restart', 'reset', 'sync', 'backup', 'restore', 'migrate',
                'export', 'import', 'upload', 'download', 'send', 'receive'
            ]
        }
        
        # Business process flows
        self.business_flows = {
            'financial_transaction': [
                'consent', 'authorize', 'initiate', 'validate', 'process', 'settle', 'confirm'
            ],
            'ecommerce_purchase': [
                'browse', 'select', 'cart', 'checkout', 'payment', 'fulfill', 'deliver'
            ],
            'user_onboarding': [
                'register', 'verify', 'approve', 'provision', 'activate', 'welcome'
            ],
            'content_workflow': [
                'create', 'edit', 'review', 'approve', 'publish', 'distribute', 'archive'
            ],
            'data_lifecycle': [
                'collect', 'validate', 'store', 'process', 'analyze', 'report', 'archive'
            ],
            'compliance_process': [
                'screen', 'verify', 'assess', 'review', 'approve', 'monitor', 'report'
            ]
        }

class EnhancedStructuralAnalyzer:
    """Enhanced structural analysis with business context."""
    
    def __init__(self, domain_classifier):
        self.domain_classifier = domain_classifier
    
    def analyze_endpoint_patterns(self, paths):
        """Analyze endpoint patterns with business context."""
        patterns = {
            'resource_types': set(),
            'operation_patterns': set(),
            'path_depth': [],
            'parameter_patterns': set(),
            'business_entities': set()
        }
        
        for path, path_info in paths.items():
            # Extract resource types
            path_segments = [seg for seg in path.split('/') if seg and not seg.startswith('{')]
            patterns['resource_types'].update(path_segments)
            patterns['path_depth'].append(len(path_segments))
            
            # Extract business entities based on domain vocabularies
            for domain, vocab in self.domain_classifier.business_domains.items():
                for keyword in vocab['keywords']:
                    if keyword in path.lower():
                        patterns['business_entities'].add(keyword)
            
            # Analyze operations
            for method, operation in path_info.get('operations', {}).items():
                op_id = operation.get('operationId', '').lower()
                patterns['operation_patterns'].add(f"{method}:{op_id}")
                
                # Extract parameter patterns
                for param in operation.get('parameters', []):
                    if isinstance(param, dict):
                        param_name = param.get('name', '')
                        param_location = param.get('in', '')
                        patterns['parameter_patterns'].add(f"{param_location}:{param_name}")
        
        return patterns
    
    def calculate_pattern_similarity(self, patterns1, patterns2):
        """Calculate similarity between endpoint patterns."""
        similarities = {}
        
        # Resource type similarity
        res1, res2 = patterns1['resource_types'], patterns2['resource_types']
        similarities['resources'] = self._jaccard_similarity(res1, res2)
        
        # Business entity similarity
        ent1, ent2 = patterns1['business_entities'], patterns2['business_entities']
        similarities['entities'] = self._jaccard_similarity(ent1, ent2)
        
        # Operation pattern similarity
        op1, op2 = patterns1['operation_patterns'], patterns2['operation_patterns']
        similarities['operations'] = self._jaccard_similarity(op1, op2)
        
        # Parameter pattern similarity
        param1, param2 = patterns1['parameter_patterns'], patterns2['parameter_patterns']
        similarities['parameters'] = self._jaccard_similarity(param1, param2)
        
        # Path complexity similarity
        depth1 = np.mean(patterns1['path_depth']) if patterns1['path_depth'] else 0
        depth2 = np.mean(patterns2['path_depth']) if patterns2['path_depth'] else 0
        depth_diff = abs(depth1 - depth2)
        similarities['complexity'] = max(0, 1 - (depth_diff / max(depth1, depth2, 1)))
        
        return similarities
    
    def _jaccard_similarity(self, set1, set2):
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

class EnhancedFunctionalAnalyzer:
    """Enhanced functional analysis with business context understanding."""
    
    def __init__(self, domain_classifier):
        self.domain_classifier = domain_classifier
    
    def analyze_business_domain(self, spec, paths, schemas):
        """Comprehensive domain analysis."""
        domain_scores = defaultdict(float)
        
        # Analyze paths for domain keywords
        for path, path_info in paths.items():
            path_text = path.lower()
            for method, operation in path_info.get('operations', {}).items():
                op_text = f"{operation.get('operationId', '')} {operation.get('summary', '')} {operation.get('description', '')}".lower()
                
                for domain, vocab in self.domain_classifier.business_domains.items():
                    # Score based on keyword frequency
                    for keyword in vocab['keywords']:
                        domain_scores[domain] += path_text.count(keyword) * 2
                        domain_scores[domain] += op_text.count(keyword)
                    
                    # Score based on operation patterns
                    for op_pattern in vocab['operations']:
                        if op_pattern in op_text:
                            domain_scores[domain] += 3
                    
                    # Score based on path patterns
                    for path_pattern in vocab['paths']:
                        if path_pattern in path_text:
                            domain_scores[domain] += 2
        
        # Analyze schemas for domain-specific fields
        for schema_name, schema_def in schemas.items():
            schema_text = json.dumps(schema_def).lower()
            for domain, vocab in self.domain_classifier.business_domains.items():
                for keyword in vocab['keywords']:
                    domain_scores[domain] += schema_text.count(keyword)
        
        # Normalize scores
        if domain_scores:
            max_score = max(domain_scores.values())
            if max_score > 0:
                for domain in domain_scores:
                    domain_scores[domain] /= max_score
        
        return dict(domain_scores)
    
    def analyze_operation_intents(self, paths):
        """Analyze operation intents."""
        intent_counts = defaultdict(int)
        
        for path_info in paths.values():
            for method, operation in path_info.get('operations', {}).items():
                op_id = operation.get('operationId', '').lower()
                op_desc = f"{operation.get('summary', '')} {operation.get('description', '')}".lower()
                
                for intent, keywords in self.domain_classifier.operation_intents.items():
                    for keyword in keywords:
                        if keyword in op_id or keyword in op_desc:
                            intent_counts[intent] += 1
                            break
        
        return dict(intent_counts)
    
    def analyze_business_flows(self, paths):
        """Analyze business process flows."""
        flow_matches = defaultdict(int)
        operations = []
        
        # Extract all operations
        for path_info in paths.values():
            for method, operation in path_info.get('operations', {}).items():
                op_id = operation.get('operationId', '').lower()
                operations.append(op_id)
        
        # Check for flow patterns
        for flow_name, flow_steps in self.domain_classifier.business_flows.items():
            matches = 0
            for step in flow_steps:
                for op in operations:
                    if step in op:
                        matches += 1
                        break
            flow_matches[flow_name] = matches / len(flow_steps) if flow_steps else 0
        
        return dict(flow_matches)
    
    def calculate_enhanced_functional_similarity(self, analysis1, analysis2):
        """Calculate enhanced functional similarity with business context."""
        
        # 1. Domain Similarity (40% weight)
        domain_sim = self._calculate_domain_similarity(
            analysis1['domains'], analysis2['domains']
        )
        
        # 2. Intent Similarity (30% weight)
        intent_sim = self._calculate_intent_similarity(
            analysis1['intents'], analysis2['intents']
        )
        
        # 3. Business Flow Similarity (20% weight)
        flow_sim = self._calculate_flow_similarity(
            analysis1['flows'], analysis2['flows']
        )
        
        # 4. CRUD Pattern Similarity (10% weight - reduced from original)
        crud_sim = self._calculate_crud_similarity(
            analysis1['intents'], analysis2['intents']
        )
        
        return {
            'domain': domain_sim,
            'intent': intent_sim,
            'flow': flow_sim,
            'crud': crud_sim,
            'composite': domain_sim * 0.4 + intent_sim * 0.3 + flow_sim * 0.2 + crud_sim * 0.1
        }
    
    def _calculate_domain_similarity(self, domains1, domains2):
        """Calculate domain similarity with penalties for different domains."""
        if not domains1 or not domains2:
            return 0.0
        
        # Get primary domains (highest scores)
        primary1 = max(domains1, key=domains1.get) if domains1 else None
        primary2 = max(domains2, key=domains2.get) if domains2 else None
        
        if primary1 == primary2:
            return 1.0
        
        # Check for related domains
        related_pairs = [
            ('banking_financial', 'kyc_compliance_onboarding'),
            ('ecommerce_retail', 'logistics_supply_chain'),
            ('user_management_auth', 'kyc_compliance_onboarding'),
            ('content_media_management', 'data_management_crud'),
            ('communication_notification', 'user_management_auth')
        ]
        
        for pair in related_pairs:
            if (primary1, primary2) in [pair, pair[::-1]]:
                return 0.6
        
        # Calculate overlap similarity
        common_domains = set(domains1.keys()).intersection(set(domains2.keys()))
        if common_domains:
            overlap_score = sum(min(domains1[d], domains2[d]) for d in common_domains)
            max_possible = max(sum(domains1.values()), sum(domains2.values()))
            return overlap_score / max_possible if max_possible > 0 else 0.0
        
        return 0.0
    
    def _calculate_intent_similarity(self, intents1, intents2):
        """Calculate operation intent similarity."""
        if not intents1 and not intents2:
            return 1.0
        if not intents1 or not intents2:
            return 0.0
        
        # Normalize intent counts
        total1 = sum(intents1.values())
        total2 = sum(intents2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        norm1 = {k: v/total1 for k, v in intents1.items()}
        norm2 = {k: v/total2 for k, v in intents2.items()}
        
        # Calculate intent distribution similarity
        all_intents = set(norm1.keys()).union(set(norm2.keys()))
        similarity = 0.0
        
        for intent in all_intents:
            val1 = norm1.get(intent, 0)
            val2 = norm2.get(intent, 0)
            similarity += 1 - abs(val1 - val2)
        
        return similarity / len(all_intents) if all_intents else 0.0
    
    def _calculate_flow_similarity(self, flows1, flows2):
        """Calculate business flow similarity."""
        if not flows1 and not flows2:
            return 1.0
        if not flows1 or not flows2:
            return 0.0
        
        # Find best matching flows
        max_similarity = 0.0
        for flow1, score1 in flows1.items():
            for flow2, score2 in flows2.items():
                if flow1 == flow2:
                    # Same flow type - compare completeness
                    flow_sim = 1 - abs(score1 - score2)
                    max_similarity = max(max_similarity, flow_sim)
        
        return max_similarity
    
    def _calculate_crud_similarity(self, intents1, intents2):
        """Calculate CRUD pattern similarity (reduced weight)."""
        crud_patterns1 = {
            'create': intents1.get('data_modification', 0),
            'read': intents1.get('data_access', 0),
            'update': intents1.get('data_modification', 0),
            'delete': intents1.get('data_modification', 0)
        }
        
        crud_patterns2 = {
            'create': intents2.get('data_modification', 0),
            'read': intents2.get('data_access', 0),
            'update': intents2.get('data_modification', 0),
            'delete': intents2.get('data_modification', 0)
        }
        
        total1 = sum(crud_patterns1.values())
        total2 = sum(crud_patterns2.values())
        
        if total1 == 0 and total2 == 0:
            return 1.0
        if total1 == 0 or total2 == 0:
            return 0.0
        
        # Normalize and compare
        norm1 = {k: v/total1 for k, v in crud_patterns1.items()}
        norm2 = {k: v/total2 for k, v in crud_patterns2.items()}
        
        similarity = 0.0
        for operation in crud_patterns1.keys():
            similarity += 1 - abs(norm1[operation] - norm2[operation])
        
        return similarity / len(crud_patterns1)

# Import and extend the original classes
from api_similarity_analyzer import (
    APIStructureExtractor, SemanticSimilarityAnalyzer, 
    SchemaSimilarityAnalyzer, format_similarity_report
)

class EnhancedAPISimilarityAnalyzer:
    """Enhanced API similarity analyzer with improved business context understanding."""
    
    def __init__(self):
        self.extractor = APIStructureExtractor()
        self.domain_classifier = EnhancedDomainClassifier()
        self.structural_analyzer = EnhancedStructuralAnalyzer(self.domain_classifier)
        self.functional_analyzer = EnhancedFunctionalAnalyzer(self.domain_classifier)
        self.semantic_analyzer = SemanticSimilarityAnalyzer()
        self.schema_analyzer = SchemaSimilarityAnalyzer()
        
        # Adjusted weights for enhanced analysis
        self.weights = {
            'structural': 0.25,
            'semantic': 0.20,
            'schema': 0.20,
            'functional': 0.35  # Increased weight for enhanced functional analysis
        }
    
    def analyze_similarity(self, api1_path, api2_path):
        """Perform enhanced similarity analysis."""
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
        
        # Enhanced analysis
        structural_analysis = self._analyze_enhanced_structural(paths1, paths2)
        functional_analysis = self._analyze_enhanced_functional(spec1, spec2, paths1, paths2, schemas1, schemas2)
        semantic_score = self._calculate_semantic_similarity(spec1, spec2, text1, text2)
        schema_score = self.schema_analyzer.calculate_schema_similarity(schemas1, schemas2)
        
        # Calculate weighted final score
        final_score = (
            structural_analysis['composite'] * self.weights['structural'] +
            semantic_score * self.weights['semantic'] +
            schema_score * self.weights['schema'] +
            functional_analysis['composite'] * self.weights['functional']
        ) * 100  # Convert to percentage
        
        return {
            'final_score': final_score,
            'detailed_scores': {
                'structural': structural_analysis['composite'] * 100,
                'semantic': semantic_score * 100,
                'schema': schema_score * 100,
                'functional': functional_analysis['composite'] * 100
            },
            'enhanced_breakdown': {
                'structural_components': {k: v * 100 for k, v in structural_analysis.items() if k != 'composite'},
                'functional_components': {k: v * 100 for k, v in functional_analysis.items() if k != 'composite'}
            },
            'metadata': {
                'api1': metadata1,
                'api2': metadata2
            },
            'analysis': self._generate_enhanced_analysis_report(
                final_score, structural_analysis, functional_analysis, 
                semantic_score, schema_score, metadata1, metadata2
            )
        }
    
    def _analyze_enhanced_structural(self, paths1, paths2):
        """Enhanced structural analysis."""
        patterns1 = self.structural_analyzer.analyze_endpoint_patterns(paths1)
        patterns2 = self.structural_analyzer.analyze_endpoint_patterns(paths2)
        
        similarities = self.structural_analyzer.calculate_pattern_similarity(patterns1, patterns2)
        
        # Calculate composite structural score
        composite = np.mean(list(similarities.values()))
        similarities['composite'] = composite
        
        return similarities
    
    def _analyze_enhanced_functional(self, spec1, spec2, paths1, paths2, schemas1, schemas2):
        """Enhanced functional analysis with business context."""
        # Analyze API 1
        analysis1 = {
            'domains': self.functional_analyzer.analyze_business_domain(spec1, paths1, schemas1),
            'intents': self.functional_analyzer.analyze_operation_intents(paths1),
            'flows': self.functional_analyzer.analyze_business_flows(paths1)
        }
        
        # Analyze API 2
        analysis2 = {
            'domains': self.functional_analyzer.analyze_business_domain(spec2, paths2, schemas2),
            'intents': self.functional_analyzer.analyze_operation_intents(paths2),
            'flows': self.functional_analyzer.analyze_business_flows(paths2)
        }
        
        # Calculate enhanced functional similarity
        return self.functional_analyzer.calculate_enhanced_functional_similarity(analysis1, analysis2)
    
    def _calculate_semantic_similarity(self, spec1, spec2, text1, text2):
        """Calculate semantic similarity using original method."""
        tfidf_sim = self.semantic_analyzer.calculate_tfidf_similarity(text1, text2)
        domain_sim = self.semantic_analyzer.calculate_domain_similarity(spec1, spec2)
        return (tfidf_sim + domain_sim) / 2
    
    def _generate_enhanced_analysis_report(self, final_score, structural, functional, semantic, schema, metadata1, metadata2):
        """Generate enhanced analysis report."""
        category = self._categorize_similarity(final_score)
        
        # Determine primary domains
        primary_domain1 = "Unknown"
        primary_domain2 = "Unknown"
        
        if 'domain' in functional and isinstance(functional['domain'], dict):
            # For detailed functional analysis
            if hasattr(self.functional_analyzer, 'analyze_business_domain'):
                # This is a placeholder - would need actual domain analysis results
                primary_domain1 = "Multiple domains detected"
                primary_domain2 = "Multiple domains detected"
        
        report = {
            'similarity_category': category,
            'recommendation': self._get_recommendation(final_score),
            'consolidation_potential': self._assess_consolidation_potential(final_score),
            'domain_analysis': {
                'api1_domain': primary_domain1,
                'api2_domain': primary_domain2,
                'domain_similarity': functional.get('domain', 0) * 100
            },
            'detailed_functional_analysis': {
                'domain_similarity': functional.get('domain', 0) * 100,
                'intent_similarity': functional.get('intent', 0) * 100,
                'flow_similarity': functional.get('flow', 0) * 100,
                'crud_similarity': functional.get('crud', 0) * 100
            },
            'structural_breakdown': structural,
            'improvement_notes': [
                f"Enhanced functional analysis now considers business domain context",
                f"Domain similarity: {functional.get('domain', 0) * 100:.1f}%",
                f"Operation intent similarity: {functional.get('intent', 0) * 100:.1f}%",
                f"Business flow similarity: {functional.get('flow', 0) * 100:.1f}%"
            ]
        }
        
        return report
    
    def _categorize_similarity(self, score):
        """Categorize similarity score."""
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

def format_enhanced_similarity_report(result, api1_name, api2_name):
    """Format the enhanced similarity analysis result."""
    if not result:
        return "Error: Could not analyze API specifications."
    
    final_score = result['final_score']
    detailed = result['detailed_scores']
    enhanced = result['enhanced_breakdown']
    metadata = result['metadata']
    analysis = result['analysis']
    
    report = f"""
# Enhanced API Similarity Analysis Report (v2)

## APIs Compared
- **Source API**: {metadata['api1']['title']} (v{metadata['api1']['version']})
- **Target API**: {metadata['api2']['title']} (v{metadata['api2']['version']})

## Similarity Score: {final_score:.1f}%

### Category: {analysis['similarity_category']}
**Recommendation**: {analysis['recommendation']}

## Enhanced Analysis Breakdown

### Primary Similarity Components
- **Structural Similarity**: {detailed['structural']:.1f}%
- **Semantic Similarity**: {detailed['semantic']:.1f}%  
- **Schema Similarity**: {detailed['schema']:.1f}%
- **Enhanced Functional Similarity**: {detailed['functional']:.1f}%

### Enhanced Functional Analysis Details
- **Domain Similarity**: {analysis['detailed_functional_analysis']['domain_similarity']:.1f}%
  - Business domain alignment and vocabulary overlap
- **Operation Intent Similarity**: {analysis['detailed_functional_analysis']['intent_similarity']:.1f}%
  - CRUD vs. business process operations
- **Business Flow Similarity**: {analysis['detailed_functional_analysis']['flow_similarity']:.1f}%
  - Sequential process patterns (e.g., consent→access→transaction)
- **CRUD Pattern Similarity**: {analysis['detailed_functional_analysis']['crud_similarity']:.1f}%
  - Basic create/read/update/delete operations

### Structural Analysis Breakdown
"""
    
    if 'structural_components' in enhanced:
        for component, score in enhanced['structural_components'].items():
            report += f"- **{component.title()} Similarity**: {score:.1f}%\n"
    
    report += f"""
### Domain Analysis
- **API 1 Domain**: {analysis['domain_analysis']['api1_domain']}
- **API 2 Domain**: {analysis['domain_analysis']['api2_domain']}
- **Cross-Domain Similarity**: {analysis['domain_analysis']['domain_similarity']:.1f}%

### Consolidation Assessment
- **Potential**: {analysis['consolidation_potential']}
- **Risk Level**: {"Low" if final_score >= 70 else "Medium" if final_score >= 50 else "High"}

## Key Improvements in v2
"""
    
    for note in analysis.get('improvement_notes', []):
        report += f"- {note}\n"
    
    report += f"""

## Summary
Based on the enhanced multi-dimensional analysis with comprehensive business context understanding,
these APIs show **{analysis['similarity_category'].lower()}** with a composite score of **{final_score:.1f}%**.

The enhanced functional analysis (weighted at 35%) now incorporates:
- Industry-standard business domain vocabularies
- Operation intent classification  
- Business process flow recognition
- Domain-specific pattern matching

**{analysis['recommendation']}**.

---
*Analysis performed using enhanced zero-cost, open-source similarity detection framework v2*
*Enhanced with comprehensive business domain understanding and improved functional analysis*
"""
    
    return report

def main():
    """Main function to run the enhanced API similarity analysis."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python api_similarity_analyzer_v2.py <api1_path> <api2_path>")
        sys.exit(1)
    
    api1_path = sys.argv[1]
    api2_path = sys.argv[2]
    
    analyzer = EnhancedAPISimilarityAnalyzer()
    result = analyzer.analyze_similarity(api1_path, api2_path)
    
    if result:
        report = format_enhanced_similarity_report(result, api1_path, api2_path)
        print(report)
        
        # Save report to file
        output_file = "api_similarity_report_v2.md"
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nEnhanced detailed report saved to: {output_file}")
    else:
        print("Error: Could not analyze the provided API specifications.")

if __name__ == "__main__":
    main()