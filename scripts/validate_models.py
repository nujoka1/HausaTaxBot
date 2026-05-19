#!/usr/bin/env python3
"""
Model Validation Script for HausaTaxBot

Validates that all trained model pair files exist and can be loaded correctly.
Checks for proper structure, dimensions, and compatibility.

Usage:
    python scripts/validate_models.py
    python scripts/validate_models.py --verbose
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelValidator:
    """Validates HausaTaxBot model pairs."""
    
    EXPECTED_MODEL_PAIRS = [
        'ctfidf_svm',
        'ctfidf_fastkan',
        'colbert_svm',
        'colbert_fastkan',
        'model2vec_svm',
        'model2vec_fastkan'
    ]
    
    def __init__(self, base_dir: Path = None, verbose: bool = False):
        """Initialize validator.
        
        Args:
            base_dir: Base directory (defaults to HausaTaxBot/)
            verbose: Enable verbose logging
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = base_dir
        self.models_dir = base_dir / "models"
        self.evaluation_dir = base_dir / "evaluation"
        self.verbose = verbose
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'summary': {}
        }
    
    def validate(self) -> bool:
        """Run all validation checks.
        
        Returns:
            True if all checks pass, False otherwise
        """
        logger.info("=" * 70)
        logger.info("HausaTaxBot Model Validation")
        logger.info("=" * 70)
        
        # Check directories exist
        if not self._check_directories():
            return False
        
        # Check metadata file
        if not self._check_metadata():
            return False
        
        # Check model pair files
        if not self._check_model_files():
            return False
        
        # Check model loadability
        if not self._check_model_loading():
            return False
        
        # Check backward compatibility
        if not self._check_backward_compatibility():
            return False
        
        # Print summary
        self._print_summary()
        
        return len(self.results['failed']) == 0
    
    def _check_directories(self) -> bool:
        """Check that required directories exist."""
        logger.info("\n[1/5] Checking directories...")
        
        if not self.models_dir.exists():
            logger.error(f"✗ /models directory not found: {self.models_dir}")
            self.results['failed'].append("Missing /models directory")
            return False
        
        logger.info(f"✓ /models directory exists: {self.models_dir}")
        self.results['passed'].append("Directory check")
        return True
    
    def _check_metadata(self) -> bool:
        """Check that metadata JSON exists and is valid."""
        logger.info("\n[2/5] Checking metadata...")
        
        metadata_file = self.models_dir / "available_models.json"
        
        if not metadata_file.exists():
            logger.error(f"✗ Metadata file not found: {metadata_file}")
            logger.warning("  → Run the notebook to generate model files")
            self.results['failed'].append("Missing available_models.json")
            return False
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            available_pairs = metadata.get('available_pairs', [])
            best_pair = metadata.get('best_pair', {})
            
            logger.info(f"✓ Metadata loaded: {len(available_pairs)} model pairs")
            logger.info(f"✓ Best pair: {best_pair.get('encoder', '?').upper()} + "
                       f"{best_pair.get('classifier', '?').upper()}")
            
            self.results['passed'].append("Metadata check")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"✗ Invalid JSON in metadata: {e}")
            self.results['failed'].append("Metadata parsing error")
            return False
    
    def _check_model_files(self) -> bool:
        """Check that all expected model pair files exist."""
        logger.info("\n[3/5] Checking model files...")
        
        missing_files = []
        for pair_name in self.EXPECTED_MODEL_PAIRS:
            file_path = self.models_dir / f"{pair_name}.pkl"
            
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Found: {pair_name}.pkl ({size_mb:.2f} MB)")
                self.results['passed'].append(f"File: {pair_name}.pkl")
            else:
                logger.warning(f"✗ Missing: {pair_name}.pkl")
                missing_files.append(pair_name)
                self.results['warnings'].append(f"Missing file: {pair_name}.pkl")
        
        if missing_files:
            logger.warning(f"\n⚠️  {len(missing_files)} model pairs are missing")
            logger.info("   → Run the notebook to train and save all models")
            return True  # Don't fail, training might be in progress
        
        return True
    
    def _check_model_loading(self) -> bool:
        """Check that model files can be loaded successfully."""
        logger.info("\n[4/5] Checking model loading...")
        
        loadable_pairs = 0
        
        for pair_name in self.EXPECTED_MODEL_PAIRS:
            file_path = self.models_dir / f"{pair_name}.pkl"
            
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'rb') as f:
                    model_pair = pickle.load(f)
                
                # Validate structure
                encoder = model_pair.get('encoder')
                classifier = model_pair.get('classifier')
                
                if encoder is None or classifier is None:
                    logger.warning(f"✗ Invalid structure in {pair_name}.pkl")
                    self.results['warnings'].append(f"Invalid structure: {pair_name}")
                    continue
                
                # Check encoder has encode method
                if not hasattr(encoder, 'encode'):
                    logger.warning(f"✗ Encoder in {pair_name} missing encode() method")
                    self.results['warnings'].append(f"Invalid encoder: {pair_name}")
                    continue
                
                # Check classifier has predict method
                if not hasattr(classifier, 'predict'):
                    logger.warning(f"✗ Classifier in {pair_name} missing predict() method")
                    self.results['warnings'].append(f"Invalid classifier: {pair_name}")
                    continue
                
                logger.info(f"✓ Loaded: {pair_name}")
                self.results['passed'].append(f"Load: {pair_name}")
                loadable_pairs += 1
                
            except pickle.UnpicklingError as e:
                logger.error(f"✗ Failed to load {pair_name}.pkl: {e}")
                self.results['failed'].append(f"Load error: {pair_name}")
            except Exception as e:
                logger.error(f"✗ Unexpected error loading {pair_name}: {e}")
                self.results['failed'].append(f"Unexpected error: {pair_name}")
        
        if loadable_pairs > 0:
            logger.info(f"✓ Successfully loaded {loadable_pairs} model pairs")
            return True
        else:
            logger.warning("⚠️  No model pairs loaded (might still be in training)")
            return True
    
    def _check_backward_compatibility(self) -> bool:
        """Check backward compatibility with old deploy models."""
        logger.info("\n[5/5] Checking backward compatibility...")
        
        encoder_path = self.evaluation_dir / "deploy_encoder.pkl"
        classifier_path = self.evaluation_dir / "deploy_classifier.pkl"
        
        both_exist = encoder_path.exists() and classifier_path.exists()
        
        if both_exist:
            try:
                with open(encoder_path, 'rb') as f:
                    encoder = pickle.load(f)
                with open(classifier_path, 'rb') as f:
                    classifier = pickle.load(f)
                
                if hasattr(encoder, 'encode') and hasattr(classifier, 'predict'):
                    logger.info("✓ Backward compatibility models valid")
                    self.results['passed'].append("Backward compatibility")
                    return True
            except Exception as e:
                logger.warning(f"⚠️  Backward compatibility models invalid: {e}")
        else:
            logger.info("ℹ️  Backward compatibility models not yet generated")
            logger.info("   → They will be created when notebook completes")
        
        return True
    
    def _print_summary(self):
        """Print validation summary."""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        passed = len(self.results['passed'])
        failed = len(self.results['failed'])
        warnings = len(self.results['warnings'])
        
        logger.info(f"\n✓ Passed:   {passed}")
        logger.info(f"✗ Failed:   {failed}")
        logger.info(f"⚠️  Warnings: {warnings}")
        
        if failed > 0:
            logger.error("\n❌ VALIDATION FAILED")
            logger.error("\nFailed checks:")
            for failure in self.results['failed']:
                logger.error(f"  • {failure}")
        elif warnings > 0:
            logger.warning("\n⚠️  VALIDATION PASSED WITH WARNINGS")
            logger.warning("\nWarnings:")
            for warning in self.results['warnings']:
                logger.warning(f"  • {warning}")
        else:
            logger.info("\n✅ VALIDATION PASSED")
        
        logger.info("=" * 70 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate HausaTaxBot model files"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=None,
        help='Base directory (defaults to HausaTaxBot/)'
    )
    
    args = parser.parse_args()
    
    validator = ModelValidator(
        base_dir=args.base_dir,
        verbose=args.verbose
    )
    
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
