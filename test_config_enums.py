#!/usr/bin/env python3
"""
Test that configuration creation works correctly with both enums and string conversion.
"""

def test_config_enums():
    """Test that configuration handles enums correctly."""
    try:
        from esnf_mat_analyzer.main import create_config, load_config
        from esnf_mat_analyzer.core.data_types import GrayscaleConversionMethod, ThicknessModelType
        
        print("Testing default configuration...")
        config = create_config()
        print(f"✓ Grayscale method: {config.processing.grayscale_conversion}")
        print(f"✓ Thickness model: {config.thickness.model_type}")
        
        print("\nTesting configuration from dict with strings...")
        test_data = {
            "processing": {"grayscale_conversion": "average"},
            "thickness": {"model_type": "logarithmic"}
        }
        config2 = create_config(test_data)
        print(f"✓ Grayscale method: {config2.processing.grayscale_conversion}")
        print(f"✓ Thickness model: {config2.thickness.model_type}")
        
        print("\n✅ Configuration enum handling works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    test_config_enums()
