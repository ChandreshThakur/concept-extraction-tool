"""
Configuration management for the Concept Extraction project.
"""
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ExtractionConfig:
    """Configuration for concept extraction parameters."""
    confidence_threshold: float = 0.5
    max_concepts: int = 10
    min_concept_length: int = 3
    use_patterns: bool = True
    use_fuzzy_matching: bool = True
    rake_max_phrases: int = 15
    output_format: str = "csv"  # csv, json, both

@dataclass
class DirectoryConfig:
    """Configuration for project directories."""
    resources_dir: str = "resources"
    dictionaries_dir: str = "dictionaries"
    output_dir: str = "output"
    logs_dir: str = "logs"
    batch_output_dir: str = "batch_output"

@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    provider: str = "simulated"  # openai, anthropic, simulated
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 150
    timeout: int = 30

@dataclass
class ProjectConfig:
    """Main project configuration."""
    extraction: ExtractionConfig
    directories: DirectoryConfig
    llm: LLMConfig
    
    @classmethod
    def load_from_file(cls, config_file: str = "config.json") -> "ProjectConfig":
        """Load configuration from a JSON file."""
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return cls(
                extraction=ExtractionConfig(**config_data.get('extraction', {})),
                directories=DirectoryConfig(**config_data.get('directories', {})),
                llm=LLMConfig(**config_data.get('llm', {}))
            )
        else:
            # Return default configuration
            return cls(
                extraction=ExtractionConfig(),
                directories=DirectoryConfig(),
                llm=LLMConfig()
            )
    
    def save_to_file(self, config_file: str = "config.json"):
        """Save configuration to a JSON file."""
        config_data = {
            'extraction': asdict(self.extraction),
            'directories': asdict(self.directories),
            'llm': asdict(self.llm)
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def create_directories(self):
        """Create all configured directories."""
        for dir_name in [self.directories.resources_dir, self.directories.dictionaries_dir,
                        self.directories.output_dir, self.directories.logs_dir,
                        self.directories.batch_output_dir]:
            Path(dir_name).mkdir(exist_ok=True)

class ConfigManager:
    """Manager for handling project configuration."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = ProjectConfig.load_from_file(config_file)
    
    def get_config(self) -> ProjectConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for section, params in kwargs.items():
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                for key, value in params.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def save_config(self):
        """Save the current configuration to file."""
        self.config.save_to_file(self.config_file)
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = ProjectConfig(
            extraction=ExtractionConfig(),
            directories=DirectoryConfig(),
            llm=LLMConfig()
        )
    
    def validate_config(self) -> List[str]:
        """Validate the current configuration and return any issues."""
        issues = []
        
        # Validate extraction config
        if self.config.extraction.confidence_threshold < 0 or self.config.extraction.confidence_threshold > 1:
            issues.append("confidence_threshold must be between 0 and 1")
        
        if self.config.extraction.max_concepts < 1:
            issues.append("max_concepts must be at least 1")
        
        # Validate directories
        required_dirs = [
            self.config.directories.resources_dir,
            self.config.directories.dictionaries_dir
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                issues.append(f"Required directory does not exist: {dir_path}")
        
        # Validate LLM config
        if self.config.llm.provider in ['openai', 'anthropic'] and not self.config.llm.api_key:
            issues.append(f"API key required for provider: {self.config.llm.provider}")
        
        return issues

def create_default_config():
    """Create a default configuration file."""
    config = ProjectConfig(
        extraction=ExtractionConfig(),
        directories=DirectoryConfig(),
        llm=LLMConfig()
    )
    config.save_to_file()
    config.create_directories()
    print("Default configuration created: config.json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management for Concept Extraction")
    parser.add_argument("--create-default", action="store_true", help="Create default configuration file")
    parser.add_argument("--validate", action="store_true", help="Validate current configuration")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config()
    
    if args.validate:
        manager = ConfigManager()
        issues = manager.validate_config()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid.")
    
    if args.show:
        manager = ConfigManager()
        config_dict = {
            'extraction': asdict(manager.config.extraction),
            'directories': asdict(manager.config.directories),
            'llm': asdict(manager.config.llm)
        }
        print(json.dumps(config_dict, indent=2))
