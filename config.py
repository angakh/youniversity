"""
config.py - Centralized Configuration Management for Youniversity

This module provides a centralized configuration system for the Youniversity application.
It handles loading configuration from multiple sources with a priority order:
1. Environment variables (highest priority)
2. Configuration files (.env file)
3. Default values (lowest priority)

Key features:
- Type validation for configuration values
- Environment-specific configurations (dev, test, prod)
- Sensible defaults for optional settings
- Helpful error messages for missing required values
- Singleton pattern for application-wide access

The Config class follows a singleton pattern to ensure consistent configuration
throughout the application. It provides typed access to all configuration values
and validates required settings at initialization time.

Usage:
    from config import Config
    
    # Get the singleton instance
    config = Config.get_instance()
    
    # Access configuration values with proper typing
    api_url = config.local_llm_api_url
    model_name = config.local_llm_standard_model
    debug_mode = config.debug
"""

import os
import logging
from typing import Dict, List, Optional, Any, Set, ClassVar
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """
    Centralized configuration management for Youniversity.
    
    This class follows the Singleton pattern to ensure consistent configuration
    across the application. It loads configuration from environment variables,
    .env files, and provides sensible defaults.
    """
    
    # Class variable to store the singleton instance
    _instance: ClassVar[Optional['Config']] = None
    
    # Configuration default values
    _DEFAULTS = {
        # Application configuration
        "APP_NAME": "Youniversity",
        "DEBUG": False,
        
        # YouTube transcript configuration
        "WHISPER_MODEL_SIZE": "base",
        "TRANSCRIPTION_LANGUAGES": "en,es,fr,de",
        
        # LLM Providers configuration
        "DEFAULT_PROVIDER": "local",
        
        # Local LLM configuration
        "LOCAL_LLM_TYPE": "ollama",
        "LOCAL_LLM_API_URL": "http://localhost:11434",
        "LOCAL_LLM_STANDARD_MODEL": "llama3",
        "LOCAL_LLM_REASONING_MODEL": "mistral",
        
        # Model parameters
        "MODEL_TEMPERATURE": 0.7,
        "MODEL_MAX_TOKENS": 1000,
        
        # Directory paths
        "PROMPTS_DIR": "prompts"
    }
    
    # Set of required configuration keys (will raise error if missing)
    _REQUIRED_KEYS: Set[str] = set()
    
    @classmethod
    def get_instance(cls) -> 'Config':
        """
        Get the singleton instance of the Config class.
        
        Returns:
            Config: The singleton instance
        """
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance
    
    def __init__(self):
        """
        Initialize the configuration by loading from environment variables
        and setting default values.
        """
        # Don't re-initialize if this is already instantiated
        if Config._instance is not None:
            return
        
        # Load environment variables from .env file if it exists
        dotenv_path = Path('.env')
        if dotenv_path.exists():
            load_dotenv(dotenv_path=str(dotenv_path))
            logger.info(f"Loaded configuration from {dotenv_path}")
        
        # Store configuration values
        self._config = {}
        
        # Load configuration values
        self._load_configuration()
        
        # Validate required configuration
        self._validate_required_config()
        
        # Log the initialization (omitting sensitive values)
        self._log_config()
    
    def _load_configuration(self):
        """
        Load configuration from environment variables with defaults.
        """
        # Set default values
        for key, default_value in self._DEFAULTS.items():
            self._config[key] = default_value
        
        # Override with environment variables if they exist
        for key in self._DEFAULTS.keys():
            env_value = os.environ.get(key)
            if env_value is not None:
                # Convert boolean strings to actual booleans
                if env_value.lower() in ('true', 'yes', '1'):
                    self._config[key] = True
                elif env_value.lower() in ('false', 'no', '0'):
                    self._config[key] = False
                # Convert comma-separated strings to lists
                elif isinstance(self._DEFAULTS[key], list) and ',' in env_value:
                    self._config[key] = [item.strip() for item in env_value.split(',')]
                # Convert numeric strings to float or int
                elif isinstance(self._DEFAULTS[key], (int, float)):
                    try:
                        if isinstance(self._DEFAULTS[key], int):
                            self._config[key] = int(env_value)
                        else:
                            self._config[key] = float(env_value)
                    except ValueError:
                        logger.warning(f"Could not convert {key}={env_value} to a number, using default {self._DEFAULTS[key]}")
                # Otherwise, use the string value
                else:
                    self._config[key] = env_value
    
    def _validate_required_config(self):
        """
        Validate that all required configuration keys have values.
        """
        missing_keys = []
        for key in self._REQUIRED_KEYS:
            if key not in self._config or self._config[key] is None:
                missing_keys.append(key)
        
        if missing_keys:
            error_msg = f"Missing required configuration: {', '.join(missing_keys)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _log_config(self):
        """
        Log the current configuration, omitting sensitive values.
        """
        # Define keys that might contain sensitive information
        sensitive_keys = {'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'}
        
        # Create a copy of the config with sensitive values masked
        masked_config = {}
        for key, value in self._config.items():
            if key in sensitive_keys and value:
                masked_config[key] = '********'
            else:
                masked_config[key] = value
        
        logger.info(f"Configuration initialized: {masked_config}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value if the key doesn't exist
            
        Returns:
            The configuration value or the default
        """
        return self._config.get(key, default)
    
    def set_for_testing(self, key: str, value: Any):
        """
        Set a configuration value (for testing purposes only).
        
        Args:
            key: The configuration key
            value: The value to set
            
        Note: This method should only be used in test code.
        """
        self._config[key] = value
    
    # Properties for typed access to configuration values
    
    # Application configuration
    @property
    def app_name(self) -> str:
        """Get the application name."""
        return str(self._config.get('APP_NAME', self._DEFAULTS['APP_NAME']))
    
    @property
    def debug(self) -> bool:
        """Get the debug mode flag."""
        return bool(self._config.get('DEBUG', self._DEFAULTS['DEBUG']))
    
    # YouTube transcript configuration
    @property
    def whisper_model_size(self) -> str:
        """Get the Whisper model size for audio transcription."""
        return str(self._config.get('WHISPER_MODEL_SIZE', self._DEFAULTS['WHISPER_MODEL_SIZE']))
    
    @property
    def transcription_languages(self) -> List[str]:
        """Get the list of transcription languages."""
        langs = self._config.get('TRANSCRIPTION_LANGUAGES', self._DEFAULTS['TRANSCRIPTION_LANGUAGES'])
        if isinstance(langs, str):
            return [lang.strip() for lang in langs.split(',')]
        return langs
    
    # LLM Providers configuration
    @property
    def default_provider(self) -> str:
        """Get the default LLM provider."""
        return str(self._config.get('DEFAULT_PROVIDER', self._DEFAULTS['DEFAULT_PROVIDER']))
    
    # Local LLM configuration
    @property
    def local_llm_type(self) -> str:
        """Get the type of local LLM provider."""
        return str(self._config.get('LOCAL_LLM_TYPE', self._DEFAULTS['LOCAL_LLM_TYPE']))
    
    @property
    def local_llm_api_url(self) -> str:
        """Get the API URL for the local LLM provider."""
        return str(self._config.get('LOCAL_LLM_API_URL', self._DEFAULTS['LOCAL_LLM_API_URL']))
    
    @property
    def local_llm_standard_model(self) -> str:
        """Get the standard model for the local LLM provider."""
        return str(self._config.get('LOCAL_LLM_STANDARD_MODEL', self._DEFAULTS['LOCAL_LLM_STANDARD_MODEL']))
    
    @property
    def local_llm_reasoning_model(self) -> str:
        """Get the reasoning model for the local LLM provider."""
        return str(self._config.get('LOCAL_LLM_REASONING_MODEL', self._DEFAULTS['LOCAL_LLM_REASONING_MODEL']))
    
    # OpenAI configuration
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get the OpenAI API key."""
        return self._config.get('OPENAI_API_KEY')
    
    @property
    def openai_default_model(self) -> str:
        """Get the default OpenAI model."""
        return str(self._config.get('OPENAI_DEFAULT_MODEL', 'gpt-4o'))
    
    # Anthropic configuration
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Get the Anthropic API key."""
        return self._config.get('ANTHROPIC_API_KEY')
    
    @property
    def anthropic_default_model(self) -> str:
        """Get the default Anthropic model."""
        return str(self._config.get('ANTHROPIC_DEFAULT_MODEL', 'claude-3-opus-20240229'))
    
    # Model parameters
    @property
    def model_temperature(self) -> float:
        """Get the temperature parameter for LLM generation."""
        return float(self._config.get('MODEL_TEMPERATURE', self._DEFAULTS['MODEL_TEMPERATURE']))
    
    @property
    def model_max_tokens(self) -> int:
        """Get the maximum tokens for LLM generation."""
        return int(self._config.get('MODEL_MAX_TOKENS', self._DEFAULTS['MODEL_MAX_TOKENS']))
    
    # Directory paths
    @property
    def prompts_dir(self) -> str:
        """Get the directory path for prompt templates."""
        return str(self._config.get('PROMPTS_DIR', self._DEFAULTS['PROMPTS_DIR']))