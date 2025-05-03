"""
llm_providers.py - LLM Integration Framework for Youniversity

This module implements a flexible and extensible framework for connecting to various
Large Language Model (LLM) providers, allowing the application to leverage different
AI models based on availability, performance needs, or user preference.

Key components:
1. Base Class:
   - LLMProvider: Abstract base class defining the common interface for all providers
     with methods for listing available models and generating responses.

2. Provider Implementations:
   - LocalLLMProvider: Connects to locally-running LLM servers (Ollama, KoboldAI, etc.)
   - OpenAIProvider: Integrates with OpenAI's API for models like GPT-4o
   - AnthropicProvider: Connects to Anthropic's API for Claude models

3. Manager Class:
   - ProviderManager: Orchestrates provider registration, selection, and usage,
     providing a unified interface for the application to interact with any LLM
     regardless of the backend service.

Key functionalities:
- Provider registration and discovery
- Model listing for each provider
- Unified prompt formatting and response generation
- Configuration loading from environment variables or config files
- Error handling and logging for API interactions

The architecture follows a plugin-style design pattern where new LLM providers can be
added by implementing the LLMProvider interface and registering with the ProviderManager.
Each provider handles the specific requirements of its API (authentication, request
formatting, response parsing), while maintaining a consistent interface for the application.

The module supports dynamic provider and model selection at runtime, allowing users to
switch between different LLMs without application restart, and handles the appropriate
context formatting for each provider to ensure optimal performance with video transcripts.

Dependencies:
- openai: For OpenAI API integration
- requests: For HTTP communication with Local LLM servers and Anthropic APIs
- json: For parsing API responses
- logging: For error tracking and debugging
"""
import json
from typing import Dict, List, Optional, Any, Tuple
import requests
import logging
from openai import OpenAI
from pathlib import Path
import importlib.util

# Import the Config class
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the LLM provider with configuration.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self.config = config or {}
        self.name = "base"
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model names
        """
        return []
    
    def generate_response(self, prompt: str, context: str, model: str, 
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt template
            context: The context to use (transcript)
            model: The model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response as string
        """
        raise NotImplementedError("Subclasses must implement generate_response")
    
    def format_message(self, prompt: str, context: str) -> str:
        """
        Format the message for the model.
        
        Args:
            prompt: The prompt template
            context: The context (transcript)
            
        Returns:
            Formatted message
        """
        return f"{prompt}\n\nCONTEXT:\n{context}"


class LocalLLMProvider(LLMProvider):
    """Provider for locally-hosted LLM services (Ollama, KoboldAI, etc.)"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Local LLM provider.
        
        Args:
            config: Configuration dictionary containing:
                   - api_url: Base URL for the API
                   - provider_type: Type of local provider (ollama, kobold, etc.)
        """
        super().__init__(config)
        self.name = "local"
        
        # Get application config
        app_config = Config.get_instance()
        
        # Use config values from provider config first, then fall back to app config
        self.base_url = self.config.get("api_url", app_config.local_llm_api_url)
        self.provider_type = self.config.get("provider_type", app_config.local_llm_type).lower()
        
        logger.info(f"Initialized Local LLM Provider of type '{self.provider_type}' at {self.base_url}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from the local LLM server.
        
        Returns:
            List of model names
        """
        if self.provider_type == "ollama":
            return self._get_ollama_models()
        elif self.provider_type == "kobold":
            return self._get_kobold_models()
        else:
            logger.warning(f"Unsupported local provider type: {self.provider_type}")
            return []
    
    def _get_ollama_models(self) -> List[str]:
        """Get available models from Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
            else:
                logger.error(f"Failed to get Ollama models: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []
    
    def _get_kobold_models(self) -> List[str]:
        """Get available models from KoboldAI server"""
        try:
            # KoboldAI API endpoint for model listing
            # Adjust this based on the actual KoboldAI API
            response = requests.get(f"{self.base_url}/api/v1/model")
            if response.status_code == 200:
                data = response.json()
                # Parse according to KoboldAI's API response format
                # This is a placeholder - adjust based on actual API
                models = data.get('models', [])
                return models
            else:
                logger.error(f"Failed to get KoboldAI models: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting KoboldAI models: {e}")
            return []
    
    def generate_response(self, prompt: str, context: str, model: str = None, 
                          temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate a response using the local LLM provider.
        
        Args:
            prompt: The prompt template
            context: The context to use (transcript)
            model: The model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        # Get application config for default values
        app_config = Config.get_instance()
        
        # Use passed model or fall back to config
        if model is None:
            model = app_config.local_llm_standard_model
        
        if self.provider_type == "ollama":
            return self._generate_ollama_response(prompt, context, model, temperature, max_tokens)
        elif self.provider_type == "kobold":
            return self._generate_kobold_response(prompt, context, model, temperature, max_tokens)
        else:
            return f"Error: Unsupported local provider type: {self.provider_type}"

    def _generate_ollama_response(self, prompt: str, context: str, model: str = "llama3", 
                                 temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate response using Ollama"""
        try:
            formatted_message = self._format_ollama_message(prompt, context)
            
            payload = {
                "model": model,
                "prompt": formatted_message,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response generated")
            else:
                error_msg = f"Ollama error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            return f"Error: {str(e)}"
    
    def _generate_kobold_response(self, prompt: str, context: str, model: str = None, 
                                 temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate response using KoboldAI"""
        try:
            formatted_message = self._format_kobold_message(prompt, context)
            
            # Adjust this based on the actual KoboldAI API
            payload = {
                "prompt": formatted_message,
                "max_length": max_tokens,
                "temperature": temperature,
                "model": model
            }
            
            # KoboldAI API endpoint for text generation
            # Adjust this based on the actual KoboldAI API
            response = requests.post(
                f"{self.base_url}/api/v1/generate",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                # Parse according to KoboldAI's API response format
                # This is a placeholder - adjust based on actual API
                return data.get("text", "No response generated")
            else:
                error_msg = f"KoboldAI error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error generating response with KoboldAI: {e}")
            return f"Error: {str(e)}"
    
    def _format_ollama_message(self, prompt: str, context: str) -> str:
        """Format the message for Ollama models"""
        return f"{prompt}\n\nCONTEXT:\n{context}\n\nAnswer:"
    
    def _format_kobold_message(self, prompt: str, context: str) -> str:
        """Format the message for KoboldAI models"""
        return f"{prompt}\n\nCONTEXT:\n{context}\n\nAnswer:"
    
    def format_message(self, prompt: str, context: str) -> str:
        """Format the message based on provider type"""
        if self.provider_type == "ollama":
            return self._format_ollama_message(prompt, context)
        elif self.provider_type == "kobold":
            return self._format_kobold_message(prompt, context)
        else:
            return super().format_message(prompt, context)


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "openai"
        
        # Get application config
        app_config = Config.get_instance()
        
        # Use config values from provider config first, then fall back to app config
        self.api_key = self.config.get("api_key", app_config.openai_api_key)
        self.client = OpenAI(api_key=self.api_key)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from OpenAI.
        
        Returns:
            List of model names
        """
        try:
            if not self.api_key:
                logger.warning("OpenAI API key not set")
                return []
                
            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            # Filter to just the chat models
            chat_models = [m for m in model_names if any(name in m for name in ["gpt", "claude", "GPT", "dall-e"])]
            return chat_models
        except Exception as e:
            logger.error(f"Error getting OpenAI models: {e}")
            return []
    
    def generate_response(self, prompt: str, context: str, model: str = None, 
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate a response using OpenAI.
        
        Args:
            prompt: The prompt template
            context: The context to use (transcript)
            model: The model to use (e.g., "gpt-4o")
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            if not self.api_key:
                return "Error: OpenAI API key not set"
            
            # Get application config for default values
            app_config = Config.get_instance()
            
            # Use passed model or fall back to config
            if model is None:
                model = app_config.openai_default_model
            
            system_prompt = """You are a helpful AI assistant that answers questions about YouTube videos.
                                Use the provided video transcript to answer questions accurately.
                                When referencing specific parts of the video, include single timestamps in [MM:SS] format showing when that topic begins.
                                Never use timestamp ranges - only use the starting timestamp for each reference.
                                Include multiple timestamps throughout your response when discussing different parts of the video.
                                If you don't know the answer, say so rather than making up information."""
            
            formatted_context = f"TRANSCRIPT:\n{context}"
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_context},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return f"Error: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "anthropic"
        
        # Get application config
        app_config = Config.get_instance()
        
        # Use config values from provider config first, then fall back to app config
        self.api_key = self.config.get("api_key", app_config.anthropic_api_key)
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Claude models.
        
        Returns:
            List of model names
        """
        # Anthropic doesn't have a models list endpoint, so we hardcode the models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-3.5-sonnet-20240620",
            "claude-3.7-sonnet-20250219"
        ]
    
    def generate_response(self, prompt: str, context: str, model: str = None, 
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate a response using Anthropic Claude.
        
        Args:
            prompt: The prompt template
            context: The context to use (transcript)
            model: The model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            if not self.api_key:
                return "Error: Anthropic API key not set"
            
            # Get application config for default values
            app_config = Config.get_instance()
            
            # Use passed model or fall back to config
            if model is None:
                model = app_config.anthropic_default_model
            
            system_prompt = """You are a helpful AI assistant that answers questions about YouTube videos.
                                Use the provided video transcript to answer questions accurately.
                                When referencing specific parts of the video, include single timestamps in [MM:SS] format showing when that topic begins.
                                Never use timestamp ranges - only use the starting timestamp for each reference.
                                Include multiple timestamps throughout your response when discussing different parts of the video.
                                If you don't know the answer, say so rather than making up information."""
            
            formatted_message = self.format_message(prompt, context)
            
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": formatted_message}
                ],
                "system": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("content", [{"text": "No response generated"}])[0]["text"]
            else:
                error_msg = f"Anthropic error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error generating response with Anthropic: {e}")
            return f"Error: {str(e)}"


class ProviderManager:
    """Manager for LLM providers"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the provider manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.providers = {}
        self.config = {}
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Register default providers
        self.register_default_providers()
    
    def load_config(self, config_path: str):
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}
    
    def register_default_providers(self):
        """Register the default LLM providers."""
        # Get application config
        app_config = Config.get_instance()
        
        # Configure Local LLM provider
        local_config = self.config.get("local", {})
        if not local_config:
            local_config = {
                "api_url": app_config.local_llm_api_url,
                "provider_type": app_config.local_llm_type
            }
        self.providers["local"] = LocalLLMProvider(local_config)
        
        # Configure OpenAI provider
        openai_config = self.config.get("openai", {})
        if not openai_config:
            openai_config = {"api_key": app_config.openai_api_key}
        self.providers["openai"] = OpenAIProvider(openai_config)
        
        # Configure Anthropic provider
        anthropic_config = self.config.get("anthropic", {})
        if not anthropic_config:
            anthropic_config = {"api_key": app_config.anthropic_api_key}
        self.providers["anthropic"] = AnthropicProvider(anthropic_config)
    
    def register_provider(self, provider_name: str, provider_instance: LLMProvider):
        """
        Register a new provider.
        
        Args:
            provider_name: Name of the provider
            provider_instance: Provider instance
        """
        self.providers[provider_name] = provider_instance
    
    def get_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """
        Get a provider by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider instance, or None if not found
        """
        return self.providers.get(provider_name)
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider names.
        
        Returns:
            List of provider names
        """
        return list(self.providers.keys())
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """
        Get all available models from all providers.
        
        Returns:
            Dictionary mapping provider names to lists of model names
        """
        models = {}
        for provider_name, provider in self.providers.items():
            try:
                provider_models = provider.get_available_models()
                if provider_models:
                    models[provider_name] = provider_models
            except Exception as e:
                logger.error(f"Error getting models for {provider_name}: {e}")
        
        return models
    
    def generate_response(self, provider_name: str, model: str, prompt: str, context: str,
                         temperature: float = None, max_tokens: int = None) -> Tuple[str, str]:
        """
        Generate a response using the specified provider and model.
        
        Args:
            provider_name: Name of the provider
            model: Name of the model to use
            prompt: The prompt template
            context: The context to use (transcript)
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response text, error message if any)
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return "", f"Provider '{provider_name}' not found"
        
        try:
            # Get application config for default values
            app_config = Config.get_instance()
            
            # Use provided values or fall back to config
            if temperature is None:
                temperature = app_config.model_temperature
            
            if max_tokens is None:
                max_tokens = app_config.model_max_tokens
            
            response = provider.generate_response(
                prompt=prompt,
                context=context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response, ""
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return "", error_msg