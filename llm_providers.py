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
   - OllamaProvider: Connects to locally-running Ollama models (e.g., LLaMA, Mistral)
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
- requests: For HTTP communication with Ollama and Anthropic APIs
- json: For parsing API responses
- logging: For error tracking and debugging
"""
import os
import json
from typing import Dict, List, Optional, Any, Tuple
import requests
import logging
from openai import OpenAI
from pathlib import Path
import importlib.util

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
        

class OllamaProvider(LLMProvider):
    """Provider for Ollama models"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Ollama provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "ollama"
        self.base_url = self.config.get("api_url", "http://localhost:11434")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of model names
        """
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
    
    def generate_response(self, prompt: str, context: str, model: str = "llama3", 
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate a response using Ollama.
        
        Args:
            prompt: The prompt template
            context: The context to use (transcript)
            model: The model to use (e.g., "llama3")
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            formatted_message = self.format_message(prompt, context)
            
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
    
    def format_message(self, prompt: str, context: str) -> str:
        """Format the message for Ollama models"""
        return f"{prompt}\n\nCONTEXT:\n{context}\n\nAnswer:"


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
        self.api_key = self.config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
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
    
    def generate_response(self, prompt: str, context: str, model: str = "gpt-4o", 
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
            
            system_prompt = """You are a helpful AI assistant that answers questions about YouTube videos.
            Use the provided video transcript to answer questions accurately.
            When referencing specific parts of the video, include the timestamp URL so the user can jump to that part.
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
        self.api_key = self.config.get("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.default_model = "claude-3-opus-20240229"
    
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
            "claude-2.1", 
            "claude-2.0"
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
            
            model = model or self.default_model
            
            system_prompt = """You are a helpful AI assistant that answers questions about YouTube videos.
            Use the provided video transcript to answer questions accurately.
            When referencing specific parts of the video, include the timestamp URL so the user can jump to that part.
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
        # Configure Ollama provider
        ollama_config = self.config.get("ollama", {})
        if not ollama_config:
            ollama_config = {"api_url": "http://localhost:11434"}
        self.providers["ollama"] = OllamaProvider(ollama_config)
        
        # Configure OpenAI provider
        openai_config = self.config.get("openai", {})
        if not openai_config:
            openai_config = {"api_key": os.environ.get("OPENAI_API_KEY", "")}
        self.providers["openai"] = OpenAIProvider(openai_config)
        
        # Configure Anthropic provider
        anthropic_config = self.config.get("anthropic", {})
        if not anthropic_config:
            anthropic_config = {"api_key": os.environ.get("ANTHROPIC_API_KEY", "")}
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
                         temperature: float = 0.7, max_tokens: int = 1000) -> Tuple[str, str]:
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