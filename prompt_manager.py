"""
prompt_manager.py - Prompt Template Management System for Youniversity

This module provides a complete system for managing, customizing, and applying prompt
templates that define how the application interacts with LLMs. It enables consistent
and configurable AI behavior through user-editable prompt templates.

Key components:
1. PromptManager Class: Central class responsible for:
   - Loading prompt templates from filesystem
   - Saving and updating user-modified templates
   - Managing the template directory structure
   - Formatting templates with dynamic content
   - Maintaining default templates for new installations

Key functionalities:
- Template discovery and listing: Finds available templates in the configured directory
- Template loading: Reads template content from files
- Template saving: Persists new or modified templates
- Template deletion: Removes templates (with protection for default templates)
- Template formatting: Applies variables like user questions to template placeholders
- Directory management: Creates and maintains the template directory structure
- Default template generation: Ensures a basic template exists for new installations

The prompt template system provides several benefits:
- Consistency: Standardizes AI interactions through carefully crafted prompts
- Customization: Allows users to modify prompts for different use cases
- Transparency: Makes AI behavior visible and adjustable
- Specialization: Enables domain-specific prompts for different types of videos

Templates use Python's string formatting syntax with placeholders like {question}
that get replaced with actual values during runtime. Default templates provide
instructions for the AI to focus on video transcript content, include timestamps
in references, and maintain helpful, accurate responses.

This module interacts primarily with the filesystem for template storage and with
the app.py module which uses the templates for generating LLM prompts.
"""
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Import the Config class
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptManager:
    """Manager for prompt templates"""
    
    def __init__(self, prompts_dir: str = None):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt template files
        """
        # Get application configuration
        config = Config.get_instance()
        
        # Use provided prompts directory or fall back to config
        self.prompts_dir = Path(prompts_dir or config.prompts_dir)
        
        logger.info(f"Initializing PromptManager with prompts directory: {self.prompts_dir}")
        self.ensure_prompts_directory()
        self.ensure_default_prompt()
    
    def ensure_prompts_directory(self):
        """Ensure the prompts directory exists."""
        if not self.prompts_dir.exists():
            logger.info(f"Creating prompts directory: {self.prompts_dir}")
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
    
    def ensure_default_prompt(self):
        """Ensure the default prompt template exists."""
        default_prompt_path = self.prompts_dir / "default.txt"
        
        if not default_prompt_path.exists():
            logger.info(f"Creating default prompt template: {default_prompt_path}")
            default_prompt = """You are a helpful assistant that answers questions about the content of a YouTube video.

Your task is to:
1. Answer the user's questions based only on the transcript provided
2. When referencing specific content, include the timestamp URL so the user can jump to that section
3. Be concise and accurate in your answers
4. If the transcript doesn't contain information to answer the question, be honest about this

User question: {question}"""
            
            with open(default_prompt_path, "w") as f:
                f.write(default_prompt)
    
    def get_prompt_files(self) -> List[str]:
        """
        Get a list of available prompt template files.
        
        Returns:
            List of prompt template filenames
        """
        try:
            return [f.name for f in self.prompts_dir.glob("*.txt")]
        except Exception as e:
            logger.error(f"Error getting prompt files: {e}")
            return []
    
    def load_prompt(self, filename: str) -> Optional[str]:
        """
        Load a prompt template from a file.
        
        Args:
            filename: Name of the prompt template file
            
        Returns:
            Prompt template text, or None if the file doesn't exist
        """
        try:
            prompt_path = self.prompts_dir / filename
            
            if not prompt_path.exists():
                logger.error(f"Prompt file not found: {prompt_path}")
                return None
            
            with open(prompt_path, "r") as f:
                content = f.read()
                logger.debug(f"Loaded prompt template: {filename}")
                return content
                
        except Exception as e:
            logger.error(f"Error loading prompt: {e}")
            return None
    
    def save_prompt(self, filename: str, content: str) -> bool:
        """
        Save a prompt template to a file.
        
        Args:
            filename: Name of the prompt template file
            content: Prompt template text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure filename has .txt extension
            if not filename.endswith(".txt"):
                filename += ".txt"
            
            prompt_path = self.prompts_dir / filename
            
            with open(prompt_path, "w") as f:
                f.write(content)
            
            logger.info(f"Saved prompt template: {prompt_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prompt: {e}")
            return False
    
    def delete_prompt(self, filename: str) -> bool:
        """
        Delete a prompt template file.
        
        Args:
            filename: Name of the prompt template file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prompt_path = self.prompts_dir / filename
            
            if not prompt_path.exists():
                logger.error(f"Prompt file not found: {prompt_path}")
                return False
            
            # Don't allow deletion of default.txt
            if prompt_path.name == "default.txt":
                logger.error("Cannot delete the default prompt template")
                return False
            
            prompt_path.unlink()
            logger.info(f"Deleted prompt template: {prompt_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting prompt: {e}")
            return False
    
    def format_prompt(self, template: str, question: str, **kwargs) -> str:
        """
        Format a prompt template with the given parameters.
        
        Args:
            template: Prompt template text
            question: User's question
            **kwargs: Additional parameters for formatting
            
        Returns:
            Formatted prompt
        """
        try:
            # Combine question and any additional kwargs
            format_params = {"question": question, **kwargs}
            
            # Format the template using the parameters
            return template.format(**format_params)
            
        except KeyError as e:
            logger.error(f"Missing parameter in prompt template: {e}")
            # Fall back to simple substitution if formatting fails
            return f"{template}\n\nUser question: {question}"
            
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return template