"""
app.py - Main Application Entry Point for Youniversity

This file serves as the main entry point for the Youniversity application, a Streamlit-based
tool that allows users to learn from YouTube videos through AI-powered conversations.

Key responsibilities:
1. User Interface Management: Creates and manages the Streamlit UI, including video transcript display,
   chat interface, and settings sidebar.
2. Session State Management: Maintains application state across interactions, storing transcript data,
   video information, chat history, and user preferences.
3. Provider & Resource Management: Initializes and provides access to core services:
   - YouTubeTranscriptFetcher: Retrieves video transcripts and metadata
   - ProviderManager: Handles communication with various LLM providers (Ollama, OpenAI, Anthropic)
   - PromptManager: Manages prompt templates for AI interactions
4. Workflow Coordination: Orchestrates the main application workflows:
   - Transcript fetching and display
   - Chat interaction with LLMs about video content
   - Prompt template management

The application follows a single-page design where the chat interface appears directly
below the video transcript once loaded, providing a seamless experience for users to
interact with video content through natural language.

Dependencies:
- External Python packages: streamlit, dotenv, logging
- Custom modules: youtube_utils, llm_providers, prompt_manager
- Environment variables: Configuration loaded from .env file

User interaction flow:
1. User enters a YouTube URL and clicks "Fetch Transcript"
2. App displays video metadata and transcript
3. User can ask questions about the video in the chat section
4. AI responds based on the video content using the selected LLM

The application also provides customization options through the sidebar, allowing users
to select different LLM providers, models, and prompt templates.
"""

import streamlit as st # type: ignore
import os
import logging
import re
from pathlib import Path
import time
from dotenv import load_dotenv
import warnings

# Suppress torch warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch._classes")

# Import our custom modules
from youtube_utils import YouTubeTranscriptFetcher
from llm_providers import ProviderManager
from prompt_manager import PromptManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# init streamlit
st.set_page_config(page_title="Youniversity", page_icon="🎓", layout="wide")

# Add custom CSS for chat styling
st.markdown("""
<style>
    /* User message styling */
    .user-message {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    
    /* AI message styling */
    .ai-message {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    
    /* Message headers */
    .message-header {
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    /* Separator */
    .message-separator {
        margin: 15px 0;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state if not already initialized
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ""
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = "default.txt"
if 'editing_prompt' not in st.session_state:
    st.session_state.editing_prompt = False
if 'fetching_in_progress' not in st.session_state:
    st.session_state.fetching_in_progress = False

# Initialize our managers
@st.cache_resource
def get_transcript_fetcher():
    # Get whisper model size from environment or use default
    whisper_model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
    return YouTubeTranscriptFetcher(model_size=whisper_model_size)

@st.cache_resource
def get_provider_manager():
    config_path = os.environ.get("CONFIG_PATH", None)
    return ProviderManager(config_path)

@st.cache_resource
def get_prompt_manager():
    prompts_dir = os.environ.get("PROMPTS_DIR", "prompts")
    return PromptManager(prompts_dir)

# Get our manager instances
transcript_fetcher = get_transcript_fetcher()
provider_manager = get_provider_manager()
prompt_manager = get_prompt_manager()

# Function to process AI responses to convert timestamps to links
def process_timestamps(text, video_url):
    # Regular expression to find timestamps in format [MM:SS]
    pattern = r'\[(\d{2}:\d{2})\]'
    
    def replace_timestamp(match):
        timestamp = match.group(1)
        minutes, seconds = map(int, timestamp.split(':'))
        total_seconds = minutes * 60 + seconds
        timestamped_url = transcript_fetcher.get_timestamped_url(video_url, total_seconds)
        return f'<a href="{timestamped_url}" target="_blank">[{timestamp}]</a>'
    
    # Replace all timestamps with clickable links that open in a new tab
    processed_text = re.sub(pattern, replace_timestamp, text)
    return processed_text

# Function to fetch a transcript
def fetch_transcript(url):
    try:
        # Fetch the transcript
        with st.spinner("Fetching transcript..."):
            # Show a progress indicator
            progress = st.progress(0)
            
            # Fetch the transcript
            transcript, video_info = transcript_fetcher.get_transcript(url)
            
            if transcript is None:
                st.error("Could not fetch or generate a transcript for this video.")
                st.session_state.fetching_in_progress = False
                return False
            
            # Update session state
            st.session_state.transcript = transcript
            st.session_state.video_info = video_info
            st.session_state.youtube_url = url
            
            # Update progress
            progress.progress(100)
            
            # Set fetching flag to False
            st.session_state.fetching_in_progress = False
            return True
            
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        logger.error(f"Error fetching transcript: {e}", exc_info=True)
        st.session_state.fetching_in_progress = False
        return False

# Function to generate a response
def generate_response(question, provider_name, model_name):
    with st.spinner("Generating response..."):
        try:
            # Get the current prompt template
            prompt_template = prompt_manager.load_prompt(st.session_state.current_prompt)
            if not prompt_template:
                st.error(f"Could not load prompt template: {st.session_state.current_prompt}")
                return None
            
            # Format the prompt with the question
            formatted_prompt = prompt_manager.format_prompt(prompt_template, question)
            
            # Format the transcript for context
            context = transcript_fetcher.format_transcript_for_context(
                st.session_state.transcript,
                st.session_state.video_info,
                st.session_state.youtube_url
            )
            
            # Generate the response
            response, error = provider_manager.generate_response(
                provider_name=provider_name,
                model=model_name,
                prompt=formatted_prompt,
                context=context,
                temperature=0.7,
                max_tokens=1000
            )
            
            if error:
                st.error(f"Error generating response: {error}")
                return None
            
            return response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logger.error(f"Error generating response: {e}", exc_info=True)
            return None

# App title and description
st.title("🎓 Youniversity")
st.markdown("Learn from YouTube videos with AI-powered conversations")

# Create a sidebar for settings
st.sidebar.title("Settings")

# Select LLM provider and model
available_providers = provider_manager.get_available_providers()
default_provider = "ollama" if "ollama" in available_providers else available_providers[0] if available_providers else None

if default_provider:
    provider_name = st.sidebar.selectbox(
        "Select LLM Provider",
        options=available_providers,
        index=available_providers.index(default_provider) if default_provider in available_providers else 0
    )
    
    # Get models for the selected provider
    provider = provider_manager.get_provider(provider_name)
    available_models = provider.get_available_models() if provider else []
    
    # Set default model (llama3 for ollama, otherwise first available)
    default_model = "llama3" if provider_name == "ollama" and "llama3" in available_models else available_models[0] if available_models else None
    
    if default_model:
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=available_models,
            index=available_models.index(default_model) if default_model in available_models else 0
        )
    else:
        st.sidebar.warning(f"No models available for {provider_name}")
        model_name = None
else:
    st.sidebar.warning("No LLM providers available")
    provider_name = None
    model_name = None

# Prompt template selection
st.sidebar.markdown("---")
st.sidebar.subheader("Prompt Templates")

# Get available prompt templates
prompt_files = prompt_manager.get_prompt_files()

if prompt_files:
    # Select prompt template
    st.session_state.current_prompt = st.sidebar.selectbox(
        "Select Prompt Template",
        options=prompt_files,
        index=prompt_files.index("default.txt") if "default.txt" in prompt_files else 0
    )
    
    # Display the current prompt template
    current_prompt_content = prompt_manager.load_prompt(st.session_state.current_prompt)
    
    if st.sidebar.button("Edit Prompt Template"):
        st.session_state.editing_prompt = True
    
    if st.sidebar.button("Create New Prompt"):
        st.session_state.editing_prompt = True
        st.session_state.current_prompt = "new_prompt.txt"
        current_prompt_content = ""
else:
    st.sidebar.warning("No prompt templates available")
    current_prompt_content = ""

# Prompt template editor
if st.session_state.editing_prompt:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Edit Prompt Template")
    
    prompt_name = st.sidebar.text_input("Prompt Name", value=st.session_state.current_prompt)
    prompt_content = st.sidebar.text_area("Prompt Content", value=current_prompt_content, height=300)
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Save", key="save_prompt"):
            if prompt_manager.save_prompt(prompt_name, prompt_content):
                st.session_state.current_prompt = prompt_name
                st.session_state.editing_prompt = False
                st.rerun()
            else:
                st.error("Failed to save prompt template")
    
    with col2:
        if st.button("Cancel", key="cancel_edit"):
            st.session_state.editing_prompt = False
            st.rerun()

# Main content area - unified view
st.subheader("Video Transcript")
# YouTube URL input
youtube_url = st.text_input("Enter YouTube URL", value=st.session_state.youtube_url)

# Fetch transcript button
if st.button("Fetch Transcript"):
    if youtube_url:
        if not st.session_state.fetching_in_progress:
            # Set fetching flag to True to prevent duplicate requests
            st.session_state.fetching_in_progress = True
            # Schedule the fetching function to run after the rerun
            st.rerun()
    else:
        st.warning("Please enter a YouTube URL")

# Execute fetching if the flag is set
if st.session_state.fetching_in_progress:
    success = fetch_transcript(youtube_url)
    if success:
        st.success("Transcript fetched successfully!")

# Display video information if available
if st.session_state.transcript and st.session_state.video_info:
    # Display video info
    video_info = st.session_state.video_info
    
    # Debug info (uncomment for troubleshooting)
    # st.write("Debug - Video Info:", video_info)
    
    # Create two columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if video_info.get("thumbnail_url"):
            st.image(video_info["thumbnail_url"], use_container_width=True)
    
    with col2:
        st.subheader(video_info.get("title", "Unknown Title"))
        st.write(f"Author: {video_info.get('author', 'Unknown')}")
        if video_info.get("length", 0) > 0:
            st.write(f"Length: {transcript_fetcher.format_time(video_info.get('length', 0))}")
        
        # Link to the video
        st.markdown(f"[Watch Video]({st.session_state.youtube_url})")
    
    # Display the transcript
    st.markdown("### Transcript")
    
    # Create an expander for the full transcript
    with st.expander("Show Full Transcript", expanded=False):
        for segment in st.session_state.transcript:
            try:
                # Try to access as dictionary first (original method)
                time_str = transcript_fetcher.format_time(segment["start"])
                text = segment["text"]
            except (TypeError, KeyError):
                # Handle FetchedTranscriptSnippet objects
                time_str = transcript_fetcher.format_time(segment.start)
                text = segment.text
                
            # Get timestamped URL
            timestamped_url = transcript_fetcher.get_timestamped_url(
                st.session_state.youtube_url, 
                segment.start if hasattr(segment, 'start') else segment["start"]
            )
            
            # Display segment with timestamp link that opens in a new tab
            st.markdown(f'<a href="{timestamped_url}" target="_blank">[{time_str}]</a>: {text}', unsafe_allow_html=True)
    
    # Add a separator
    st.markdown("---")
    
    # Chat section - only show if transcript is loaded
    st.markdown("### Chat with the Video")
    
    if not provider_name or not model_name:
        st.warning("Please select a valid LLM provider and model in the sidebar to chat with the video")
    else:
        # Display chat history with improved styling
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            # Process the answer to convert timestamps to clickable links
            processed_answer = process_timestamps(answer, st.session_state.youtube_url)
            
            # User message with light gray background
            st.markdown(f"""
            <div class="user-message">
                <div class="message-header">You:</div>
                {question}
            </div>
            """, unsafe_allow_html=True)
            
            # AI response with light blue background and processed timestamps
            st.markdown(f"""
            <div class="ai-message">
                <div class="message-header">AI:</div>
                {processed_answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Add a separator
            st.markdown('<div class="message-separator"></div>', unsafe_allow_html=True)
        
        # Input for new questions
        question = st.text_input("Ask a question about the video")
        
        if st.button("Send", key="send_question"):
            if question:
                # Add to chat history immediately to display the question
                response = generate_response(question, provider_name, model_name)
                
                if response:
                    # Add the question and response to chat history
                    st.session_state.chat_history.append((question, response))
                    # Clear the input
                    question = ""
                    # Rerun to update the UI
                    st.rerun()
            else:
                st.warning("Please enter a question")
else:
    # If no transcript is loaded, show a message
    st.info("Please enter a YouTube URL and fetch a transcript to start chatting with the video")

# Footer
st.markdown("---")
st.markdown("Youniversity - Learn from YouTube videos with AI")