"""
youtube_utils.py - YouTube Transcript and Metadata Handling for Youniversity

This module is responsible for all interactions with YouTube content, providing functionality
to fetch, process, and format video transcripts and metadata. It serves as the foundation
for enabling content-aware AI conversations about YouTube videos.

Key components:
1. YouTubeTranscriptFetcher Class: The main class that handles:
   - Extracting video IDs from various YouTube URL formats
   - Retrieving video metadata (title, author, length, thumbnail) using yt-dlp
   - Fetching transcripts through multiple methods:
     a. Primary: YouTube's transcript API
     b. Fallback: Audio extraction and transcription using Whisper
   - Formatting transcripts for AI context
   - Creating timestamped URLs for referencing specific video segments

Primary functions:
- get_transcript(): Retrieves transcript either from YouTube's API or by generating
  one from audio using Whisper
- get_video_info(): Fetches video metadata using yt-dlp with fallback options
- format_transcript_for_context(): Formats the transcript for use in LLM prompts
- format_time(): Converts seconds to MM:SS format
- get_timestamped_url(): Generates URLs that link to specific timestamps in videos

The module implements a robust approach to transcript retrieval with multiple fallback
mechanisms to handle various YouTube video configurations:
1. First attempts to get official transcripts via YouTube Transcript API
2. If official transcripts aren't available, downloads audio and generates transcripts
   using Whisper (OpenAI's speech-to-text model)

For performance optimization, the Whisper model is lazy-loaded only when needed for
transcription, minimizing memory usage when official transcripts are available.

Dependencies:
- youtube_transcript_api: For fetching official transcripts
- pytube: For downloading video audio
- yt-dlp: For reliable video metadata extraction
- whisper: For speech-to-text transcription
- torch: For running the Whisper model
"""
# Standard library imports
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import parse_qs, urlparse

# Third-party imports
import json
import requests
import whisper
import yt_dlp
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeTranscriptFetcher:
    """Fetches transcripts from YouTube videos"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the transcript fetcher.
        
        Args:
            model_size: Size of the whisper model (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.whisper_model = None  # Lazy load to save memory
    
    def get_video_id(self, url: str) -> str:
        """
        Extract the video ID from a YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            YouTube video ID
        """
        # Handle different URL formats
        if "youtu.be" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
        elif "youtube.com/watch" in url:
            # Parse the URL and extract the v parameter
            parsed_url = urlparse(url)
            video_id = parse_qs(parsed_url.query).get('v', [''])[0]
        elif "youtube.com/embed/" in url:
            video_id = url.split("youtube.com/embed/")[-1].split("?")[0]
        else:
            raise ValueError(f"Unsupported YouTube URL format: {url}")
        
        # Remove any additional parameters after the video ID
        video_id = video_id.split('&')[0]
        logger.info(f"Extracted video ID: {video_id}")
        
        return video_id
    
    def get_transcript(self, url: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get the transcript for a YouTube video.
        
        Args:
            url: YouTube URL
            
        Returns:
            Tuple of (transcript segments, video info)
        """
        video_id = self.get_video_id(url)
        logger.info(f"Fetching transcript for video ID: {video_id}")
        
        # Get video info (try multiple approaches)
        video_info = self.get_video_info(video_id, url)
        
        # Try to get transcript from YouTube API
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get English transcript first, then any available
            try:
                transcript = transcript_list.find_transcript(['en'])
            except NoTranscriptFound:
                # Try to get any manually created transcript
                try:
                    transcript = transcript_list.find_manually_created_transcript()
                except:
                    # Fall back to any generated transcript
                    transcript = transcript_list.find_generated_transcript()
            
            # Get the transcript data
            transcript_data = transcript.fetch()
            logger.info(f"Successfully fetched transcript with {len(transcript_data)} segments")
            
            return transcript_data, video_info
            
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning(f"No transcript available via API: {str(e)}")
            logger.info("Attempting to generate transcript from audio...")
            
            return self.generate_transcript_from_audio(url, video_info)
            
        except Exception as e:
            logger.error(f"Error fetching transcript: {str(e)}")
            raise
    def get_video_info(self, video_id: str, url: str) -> Dict[str, Any]:
        """
        Get video information using yt-dlp.
        
        Args:
            video_id: YouTube video ID
            url: Full YouTube URL
            
        Returns:
            Dictionary of video information
        """
        logger.info(f"Getting video info for ID: {video_id} using yt-dlp")
        
        # Default video info in case extraction fails
        default_info = {
            "title": "Video Information Not Available",
            "author": "Unknown Creator",
            "length": 0,
            "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            "video_id": video_id
        }
        
        try:
            # Configure yt-dlp options to only fetch metadata without downloading
            ydl_opts = {
                'skip_download': True,            # Don't download the video
                'quiet': True,                    # Don't print progress
                'no_warnings': True,              # Don't print warnings
                'extract_flat': True,             # Only extract metadata
                'force_generic_extractor': False  # Allow YouTube-specific extractor
            }
            
            # Create yt-dlp instance with options
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video information
                info = ydl.extract_info(url, download=False)
                
                if info:
                    # Extract the relevant fields
                    video_info = {
                        "title": info.get("title", default_info["title"]),
                        "author": info.get("uploader", default_info["author"]),
                        "length": info.get("duration", default_info["length"]),
                        "thumbnail_url": info.get("thumbnail", default_info["thumbnail_url"]),
                        "video_id": video_id
                    }
                    
                    logger.info(f"Successfully fetched video info: {video_info['title']}")
                    return video_info
        
        except Exception as e:
            logger.error(f"Error extracting video info with yt-dlp: {str(e)}")
            
            # Try to extract video_id if it wasn't provided
            if not video_id and url:
                try:
                    parsed_url = urlparse(url)
                    if 'youtube.com' in parsed_url.netloc and 'watch' in parsed_url.path:
                        video_id = parse_qs(parsed_url.query).get('v', [''])[0]
                    elif 'youtu.be' in parsed_url.netloc:
                        video_id = parsed_url.path.lstrip('/')
                    
                    if video_id:
                        default_info["video_id"] = video_id
                        default_info["thumbnail_url"] = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                except Exception:
                    pass
        
        # Return default info if extraction failed
        logger.warning("Using fallback video information")
        return default_info
    
    def load_whisper_model(self):
        """Load the Whisper model if not already loaded."""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model (size: {self.model_size})...")
            self.whisper_model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded")
    
    def generate_transcript_from_audio(self, url: str, video_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a transcript from the audio track of a video.
        
        Args:
            url: YouTube URL
            video_info: Video information dictionary
            
        Returns:
            Tuple of (transcript segments, video info)
        """
        try:
            # Lazy load the Whisper model
            self.load_whisper_model()
            
            # Download audio
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            
            # Update video info if we got it now
            if not video_info.get("title") or video_info.get("title") == "Video Information Not Available":
                video_info = {
                    "title": yt.title,
                    "author": yt.author,
                    "length": yt.length,
                    "thumbnail_url": yt.thumbnail_url,
                    "video_id": video_info.get("video_id")
                }
            
            # Use a temporary directory for the download
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                audio_file = temp_path / "audio.mp4"
                
                logger.info(f"Downloading audio from: {url}")
                audio_stream.download(output_path=str(temp_path), filename="audio.mp4")
                
                # Transcribe the audio
                logger.info("Transcribing audio...")
                result = self.whisper_model.transcribe(str(audio_file))
                
                # Convert Whisper result to YouTube transcript format
                transcript_data = []
                
                for segment in result["segments"]:
                    transcript_data.append({
                        "text": segment["text"].strip(),
                        "start": segment["start"],
                        "duration": segment["end"] - segment["start"]
                    })
                
                logger.info(f"Generated transcript with {len(transcript_data)} segments")
                return transcript_data, video_info
                
        except Exception as e:
            logger.error(f"Error generating transcript from audio: {str(e)}")
            raise
    
    def format_transcript_for_context(self, transcript: List[Dict[str, Any]], 
                                    video_info: Dict[str, Any], 
                                    url: str) -> str:
        """
        Format the transcript for use as context in LLM prompts.
        
        Args:
            transcript: List of transcript segments
            video_info: Video information dictionary
            url: YouTube URL
            
        Returns:
            Formatted transcript text
        """
        if not transcript:
            return "No transcript available."
        
        title = video_info.get("title", "Unknown Title")
        author = video_info.get("author", "Unknown Author")
        
        # Format header
        formatted_text = f"TITLE: {title}\n"
        formatted_text += f"CREATOR: {author}\n"
        formatted_text += f"URL: {url}\n\n"
        formatted_text += "TRANSCRIPT:\n"
        
        # Format each segment with timestamp
        for segment in transcript:
            try:
                # Try to access as dictionary first (original method)
                time_str = self.format_time(segment["start"])
                text = segment["text"]
            except (TypeError, KeyError):
                # Handle FetchedTranscriptSnippet objects
                time_str = self.format_time(segment.start)
                text = segment.text
                
            timestamped_url = self.get_timestamped_url(url, segment.start if hasattr(segment, 'start') else segment["start"])
            formatted_text += f"[{time_str}] {text}\n"
        
        return formatted_text
    
    def format_time(self, seconds: float) -> str:
        """
        Format a timestamp in seconds to MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (MM:SS)
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_timestamped_url(self, url: str, seconds: float) -> str:
        """
        Generate a timestamped YouTube URL.
        
        Args:
            url: Base YouTube URL
            seconds: Time in seconds
            
        Returns:
            Timestamped YouTube URL
        """
        # Remove any existing timestamp
        if "?" in url:
            base_url = url.split("?")[0]
            query_params = url.split("?")[1]
            params = {}
            
            for param in query_params.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value
            
            # Remove any existing timestamp parameter
            if "t" in params:
                del params["t"]
            
            # Rebuild the URL without the timestamp
            if params:
                base_url += "?" + "&".join([f"{key}={value}" for key, value in params.items()])
            
            url = base_url
        
        # Add the timestamp parameter
        if "?" in url:
            time_param = f"&t={int(seconds)}s"
        else:
            time_param = f"?t={int(seconds)}s"
        
        return url + time_param