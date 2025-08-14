import logging
from typing import Optional
from pathlib import Path
import sys
import yaml
import logging
from litellm import acompletion

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    
    def __init__(self):
        self.model_name = f"ollama/{settings.ollama_model}"
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> dict:
        prompts_file = Path(__file__).parent.parent.parent / "prompts.yaml"
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            return {
                "system_prompt": "You are a helpful document analysis assistant.",
                "summary_prompt": "You are a document summarization assistant."
            }
        
    async def initialize(self):
        try:
            logging.getLogger("litellm").setLevel(logging.CRITICAL)
            logging.getLogger("litellm.proxy").setLevel(logging.CRITICAL)
            logging.getLogger("litellm.llms").setLevel(logging.CRITICAL)
            logging.getLogger("httpx").setLevel(logging.CRITICAL)
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)
            
        except Exception as e:
            logger.error(f"Error initializing LLM service: {str(e)}")
            raise
    
    async def generate_response(self, question: str, context: str, chat_history: Optional[str] = None, language: str = "en") -> str:
        try:
            language_name = "English" if language == "en" else "Turkish"
            system_prompt_template = self.prompts.get("system_prompt", "You are a helpful assistant.")
            system_prompt = system_prompt_template.format(language=language_name)
            user_message = self._build_user_message(question, context, chat_history)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _build_user_message(self, question: str, context: str, chat_history: Optional[str] = None) -> str:
        message_parts = []
        
        if chat_history:
            message_parts.append("PREVIOUS CONVERSATION:")
            message_parts.append(chat_history)
            message_parts.append("")
        
        message_parts.append("DOCUMENT CONTENT:")
        message_parts.append(context)
        message_parts.append("")
        message_parts.append("USER QUESTION:")
        message_parts.append(question)
        
        return "\n".join(message_parts)
    
    async def summarize_document(self, document_content: str, document_name: str) -> str:
        try:
            system_prompt = self.prompts.get("summary_prompt", "You are a document summarization assistant.")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"Please summarize the following document:\n\nDocument Name: {document_name}\n\nContent:\n{document_content[:5000]}..."
                }
            ]
            
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.5
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Summary could not be generated."