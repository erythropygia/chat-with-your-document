import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

class ChatLogger:
    
    def __init__(self):
        self.chat_logs_dir = settings.logs_dir / "chats"
        self.chat_logs_dir.mkdir(exist_ok=True)
        
    def log_chat_interaction(
        self,
        question: str,
        response: str,
        language: str,
        context: str,
        chat_history: List[Dict[str, Any]] = None,
        sources: List[Dict[str, Any]] = None,
        processing_time: float = 0.0
    ) -> str:
        
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        chat_data = {
            "chat_id": chat_id,
            "timestamp": timestamp,
            "language": language,
            "processing_time": processing_time,
            "interaction": {
                "question": question,
                "response": response,
                "context_length": len(context) if context else 0,
                "chat_history_length": len(chat_history) if chat_history else 0
            },
            "sources": sources or [],
            "chat_history": chat_history or [],
            "context": context[:1000] if context else ""  # First 1000 chars only
        }
        
        # Save to daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.chat_logs_dir / f"chat_log_{date_str}.jsonl"
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(chat_data, ensure_ascii=False) + "\n")
        except Exception:
            # Silent fail - don't interrupt chat if logging fails
            pass
            
        return chat_id
    
    def get_chat_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_chats": 0,
            "languages": {},
            "avg_processing_time": 0.0
        }
        
        total_time = 0.0
        
        try:
            for log_file in self.chat_logs_dir.glob("*.jsonl"):
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            stats["total_chats"] += 1
                            
                            lang = data.get("language", "unknown")
                            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                            
                            total_time += data.get("processing_time", 0.0)
            
            if stats["total_chats"] > 0:
                stats["avg_processing_time"] = total_time / stats["total_chats"]
                
        except Exception:
            pass
            
        return stats