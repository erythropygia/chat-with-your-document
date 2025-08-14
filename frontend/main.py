import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import re

st.set_page_config(
    page_title="Search on Documents",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_file = Path(__file__).parent / "styles.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

BACKEND_URL = "http://localhost:8000"

LANGUAGES = {
    "en": "English",
    "tr": "Türkçe"
}

class DocumentSearchApp:
    def __init__(self):
        self.backend_url = BACKEND_URL
    
    def _clean_source_content(self, content: str) -> str:
        rtf_patterns = [
            r'\\f\d+',          
            r'\\b\d*',          
            r'\\cf\d+',         
            r'\\strokec\d+',    
            r'\\uc\d+',         
            r'\\[a-zA-Z]+\d*',  
            r'\\[\{\}\\]',      
        ]
        
        cleaned_content = content
        for pattern in rtf_patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content)
        
        try:
            cleaned_content = cleaned_content.encode('utf-8').decode('unicode_escape')
        except (UnicodeDecodeError, UnicodeEncodeError):
            cleaned_content = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), cleaned_content)
        
        try:
            cleaned_content = re.sub(r"\\\'([a-fA-F0-9]{2})", 
                                   lambda m: bytes.fromhex(m.group(1)).decode('latin-1'), 
                                   cleaned_content)
        except:
            pass
        
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
        
    def check_backend_connection(self) -> bool:
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def upload_document(self, uploaded_file) -> Dict[str, Any]:
        try:
            files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
            response = requests.post(f"{self.backend_url}/upload-document", files=files)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_documents(self) -> List[Dict[str, Any]]:
        try:
            response = requests.get(f"{self.backend_url}/documents")
            return response.json().get("documents", [])
        except:
            return []
    
    def delete_document(self, document_id: str) -> bool:
        try:
            response = requests.delete(f"{self.backend_url}/documents/{document_id}")
            return response.status_code == 200
        except:
            return False
    
    def chat_with_documents(self, question: str, document_ids: Optional[List[str]] = None, chat_history: Optional[List[Dict]] = None, language: str = "en") -> Dict[str, Any]:
        try:
            payload = {
                "question": question,
                "document_ids": document_ids,
                "chat_history": chat_history or [],
                "max_results": 5,
                "language": language
            }
            response = requests.post(f"{self.backend_url}/chat", json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    app = DocumentSearchApp()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "language" not in st.session_state:
        st.session_state.language = "en"
    
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    language = st.selectbox(
        "",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: f"Response Language: {LANGUAGES[x]}",
        index=list(LANGUAGES.keys()).index(st.session_state.language),
        key="language_selector"
    )
    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>Search on Documents</h1>
        <p>Chat with your documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not app.check_backend_connection():
        st.error("Backend connection failed. Please ensure the backend service is running.")
        st.code("python start_app.py")
        return
    
    with st.sidebar:
        st.header("Upload Documents")
        
        uploaded_file = st.file_uploader(
            "",
            type=["pdf", "docx", "txt", "md"],
            help="Select PDF, DOCX, TXT or MD file"
        )
        
        if uploaded_file is not None:
            if st.button("Upload & Process"):
                with st.spinner("Processing document..."):
                    result = app.upload_document(uploaded_file)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(f"{uploaded_file.name} uploaded successfully!")
                        st.rerun()
        
        st.divider()
        
        st.subheader("Uploaded Documents")
        documents = app.get_documents()
        st.session_state.documents = documents
        
        if documents:
            for doc in documents:
                with st.expander(f"{doc['filename']}", expanded=False):
                    st.write(f"**ID:** {doc['document_id'][:8]}...")
                    st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    st.write(f"**Date:** {doc['upload_date'][:10]}")
                    
                    if st.button("Delete", key=f"delete_{doc['document_id']}"):
                        if app.delete_document(doc['document_id']):
                            st.success("Document deleted!")
                            st.rerun()
                        else:
                            st.error("Delete failed!")
        else:
            st.info("No documents uploaded yet")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Chat Interface")
        
        with col2:
            if documents:
                selected_docs = st.multiselect(
                    "Search in specific documents",
                    options=[doc['document_id'] for doc in documents],
                    format_func=lambda x: next(doc['filename'] for doc in documents if doc['document_id'] == x),
                    help="Leave empty to search all documents"
                )
            else:
                selected_docs = []
        
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Clean HTML tags from message content for display
                    import html
                    clean_message = html.unescape(message["content"])
                    clean_message = re.sub(r'<[^>]+>', '', clean_message)
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong><br>
                        {clean_message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "sources" in message and message["sources"]:
                        with st.expander(f"Sources ({len(message['sources'])} found)"):
                            for j, source in enumerate(message["sources"], 1):
                                page_info = f" - Page {source['page_number']}" if source.get('page_number') else ""
                                
                                clean_source = app._clean_source_content(source['content'])
                                
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {j}: {source['document_name']}{page_info}</strong><br>
                                    <small>Similarity: {source['similarity_score']:.2%}</small><br><br>
                                    {clean_source[:300]}...
                                </div>
                                """, unsafe_allow_html=True)
        
        st.divider()
        
        if not documents:
            st.warning("Upload documents to start asking questions.")
        else:
            with st.form(key="chat_form", clear_on_submit=True):
                question =             st.text_area(
                "Question",
                placeholder="Ask a question about your documents...",
                height=100,
                key="question_input",
                label_visibility="collapsed"
            )
                
                col_send, col_clear = st.columns([5, 1])
                
                with col_send:
                    submitted = st.form_submit_button("Send", type="primary", use_container_width=True)
                
                with col_clear:
                    clear_clicked = st.form_submit_button("Clear", use_container_width=True)
                
                if clear_clicked:
                    st.session_state.chat_history = []
                    st.rerun()
                
                if submitted and question.strip():
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    with st.spinner("Processing..."):
                        api_history = [
                            {"role": msg["role"], "content": msg["content"]} 
                            for msg in st.session_state.chat_history[:-1]
                        ]
                        
                        response = app.chat_with_documents(
                            question=question,
                            document_ids=selected_docs if selected_docs else None,
                            chat_history=api_history,
                            language=st.session_state.language
                        )
                        
                        if "error" in response:
                            st.error(f"Error: {response['error']}")
                        else:
                            assistant_message = {
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": response.get("sources", []),
                                "timestamp": datetime.now().isoformat()
                            }
                            st.session_state.chat_history.append(assistant_message)
                            
                            st.rerun()
        
        if st.session_state.chat_history:
            st.divider()
            user_messages = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
            st.caption(f"Questions asked: {len(user_messages)}")

if __name__ == "__main__":
    main()