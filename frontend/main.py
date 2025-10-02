import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import re
import html

st.set_page_config(
    page_title="Chat With Your Document",
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
    
    def search_images(self, query: str, document_ids: Optional[List[str]] = None, limit: int = 5) -> Dict[str, Any]:
        try:
            params = {
                "query": query,
                "limit": limit
            }
            if document_ids:
                params["document_ids"] = ",".join(document_ids)
            
            response = requests.get(f"{self.backend_url}/search-images", params=params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_document_images(self, document_id: str) -> Dict[str, Any]:
        """Get all images for a specific document"""
        try:
            response = requests.get(f"{self.backend_url}/documents/{document_id}/images")
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
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "Chat"
    
    with st.container():
        st.markdown('<div class="language-selector">', unsafe_allow_html=True)
        language = st.selectbox(
            "",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: f"Response Language: {LANGUAGES[x]}",
            index=list(LANGUAGES.keys()).index(st.session_state.language),
            key="language_selector"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if language != st.session_state.language:
            st.session_state.language = language
            st.rerun()
    
    st.markdown("""
    <div class="main-header">
        <h1>Chat With Your Document</h1>
        <p>Ask questions about your documents and see related images</p>
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
    
    render_chat_interface(app, documents)

def render_chat_interface(app, documents):
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if documents:
            selected_docs = st.multiselect(
                "Limit chat to specific documents",
                options=[doc['document_id'] for doc in documents],
                format_func=lambda x: next(doc['filename'] for doc in documents if doc['document_id'] == x),
                help="Leave empty to use all documents"
            )
        else:
            selected_docs = []
    
    with col1:
        st.header("Chat Interface")
        
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>
                        {html.escape(message["content"])}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    clean_message = html.unescape(message["content"])
                    clean_message = re.sub(r'<[^>]+>', '', clean_message)
                    clean_message = html.escape(clean_message)
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong><br>
                        {clean_message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "images" in message and message["images"]:
                        st.markdown("**Related Images:**")
                        cols = st.columns(min(3, len(message["images"])))
                        for idx, image_info in enumerate(message["images"][:3]):
                            with cols[idx % 3]:
                                try:
                                    image_path = Path(image_info["image_path"])
                                    image_filename = image_path.name
                                    image_url = f"{app.backend_url}/images/{image_info['document_id']}/{image_filename}"
                                    
                                    st.image(
                                        image_url,
                                        caption=f"{image_info['document_name']} - Page {image_info.get('page_number', 'N/A')}",
                                        use_column_width=True
                                    )
                                    
                                    similarity_percent = image_info['similarity_score'] * 100
                                    st.caption(f"Similarity: {similarity_percent:.1f}%")
                                    
                                except Exception as e:
                                    st.error(f"Error loading image: {str(e)}")
                        
                        if len(message["images"]) > 3:
                            with st.expander(f"Show {len(message['images']) - 3} more images"):
                                remaining_cols = st.columns(3)
                                for idx, image_info in enumerate(message["images"][3:]):
                                    with remaining_cols[idx % 3]:
                                        try:
                                            image_path = Path(image_info["image_path"])
                                            image_filename = image_path.name
                                            image_url = f"{app.backend_url}/images/{image_info['document_id']}/{image_filename}"
                                            
                                            st.image(
                                                image_url,
                                                caption=f"{image_info['document_name']} - Page {image_info.get('page_number', 'N/A')}",
                                                use_column_width=True
                                            )
                                            
                                            similarity_percent = image_info['similarity_score'] * 100
                                            st.caption(f"Similarity: {similarity_percent:.1f}%")
                                            
                                        except Exception as e:
                                            st.error(f"Error loading image: {str(e)}")
                    
                    if "sources" in message and message["sources"]:
                        with st.expander(f"Text Sources ({len(message['sources'])} found)"):
                            for j, source in enumerate(message["sources"], 1):
                                page_info = f" - Page {source['page_number']}" if source.get('page_number', '') else ""
                                content_preview = html.escape(source['content'][:300])
                                document_name = html.escape(source['document_name'])
                                
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {j}: {document_name}{page_info}</strong><br>
                                    <small>Similarity: {source['similarity_score']:.2%}</small><br><br>
                                    {content_preview}...
                                </div>
                                """, unsafe_allow_html=True)
        
        st.divider()
        
        if not documents:
            st.warning("Upload documents to start asking questions.")
        else:
            with st.form(key="chat_form", clear_on_submit=True):
                question = st.text_area(
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
                            # Search for related images
                            images = []
                            try:
                                image_result = app.search_images(
                                    query=question,
                                    document_ids=selected_docs if selected_docs else None,
                                    limit=5
                                )
                                if "images" in image_result and not "error" in image_result:
                                    images = image_result["images"]
                            except Exception as e:
                                print(f"Image search failed: {str(e)}")
                            
                            assistant_message = {
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": response.get("sources", []),
                                "images": images,
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