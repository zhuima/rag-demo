import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 页面配置
st.set_page_config(
    page_title="DeepSeek RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS，优化界面布局
st.markdown("""
    <style>
    /* 整体布局 */
    .stApp {
        background-color: #ffffff;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: #f7f7f8;
    }
    
    /* 主容器样式 */
    .main-container {
        flex: 1;
        overflow-y: auto;
        padding: 1rem 2rem 100px;  /* 底部留出输入框空间 */
    }
    
    /* 聊天区域样式 */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* 聊天消息样式 */
    .chat-message {
        padding: 1.5rem 2rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        animation: fadeIn 0.3s;
        line-height: 1.6;
    }
    
    .user-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
    }
    
    .assistant-message {
        background-color: #f7f7f8;
    }
    
    /* 输入区域样式 */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: #ffffff;
        border-top: 1px solid #e5e5e5;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        padding: 12px 20px;
        border-radius: 8px;
        border: 1px solid #e5e5e5;
        font-size: 16px;
        width: 100%;
    }
    
    /* 发送按钮样式 */
    .stButton > button {
        background-color: #0066ff;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        height: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0052cc;
    }
    
    /* 文件上传区域 */
    .upload-container {
        padding: 20px;
        border-radius: 8px;
        border: 2px dashed #ccc;
        margin: 20px 0;
        text-align: center;
    }
    
    /* 动画效果 */
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    
    /* 隐藏Streamlit默认的页脚 */
    footer {display: none !important;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 主要内容区域 */
    .main {
        margin-bottom: 80px !important;  /* 为底部输入框留出空间 */
    }
    
    /* 输入表单样式 */
    .input-form {
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        gap: 10px;
    }
    
    /* 使用说明样式 */
    .instructions {
        padding: 1rem;
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# 侧边栏
with st.sidebar:
    st.title("DeepSeek RAG Chat")
    st.markdown("---")
    
    # 文件上传区域
    st.markdown("### 📄 上传PDF文件")
    uploaded_file = st.file_uploader("", type="pdf")
    
    # 使用说明
    # 优化注意事项显示
    st.markdown("---")
    with st.expander("ℹ️ 使用说明"):
        st.markdown("""
        - 📚 本应用使用 DeepSeek R1 7b 模型（由Ollama提供服务）实现RAG（检索增强生成）管道
        - ⚙️ 使用前请确保：
            1. 已安装 Ollama
            2. 已拉取 deepseek-r1:7b 模型 (`ollama pull deepseek-r1:7b`)
            3. Ollama 服务正在后台运行
        - 🔧 技术细节：
            - 使用语义分块进行更好的上下文检索
            - 默认使用 Ollama API 端点 (`http://localhost:11434/api/generate`)
            - 适用于中小型PDF文档处理
        """)

# 主界面
if uploaded_file:
    try:
        if st.session_state.qa_chain is None:
            with st.spinner("正在处理文档..."):
                # 保存和处理PDF
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # 初始化QA链
                loader = PDFPlumberLoader("temp.pdf")
                docs = loader.load()
                
                # 更新 HuggingFaceEmbeddings 的初始化
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                text_splitter = SemanticChunker(embeddings)
                documents = text_splitter.split_documents(docs)
                vector_store = FAISS.from_documents(documents, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                llm = OllamaLLM(model="deepseek-r1:32b", base_url="http://127.0.0.1:11434")
                
                # 设置提示模板
                prompt_template = """
                你是一位文本大纲生成专家，擅长根据用户的需求创建一个有条理且易于扩展成完整文章的大纲，你拥有强大的主题分析能力，能准确提取关键信息和核心要点。具备丰富的文案写作知识储备，熟悉各种文体和题材的文案大纲构建方法。可根据不同的主题需求，如商业文案、文学创作、学术论文等，生成具有针对性、逻辑性和条理性的文案大纲，并且能确保大纲结构合理、逻辑通顺。该大纲应该包含以下部分：\n引言：介绍主题背景，阐述撰写目的，并吸引读者兴趣。\n主体部分：第一段落：详细说明第一个关键点或论据，支持观点并引用相关数据或案例。\n第二段落：深入探讨第二个重点，继续论证或展开叙述，保持内容的连贯性和深度。\n第三段落：如果有必要，进一步讨论其他重要方面，或者提供不同的视角和证据。\n结论：总结所有要点，重申主要观点，并给出有力的结尾陈述，可以是呼吁行动、提出展望或其他形式的收尾。\n创意性标题：为文章构思一个引人注目的标题，确保它既反映了文章的核心内容又能激发读者的好奇心。
                
                上下文: {context}
                问题: {question}
                答案:"""
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
                )
                st.session_state.qa_chain = qa_chain

    except Exception as e:
        st.error(f"❌ 错误: {str(e)}")
        st.warning("请确保Ollama服务正在运行且模型已安装")

# 创建主要内容区域
main_content = st.container()
main_content.markdown('<div class="main-container">', unsafe_allow_html=True)

# 显示主要内容
with main_content:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    # 显示聊天历史
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <b>问题：</b><br>{message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <b>DeepSeek：</b><br>{message["content"]}
                </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
main_content.markdown('</div>', unsafe_allow_html=True)

# 创建底部输入区域
bottom_input = st.container()

# 底部输入区域
with bottom_input:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="input-form">', unsafe_allow_html=True)
    
    if st.session_state.qa_chain is not None:
        # 处理输入框重置
        if 'clear_input' not in st.session_state:
            st.session_state.clear_input = False
        
        if st.session_state.clear_input:
            st.session_state.user_question = ''
            st.session_state.clear_input = False
        
        # 创建输入表单
        with st.form(key="question_form", clear_on_submit=True):
            cols = st.columns([8, 2])
            with cols[0]:
                user_input = st.text_input(
                    label="问题输入",
                    label_visibility="collapsed",
                    placeholder="输入您的问题...",
                    key="user_question"
                )
            with cols[1]:
                submit_button = st.form_submit_button(
                    label="发送",
                    use_container_width=True
                )
            
            if submit_button and user_input:
                # 添加用户问题到历史
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                try:
                    with st.spinner("🤔 思考中..."):
                        response = st.session_state.qa_chain(user_input)
                        answer = response["result"]
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    st.session_state.clear_input = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"生成回答时出错: {str(e)}")
    else:
        st.info("👈 请先在左侧上传PDF文件")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)