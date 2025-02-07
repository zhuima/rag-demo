# DeepSeek RAG Chat 🤖

基于DeepSeek R1模型构建的智能文档问答系统，支持PDF文档解析与语义问答

![Demo Screenshot](https://via.placeholder.com/800x500.png/0077ff/ffffff?text=RAG+Chat+Demo)

## 主要特性 ✨
- 📄 PDF文档智能解析与语义分块
- 🔍 基于FAISS的向量相似度检索
- 💬 对话式交互界面（支持上下文记忆）
- 🚀 使用Ollama本地部署DeepSeek R1模型
- 🎨 响应式布局设计（适配桌面/移动端）

## 技术栈 ⚙️
- **大语言模型**: DeepSeek R1 32B (via Ollama)
- **框架**: Streamlit + LangChain
- **向量数据库**: FAISS
- **文本嵌入**: Sentence-Transformers
- **PDF解析**: PDFPlumber

## 快速开始 🚀

### 前置要求
- Python 3.9+
- Ollama 已安装并运行
- 已拉取DeepSeek模型：
  ```bash
  ollama pull deepseek-r1:32b
  ```

### 安装步骤
1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/deepseek-rag-chat.git
   cd deepseek-rag-chat
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 启动应用：
   ```bash
   streamlit run main.py
   ```

## 使用指南 📖
1. 在左侧边栏上传PDF文档
2. 等待文档处理完成（首次需要1-2分钟）
3. 在底部输入框提问，例如：
   - "请总结本文的核心观点"
   - "第三章主要讨论了哪些内容？"
   - "列出文中提到的关键技术点"

## 配置说明 ⚙️
| 组件 | 默认配置 | 可调整参数 |
|------|---------|-----------|
| Ollama | `base_url="http://127.0.0.1:11434"` | 端口号/服务器地址 |
| 向量检索 | `k=3` 个上下文块 | 检索数量/相似度阈值 |
| 文本分块 | 语义分块 | 块大小/重叠比例 |
| 模型参数 | `temperature=0.7` | 创造性/确定性平衡 |

## 项目结构 📂 

```python
.
├── main.py # 主应用程序
├── requirements.txt # 依赖列表
├── README.md # 说明文档
```

## 常见问题 ❓
**Q: 出现`ConnectionError`怎么办？**  
A: 确保：
1. Ollama服务正在运行
2. 模型已正确下载 (`ollama list`)
3. 防火墙允许11434端口通信

**Q: 如何处理大文档？**  
- 使用性能更强的GPU设备
- 调整`search_kwargs={"k": 2}`减少检索量
- 增加系统内存（推荐≥16GB）

## 开发路线图 🗺️
- [ ] 多文档支持
- [ ] 对话历史导出
- [ ] 自动文档摘要
- [ ] 混合检索策略

## 许可证 📜
MIT License © 2024 [Your Name]



