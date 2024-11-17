import streamlit as st
import pandas as pd
import hnswlib
import numpy as np
from io import BytesIO
import base64
import openai

# 設置頁面配置
st.set_page_config(page_title="多重代理分析系統", layout="wide")

# 設置 OpenAI API 金鑰
# 嘗試從 secrets 中獲取，如果沒有，則允許用戶手動輸入
if "openai_api_key" not in st.session_state:
    if "openai" in st.secrets:
        st.session_state.openai_api_key = st.secrets["openai"]["api_key"]
    else:
        st.session_state.openai_api_key = ""

if not st.session_state.openai_api_key:
    st.sidebar.header("API 金鑰設置")
    st.session_state.openai_api_key = st.sidebar.text_input(
        "輸入您的 OpenAI API 金鑰",
        type="password",
        help="您可以將 API 金鑰添加到 Streamlit 的 secrets 中，或者在這裡手動輸入。"
    )

if not st.session_state.openai_api_key:
    st.warning("請在側邊欄輸入您的 OpenAI API 金鑰以繼續。")
    st.stop()
else:
    openai.api_key = st.session_state.openai_api_key

# 初始化或載入 session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'final_report' not in st.session_state:
    st.session_state.final_report = ""

# 定義代理類別
class Agent:
    def __init__(self, name, role, index, documents):
        self.name = name
        self.role = role
        self.messages = []
        self.index = index
        self.documents = documents

    def add_message(self, message):
        if not self.messages:
            self.messages.append({"role": "system", "content": self.role})
        self.messages.append({"role": "user", "content": message})

    def retrieve_relevant_docs(self, query, top_k=5):
        # 使用 OpenAI 的嵌入模型來生成 query 向量
        response = openai.Embedding.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        query_vector = np.array(response['data'][0]['embedding']).astype('float32')
        labels, distances = self.index.knn_query(query_vector, k=top_k)
        retrieved_docs = [self.documents[i] for i in labels[0]]
        return retrieved_docs

    def get_response(self):
        try:
            user_message = self.messages[-1]['content']
            retrieved_docs = self.retrieve_relevant_docs(user_message)
            context = "\n".join(retrieved_docs)
            augmented_messages = self.messages.copy()
            augmented_messages.append({"role": "system", "content": f"相關資料如下：\n{context}"})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=augmented_messages,
                max_tokens=150,
                temperature=0.7,
                n=1,
                stop=None,
            )
            reply = response['choices'][0]['message']['content'].strip()
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            st.error(f"與代理 {self.name} 交互時出錯：{e}")
            return ""

# 建立或更新 hnswlib 索引
def build_hnsw_index(documents, dim=1536, max_elements=10000):
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=max_elements, ef_construction=200, M=16)
    
    # 使用 OpenAI 的嵌入模型來生成向量
    response = openai.Embedding.create(
        input=documents,
        model="text-embedding-ada-002"
    )
    vectors = np.array([record['embedding'] for record in response['data']]).astype('float32')
    labels = np.arange(len(documents))
    
    p.add_items(vectors, labels)
    p.set_ef(50)  # ef should always be > top_k
    return p

# 下載報告為 TXT 或 PDF
def get_download_link(report, filename, filetype='txt'):
    if filetype == 'txt':
        b64 = base64.b64encode(report.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">下載報告（TXT）</a>'
    elif filetype == 'pdf':
        # 使用 ReportLab 生成 PDF
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        p.setFont("Helvetica", 12)
        
        # 分行寫入
        lines = report.split('\n')
        y = height - 40
        for line in lines:
            p.drawString(40, y, line)
            y -= 15
            if y < 40:
                p.showPage()
                p.setFont("Helvetica", 12)
                y = height - 40
        p.save()
        buffer.seek(0)
        pdf = buffer.getvalue()
        b64 = base64.b64encode(pdf).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">下載報告（PDF）</a>'
    return href

# 上傳 CSV 文件
st.sidebar.header("上傳 CSV 文件")
uploaded_files = st.sidebar.file_uploader("選擇 CSV 文件（可多選）", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write(f"文件名：{uploaded_file.name}")
            st.sidebar.dataframe(df.head())
            # 將每行轉換為字符串
            documents = df.apply(lambda row: row.to_string(index=False), axis=1).tolist()
            st.session_state.documents.extend(documents)
        except Exception as e:
            st.sidebar.error(f"無法讀取文件 {uploaded_file.name}：{e}")
    
    # 建立 hnswlib 索引
    if st.sidebar.button("建立索引"):
        if st.session_state.documents:
            with st.spinner("正在建立索引..."):
                try:
                    st.session_state.index = build_hnsw_index(st.session_state.documents)
                    st.sidebar.success("索引建立完成！")
                except Exception as e:
                    st.sidebar.error(f"建立索引

# 提問區域
st.header("提問並生成分析報告")

if st.session_state.index is None:
    st.warning("請先上傳 CSV 文件並建立索引。")
else:
    question = st.text_input("請輸入您的問題：")
    if st.button("提交問題"):
        if question:
            # 定義代理
            agents = [
                Agent("分析師A", "您是一位數據分析師，專長於數據解讀和統計分析。", st.session_state.index, st.session_state.documents),
                Agent("分析師B", "您是一位業務分析師，專注於業務流程和市場趨勢。", st.session_state.index, st.session_state.documents),
                Agent("分析師C", "您是一位技術分析師，擅長技術指標和數據可視化。", st.session_state.index, st.session_state.documents)
            ]
            
            # 清空之前的消息
            st.session_state.messages = []
            
            # 用戶問題添加到第一個代理
            agents[0].add_message(question)
            st.session_state.messages.append({"role": "user", "content": question})
            
            # 顯示用戶消息
            st.write(f"**用戶**: {question}")
            
            # 代理 A 回應
            response_a = agents[0].get_response()
            st.session_state.messages.append({"role": "assistant", "content": response_a})
            st.write(f"**{agents[0].name}**: {response_a}")
            
            # 進行多輪討論
            for round_num in range(1, 4):
                for agent in agents:
                    context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    prompt = f"討論輪次 {round_num}，基於以下討論內容：\n{context}\n請提供您的回應。"
                    agent.add_message(prompt)
                    reply = agent.get_response()
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.write(f"**{agent.name}**: {reply}")
            
            # 生成最終報告
            final_discussion = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            summary_prompt = f"基於以下討論內容，請總結出一份完整的分析報告：\n{final_discussion}"
            summary_agent = Agent("總結者", "您是一位經驗豐富的報告撰寫者，負責總結討論內容並生成最終報告。", st.session_state.index, st.session_state.documents)
            summary_agent.add_message(summary_prompt)
            final_report = summary_agent.get_response()
            st.session_state.messages.append({"role": "assistant", "content": final_report})
            
            # 顯示最終報告
            st.subheader("最終分析報告")
            st.write(final_report)
            
            # 提供下載選項
            st.markdown(get_download_link(final_report, "final_report.txt"), unsafe_allow_html=True)
            st.markdown(get_download_link(final_report, "final_report.pdf", filetype='pdf'), unsafe_allow_html=True)

# 顯示聊天記錄（可選）
st.header("聊天記錄")
if st.session_state.messages:
    for message in st.session_state.messages:
        role = "用戶" if message["role"] == "user" else message["role"]
        st.write(f"**{role}**: {message['content']}")
