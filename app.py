import sqlite3
import chromadb
import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader
from langchain_ollama import ChatOllama

st.set_page_config(
    page_title="Enterprise AI Data & Document Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Connect to ChromaDB Vector Store and Local LLM (Ollama)
chroma_client = chromadb.PersistentClient(path="vector_store")
schema_collection = chroma_client.get_or_create_collection(name="schema_memory")
pdf_collection = chroma_client.get_or_create_collection(name="pdf_memory")
llm = ChatOllama(model="gemma3:1b", temperature=0)

# 2. Sidebar: Upload CSV or PDF
st.sidebar.title("🎛️ Data & Document Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV or PDF", type=["csv", "pdf"])

# 3. Clean Welcome State Before Any File Is Uploaded
if uploaded_file is None:
    st.title("📈 Enterprise AI Data & Document Analyst")
    st.caption("Hybrid Agentic RAG • Self-Correcting Text-to-SQL • Automated BI Visuals • Local Gemma 3")
    st.divider()

    with st.container(border=True):
        st.subheader("👋 Ready to Analyze Your Data or Documents")
        st.write("Upload a **CSV spreadsheet** or a **PDF report** in the left sidebar to get started.")
        c1, c2 = st.columns(2)
        with c1:
            st.info("📊 **CSV Mode:** Automatically builds a SQLite database, indexes table schemas in ChromaDB, generates self-correcting SQL, and renders interactive Plotly charts.")
        with c2:
            st.info("📄 **PDF Mode:** Parses document pages, creates semantic vector embeddings in ChromaDB, and answers your questions with exact page citations.")
    st.stop()

# 4. Process Uploaded CSV or PDF File
file_ext = uploaded_file.name.split(".")[-1].lower()

if file_ext == "csv":
    st.session_state.mode = "SQL"
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        with st.sidebar.status("Indexing CSV into SQLite & Schema RAG..."):
            custom_df = pd.read_csv(uploaded_file)
            custom_df.columns = [col.strip().lower().replace(" ", "_").replace("-", "_") for col in custom_df.columns]

            conn_up = sqlite3.connect("data/company_data.db")
            custom_df.to_sql("custom_data", conn_up, if_exists="replace", index=False)
            conn_up.close()

            cols_str = ", ".join(custom_df.columns)
            sample_preview = custom_df.head(2).to_dict(orient="records")
            new_doc = f"Custom uploaded table named 'custom_data' with {len(custom_df)} rows. Available columns: {cols_str}. Sample rows: {sample_preview}"
            new_rule = f"Always query the table named 'custom_data' using ONLY these exact column names: {cols_str}."

            schema_collection.upsert(
                documents=[new_doc, new_rule],
                ids=["custom_table_schema", "custom_table_rules"]
            )

            st.session_state.active_table = "custom_data"
            st.session_state.last_uploaded_name = uploaded_file.name
            col_list = list(custom_df.columns)
            first_col = col_list.pop(1) if len(col_list) > 1 else col_list.pop(0)
            st.session_state.active_prompt = f"Show {first_col} and count of rows grouped by {first_col} limit 10"
        st.sidebar.success(f"CSV Loaded: '{uploaded_file.name}'")

elif file_ext == "pdf":
    st.session_state.mode = "PDF"
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        with st.sidebar.status("Reading & indexing PDF into ChromaDB..."):
            reader = PdfReader(uploaded_file)
            chunks = []
            chunk_ids = []
            chunk_metas = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                for start in range(0, len(page_text), 800):
                    piece = page_text[start:start + 900].strip()
                    if len(piece) > 40:
                        chunks.append(piece)
                        chunk_ids.append(f"page_{page_num + 1}_part_{start}")
                        chunk_metas.append({"page": page_num + 1, "source": uploaded_file.name})

            try:
                chroma_client.delete_collection(name="pdf_memory")
            except Exception:
                pass
            pdf_collection = chroma_client.create_collection(name="pdf_memory")

            if chunks:
                pdf_collection.add(documents=chunks, ids=chunk_ids, metadatas=chunk_metas)

            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.pdf_pages = len(reader.pages)
            st.session_state.pdf_chunks = len(chunks)
            st.session_state.active_prompt = "Summarize the key findings and main topics in this document."
        st.sidebar.success(f"PDF Indexed: '{uploaded_file.name}' ({len(chunks)} chunks)")

# ==============================================================================
# MODE A: STRUCTURED DATA ANALYST (CSV / SQLITE + CHARTS)
# ==============================================================================
if st.session_state.mode == "SQL":
    current_table = st.session_state.active_table
    conn_kpi = sqlite3.connect("data/company_data.db")
    preview_df = pd.read_sql_query(f"SELECT * FROM {current_table}", conn_kpi)
    conn_kpi.close()

    total_rows = len(preview_df)
    total_cols = len(preview_df.columns)
    numeric_cols = list(preview_df.select_dtypes(include="number").columns)
    primary_num = numeric_cols[-1] if numeric_cols else "N/A"
    avg_metric = round(float(preview_df[primary_num].mean()), 2) if numeric_cols else 0
    max_metric = round(float(preview_df[primary_num].max()), 2) if numeric_cols else 0

    st.title("📈 Enterprise AI Data & Document Analyst")
    st.caption(f"Mode: **Structured SQL Analytics** • File: **{st.session_state.get('last_uploaded_name', '')}** • Local Gemma 3")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric(label="Total Records", value=f"{total_rows:,}", delta=f"Table: {current_table}")
    with m2:
        with st.container(border=True):
            st.metric(label="Total Columns", value=f"{total_cols}", delta="Schema Indexed")
    with m3:
        with st.container(border=True):
            st.metric(label=f"Avg ({primary_num})", value=f"{avg_metric:,}", delta="Dataset Mean")
    with m4:
        with st.container(border=True):
            st.metric(label=f"Max ({primary_num})", value=f"{max_metric:,}", delta="Peak Value")

    st.divider()

    st.sidebar.divider()
    chart_style = st.sidebar.selectbox("Visualization Style", ["Gradient Bar Chart", "Smooth Line Trend", "Area Chart"])

    with st.container(border=True):
        question = st.text_input("💬 Ask your dataset a question in plain English:", value=st.session_state.active_prompt)
        run_clicked = st.button("🚀 Analyze Data", type="primary")

    if run_clicked and question.strip():
        with st.status("Running Autonomous SQL Pipeline...", expanded=True) as status:
            st.write("🔍 Step 1: Retrieving active table schema from ChromaDB RAG...")
            rag_results = schema_collection.get(ids=["custom_table_schema", "custom_table_rules"])
            flat_docs = rag_results.get("documents", [])
            schema_context = "\n".join(flat_docs)

            st.write("🧠 Step 2: Generating & validating SQLite query with Gemma 3...")
            conn = sqlite3.connect("data/company_data.db")
            error_message = ""
            sql_query = ""
            result_df = pd.DataFrame()
            success = False
            exact_cols = ", ".join(preview_df.columns)

            for attempt in range(1, 4):
                prompt = f"""You are a SQLite expert. Write ONLY a valid SQLite SELECT query to answer the user question.
Rules:
- Output ONLY raw SQL. No markdown, no explanations, no ```sql tags.
- Query ONLY the table named '{current_table}'.
- ONLY use column names from this exact list: [{exact_cols}]. NEVER invent or use any column name outside this list!
- Select at least 2 columns (for example, a text/name column and a numeric column) so a chart can be drawn.

Database Schema & Rules:
{schema_context}

User Question: {question}
"""
                if error_message:
                    prompt += f"\nPrevious SQL failed with error: {error_message}\nRemember ONLY use columns from [{exact_cols}]. Fix the SQL query and output ONLY raw SQL:"

                response = llm.invoke(prompt)
                sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()

                try:
                    result_df = pd.read_sql_query(sql_query, conn)
                    if len(result_df.columns) == 1 and len(result_df) > 1:
                        only_col = list(result_df.columns).pop(0)
                        sql_query = f"SELECT {only_col}, COUNT(*) AS record_count FROM {current_table} GROUP BY {only_col} ORDER BY record_count DESC LIMIT 10"
                        result_df = pd.read_sql_query(sql_query, conn)
                    success = True
                    st.write(f"✅ Step 3: Query succeeded on attempt {attempt} ({len(result_df)} rows).")
                    break
                except Exception as e:
                    error_message = str(e)
                    st.write(f"⚠️ Attempt {attempt} failed, auto-fixing SQL...")


            conn.close()
            status.update(label="Analysis Complete", state="complete", expanded=False)

        if success and not result_df.empty:
            with st.spinner("Writing executive summary..."):
                summary_prompt = f"Write a simple 2-sentence summary of this data:\nQuestion: {question}\nData:\n{result_df.head(10).to_string(index=False)}"
                summary = llm.invoke(summary_prompt)

            with st.container(border=True):
                st.subheader("💡 Executive AI Insight")
                st.write(summary.content.strip())

            chart_col, table_col = st.columns(2)
            with chart_col:
                with st.container(border=True):
                    st.subheader("📊 Visual Analytics")
                    col_names = list(result_df.columns)
                    if len(col_names) >= 2 and len(result_df) > 1:
                        x_col = col_names.pop(0)
                        y_col = col_names.pop(0)

                        if chart_style == "Gradient Bar Chart":
                            fig = px.bar(result_df, x=x_col, y=y_col, color=y_col, color_continuous_scale="tealgrn", template="plotly_dark")
                            fig.update_xaxes(type="category")

                        elif chart_style == "Smooth Line Trend":
                            fig = px.line(result_df, x=x_col, y=y_col, markers=True, template="plotly_dark")
                            fig.update_traces(line_color="#00D4FF", line_width=3)
                        else:
                            fig = px.area(result_df, x=x_col, y=y_col, template="plotly_dark")
                            fig.update_traces(line_color="#00D4FF")

                        fig.update_layout(
                            title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=380
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Single column returned — view the table on the right.")

            with table_col:
                with st.container(border=True):
                    st.subheader("📋 Result Dataset")
                    st.dataframe(result_df, use_container_width=True, height=320)
                    csv_data = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Export CSV Report",
                        data=csv_data,
                        file_name="ai_analysis_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with st.expander("🛠️ Technical Audit Trail (SQL & Schema RAG)"):
                st.markdown("**Generated SQLite Query**")
                st.code(sql_query, language="sql")
                st.markdown("**ChromaDB Retrieved Schema**")
                st.json(flat_docs)

        elif success and result_df.empty:
            st.warning("Query ran cleanly, but returned 0 rows.")
            st.code(sql_query, language="sql")
        else:
            st.error(f"Could not generate valid SQL after 3 tries. Last error: {error_message}")
            st.code(sql_query, language="sql")

# ==============================================================================
# MODE B: UNSTRUCTURED PDF DOCUMENT ANALYST (DOCUMENT RAG + CITATIONS)
# ==============================================================================
elif st.session_state.mode == "PDF":
    st.title("📄 Enterprise AI Document & Data Analyst")
    st.caption(f"Mode: **PDF Document RAG** • File: **{st.session_state.get('last_uploaded_name', '')}** • Local Gemma 3")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        with st.container(border=True):
            st.metric(label="Document Pages", value=f"{st.session_state.get('pdf_pages', 0)}", delta="PDF Parsed")
    with p2:
        with st.container(border=True):
            st.metric(label="Indexed Chunks", value=f"{st.session_state.get('pdf_chunks', 0)}", delta="ChromaDB Ready")
    with p3:
        with st.container(border=True):
            st.metric(label="Retrieval Top-K", value="3 Chunks", delta="Semantic Search")
    with p4:
        with st.container(border=True):
            st.metric(label="Active AI Engine", value="Gemma 3 (1B)", delta="100% Local")

    st.divider()

    with st.container(border=True):
        pdf_question = st.text_input("💬 Ask any question about your uploaded PDF:", value=st.session_state.active_prompt)
        ask_pdf_clicked = st.button("🔍 Search & Answer from PDF", type="primary")

    if ask_pdf_clicked and pdf_question.strip():
        with st.spinner("Searching PDF chunks in ChromaDB & generating cited answer..."):
            search_res = pdf_collection.query(query_texts=[pdf_question], n_results=3)
            raw_docs = search_res.get("documents", [])
            raw_metas = search_res.get("metadatas", [])
            retrieved_chunks = raw_docs.pop(0) if raw_docs else []
            retrieved_metas = raw_metas.pop(0) if raw_metas else []

            context_block = ""
            for idx, text_chunk in enumerate(retrieved_chunks):
                page_no = retrieved_metas[idx].get("page", "?") if idx < len(retrieved_metas) else "?"
                context_block += f"\n[Page {page_no}]: {text_chunk}\n"

            rag_prompt = f"""You are a helpful enterprise document analyst. Answer the user's question using ONLY the provided PDF excerpts below.
If the answer is not in the excerpts, say "I cannot find that information in the uploaded PDF."
Always mention the page number(s) where you found the information.

PDF Excerpts:
{context_block}

User Question: {pdf_question}
Answer:"""
            ai_answer = llm.invoke(rag_prompt)

        with st.container(border=True):
            st.subheader("💡 AI Document Answer")
            st.write(ai_answer.content.strip())

        with st.expander("📚 Retrieved Source Passages & Page Citations"):
            for idx, text_chunk in enumerate(retrieved_chunks):
                page_no = retrieved_metas[idx].get("page", "?") if idx < len(retrieved_metas) else "?"
                st.markdown(f"**Source {idx + 1} (Page {page_no}):**")
                st.info(text_chunk)

