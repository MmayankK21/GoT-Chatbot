import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import pandas as pd

try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score

    EVALUATION_ENABLED = True
except ImportError:
    EVALUATION_ENABLED = False
    st.warning("Evaluation packages not found. Install with: pip install rouge-score bert-score")

try:
    llm = Ollama(model="llama3.2:latest", temperature=0.3)
    embeddings = OllamaEmbeddings(model="llama3.2:latest")

    vector_db = Chroma(
        persist_directory="./got_db_llama3.2",
        embedding_function=embeddings
    )

    template = """Answer this Game of Thrones question based on the context:
    Context: {context}

    Question: {question}

    Answer in 2-3 sentences:"""
    QA_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    if EVALUATION_ENABLED:
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()


def evaluate_response(predicted, reference):
    """Calculate evaluation metrics if packages are available"""
    if not EVALUATION_ENABLED:
        return None

    try:
        rouge_scores = rouge.score(reference, predicted)

        P, R, F1 = bert_score([predicted], [reference], lang="en", verbose=False)

        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")
        return None


st.set_page_config(page_title="GoT Trivia", layout="wide")
st.title("Game of Thrones Trivia (llama3.2)")
st.write("Ask me anything about Westeros!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if EVALUATION_ENABLED:
    eval_tab, chat_tab = st.tabs(["Evaluation Report", "Chat"])
else:
    chat_tab = st.container()

with chat_tab:
    if prompt := st.chat_input("Your GoT question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting the Maesters..."):
                try:
                    result = qa_chain({"query": prompt})
                    response = result["result"]

                    sources = list({doc.metadata.get("source", "Wikipedia")
                                    for doc in result["source_documents"]})

                    st.markdown(response)
                    if sources:
                        st.caption(f"📚 Source: {', '.join(sources)}")

                    st.session_state.messages.append({"role": "assistant", "content": response})

                    if result["source_documents"] and EVALUATION_ENABLED:
                        reference = result["source_documents"][0].page_content
                        evaluation = evaluate_response(response, reference)
                        if evaluation:
                            st.session_state.evaluations.append(evaluation)

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if EVALUATION_ENABLED:
    with eval_tab:
        if st.session_state.evaluations:
            df = pd.DataFrame(st.session_state.evaluations)

            st.subheader("Evaluation Metrics")
            st.write("Average scores across all queries:")

            avg_scores = df.mean().to_dict()

            cols = st.columns(3)
            cols[0].metric("ROUGE-1 F1", f"{avg_scores['rouge1']:.3f}")
            cols[1].metric("ROUGE-2 F1", f"{avg_scores['rouge2']:.3f}")
            cols[2].metric("ROUGE-L F1", f"{avg_scores['rougeL']:.3f}")

            cols = st.columns(3)
            cols[0].metric("BERTScore Precision", f"{avg_scores['bertscore_precision']:.3f}")
            cols[1].metric("BERTScore Recall", f"{avg_scores['bertscore_recall']:.3f}")
            cols[2].metric("BERTScore F1", f"{avg_scores['bertscore_f1']:.3f}")

            st.subheader("Detailed Metrics")
            st.dataframe(df.style.highlight_max(axis=0))

            st.subheader("Performance Trends")
            st.line_chart(df)
        else:
            st.warning("No evaluation data yet. Ask some questions first!")