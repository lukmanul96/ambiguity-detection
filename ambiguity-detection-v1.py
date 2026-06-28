import streamlit as st

# This must be the very first Streamlit call
st.set_page_config(page_title="Ambiguity Detector")
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import chardet
import spacy
import matplotlib.pyplot as plt
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------- Initialization ----------
st.title("🧠 Ambiguity Detection & Recommendation System")
nlp = spacy.load("en_core_web_sm")

# ---------- Upload Section ----------
st.sidebar.header("📥 Upload Files")
uploaded_xml = st.sidebar.file_uploader("Upload XML Requirement Files", type=["xml"], accept_multiple_files=True)
uploaded_csv = st.sidebar.file_uploader("Upload CSV Requirements (Optional)", type="csv")
uploaded_terms = st.sidebar.file_uploader("Upload Ambiguous Terms CSV (Optional)", type="csv")

# ---------- Load Requirements ----------
def load_xml_requirements(files):
    requirements = []
    for f in files:
        raw = f.read()
        encoding = chardet.detect(raw)['encoding']
        f.seek(0)
        tree = ET.parse(f)
        root = tree.getroot()
        for text_body in root.findall(".//{req_document.xsd}text_body"):
            parent = text_body.find("..")
            req_id = parent.attrib.get("id", "N/A") if parent is not None else "Unknown"
            requirements.append({
                "id": req_id,
                "text_body": text_body.text.strip() if text_body.text else ""
            })
    return requirements

requirements = []
if uploaded_xml:
    requirements += load_xml_requirements(uploaded_xml)
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    requirements += [{"id": row['ID'], "text_body": row['Requirement']} for _, row in df.iterrows()]

# ---------- Ambiguous Term Setup ----------
def get_ambiguous_terms():
    default_terms = ['may', 'could', 'should', 'might', 'possibly', 'unclear', 'unsure', 'some', 'many', 'potential', 'approximately']
    if uploaded_terms:
        df_terms = pd.read_csv(uploaded_terms)
        cleaned_columns = [col.strip().lower() for col in df_terms.columns]
        if 'ambiguous term' in cleaned_columns:
            term_col = df_terms.columns[cleaned_columns.index('ambiguous term')]
        else:
            term_col = df_terms.columns[0]
        return default_terms + df_terms[term_col].dropna().astype(str).str.lower().tolist()
    return default_terms

ambiguous_terms = get_ambiguous_terms()

# ---------- UI Configuration ----------
st.sidebar.header("🔧 Tuning Parameters")
st.sidebar.slider("TF-IDF Similarity Threshold", 0.0, 1.0, 0.04, 0.01, key="threshold_slider")
st.sidebar.slider("Fuzzy Score Threshold (lower = more ambiguous)", 0.0, 1.0, 0.01, 0.01, key="fuzzy_slider")
threshold = st.session_state["threshold_slider"]
fuzzy_threshold = st.session_state["fuzzy_slider"]

# ---------- Heuristic Rule Matcher ----------
heuristic_rules = {
    "lexical": ["value", "set", "get", "form", "record", "switch"],
    "referential": ["it", "they", "this", "that"],
    "coordination": ["and/or", "if and only if"],
    "scope": ["some", "many", "few", "each"],
    "vague": ["efficient", "user-friendly", "adequate"]
}
def apply_heuristic_rules(text):
    matches = []
    for category, keywords in heuristic_rules.items():
        for word in keywords:
            if word.lower() in text.lower():
                matches.append(category)
                break
    return matches

# ---------- POS Tagger Matcher + Tree Count ----------
def detect_pos_based_ambiguity(text):
    doc = nlp(text)
    tree_count = len(list(doc.sents))
    return {
        "modal": any(tok.tag_ == "MD" and tok.text.lower() not in ['shall', 'must'] for tok in doc),
        "pronoun": any(tok.pos_ == "PRON" for tok in doc),
        "vague_adj": any(tok.pos_ == "ADJ" and tok.text.lower() in ['flexible', 'efficient', 'adequate'] for tok in doc),
        "quantifier": any(tok.pos_ == "DET" and tok.text.lower() in ['some', 'many', 'few'] for tok in doc),
        "multiple_trees": tree_count > 1
    }

# ---------- Fuzzy Setup ----------
similarity = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'similarity')
ambiguity = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'ambiguity')
similarity['low'] = fuzz.trapmf(similarity.universe, [0, 0, 0.2, 0.4])
similarity['medium'] = fuzz.trimf(similarity.universe, [0.3, 0.5, 0.7])
similarity['high'] = fuzz.trapmf(similarity.universe, [0.6, 0.8, 1, 1])
ambiguity['low'] = fuzz.trapmf(ambiguity.universe, [0.5, 0.7, 1, 1])
ambiguity['high'] = fuzz.trapmf(ambiguity.universe, [0, 0, 0.3, 0.5])
rules = [ctrl.Rule(similarity['low'], ambiguity['high']),
         ctrl.Rule(similarity['medium'], ambiguity['high']),
         ctrl.Rule(similarity['high'], ambiguity['low'])]
fz_ctrl = ctrl.ControlSystem(rules)
simulator = ctrl.ControlSystemSimulation(fz_ctrl)

# ---------- Main Analysis ----------
def detect_structural_ambiguity(text):
    doc = nlp(text)
    verbs = [tok for tok in doc if tok.pos_ == "VERB"]
    cconjs = [tok for tok in doc if tok.pos_ == "CCONJ"]
    adjectives = [tok for tok in doc if tok.pos_ == "ADJ"]
    return {
        "multi_verbs_and": len(verbs) >= 2 and any(tok.text.lower() == "and" for tok in doc),
        "two_conjunctions": len(cconjs) >= 2,
        "multiple_adjectives": len(adjectives) >= 2
    }

def analyze(requirements, ambiguous_terms):
    results = []
    all_texts = [r['text_body'].lower() for r in requirements] + ambiguous_terms
    vec = TfidfVectorizer().fit_transform(all_texts)
    req_vecs = vec[:len(requirements)]
    amb_vecs = vec[len(requirements):]
    for i, req in enumerate(requirements):
        sim_scores = cosine_similarity(req_vecs[i], amb_vecs).flatten()
        max_score = max(sim_scores)
        simulator.input['similarity'] = max_score
        simulator.compute()
        fuzzy_score = simulator.output['ambiguity']
        heuristics = apply_heuristic_rules(req['text_body'])
        pos_flags = detect_pos_based_ambiguity(req['text_body'])
        structural_flags = detect_structural_ambiguity(req['text_body'])
        results.append({
            "ID": req['id'],
            "Requirement Text": req['text_body'],
            "Max Similarity": max_score,
            "Fuzzy Score": fuzzy_score,
            "Heuristic Types": heuristics,
            "POS Flags": pos_flags,
            "Structural Flags": structural_flags,
            "Is Ambiguous": max_score > threshold or fuzzy_score < fuzzy_threshold or bool(heuristics) or any(pos_flags.values())
        })
    return pd.DataFrame(results)

# ========== MAIN APP DISPLAY & INTERACTION ==========
if requirements:
    df_results = analyze(requirements, ambiguous_terms)

    st.subheader("🔍 Ambiguity Detection Results")
    st.dataframe(df_results)

    st.subheader("📊 Ambiguity Classification Summary")
    ambiguity_counts = df_results['Is Ambiguous'].value_counts().rename(index={True: 'Ambiguous', False: 'Not Ambiguous'})
    fig1, ax1 = plt.subplots()
    ambiguity_counts.plot(kind='bar', color=['red', 'green'], ax=ax1)
    ax1.set_title("Ambiguity Detection Result")
    ax1.set_ylabel("Number of Requirements")
    st.pyplot(fig1)

    st.subheader("📈 Fuzzy Score Distribution")
    fig2, ax2 = plt.subplots()
    df_results['Fuzzy Score'].plot(kind='hist', bins=10, color='skyblue', edgecolor='black', ax=ax2)
    ax2.set_title("Distribution of Fuzzy Scores")
    ax2.set_xlabel("Fuzzy Score")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    st.subheader("📎 POS Flag Breakdown")
    pos_flag_counts = pd.DataFrame(df_results['POS Flags'].apply(pd.Series).sum()).rename(columns={0: 'Count'})
    fig3, ax3 = plt.subplots()
    pos_flag_counts.plot(kind='bar', ax=ax3, legend=False)
    ax3.set_title("POS-based Ambiguity Indicators")
    ax3.set_ylabel("Occurrences")
    st.pyplot(fig3)

    st.subheader("🧮 Similarity vs Fuzzy Score")
    fig4, ax4 = plt.subplots()
    ax4.scatter(df_results['Max Similarity'], df_results['Fuzzy Score'], color='purple')
    ax4.set_xlabel("Max Similarity")
    ax4.set_ylabel("Fuzzy Score")
    ax4.set_title("Correlation between Similarity and Fuzzy Interpretation")
    st.pyplot(fig4)

    st.subheader("📤 Export")
    if st.button("Export to CSV"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df_results.to_csv(tmp.name, index=False)
            st.success("Exported to CSV file")
            st.download_button("📥 Download CSV", data=open(tmp.name, "rb").read(), file_name="ambiguity_results.csv")

    # ========== PHASE 4: USER FEEDBACK & INTERACTION ==========
    st.subheader("📝 User Feedback on Ambiguity Results")
    st.write("Review each requirement and update the ambiguity label if needed.")

    # Feedback session state to persist user choices
    if "user_feedback" not in st.session_state:
        st.session_state["user_feedback"] = ["Ambiguous" if val else "Not Ambiguous" for val in df_results["Is Ambiguous"]]

    feedback_list = []
    for idx, row in df_results.iterrows():
        default_val = "Ambiguous" if row["Is Ambiguous"] else "Not Ambiguous"
        user_choice = st.selectbox(
            f"Requirement ID {row['ID']} ({row['Requirement Text'][:45]}...)", 
            ["Ambiguous", "Not Ambiguous"], 
            index=0 if st.session_state["user_feedback"][idx] == "Ambiguous" else 1, 
            key=f"feedback_{idx}"
        )
        feedback_list.append(user_choice)
    # Override ambiguity if user marked it as Not Ambiguous
        

    if st.button("Save Feedback"):
        df_results["Is Ambiguous"] = [
            False if feedback == "Not Ambiguous" else row["Is Ambiguous"]
            for feedback, (_, row) in zip(feedback_list, df_results.iterrows())
        ]
        st.session_state["user_feedback"] = feedback_list
        df_results["User Feedback"] = feedback_list
        st.success("User feedback has been saved! You can now reanalyze the results.")

        # Save feedback as downloadable file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df_results.to_csv(tmp.name, index=False)
            st.download_button("📥 Download Feedback CSV", data=open(tmp.name, "rb").read(), file_name="user_feedback.csv")

    # ========== PHASE 5: FEEDBACK INTEGRATION & REANALYSIS ==========
    if "User Feedback" in df_results.columns:
        st.subheader("🔁 Reanalyze Requirements Based on Feedback")
        if st.button("Reanalyze with User Feedback"):
            # Add ambiguous requirements (based on feedback) to ambiguous_terms for learning
            updated_ambiguous_terms = ambiguous_terms.copy()
            for i, row in df_results.iterrows():
                if row["User Feedback"] == "Ambiguous":
                    updated_ambiguous_terms.append(row["Requirement Text"].lower())
            # Reanalyze and display
            updated_df = analyze(requirements, updated_ambiguous_terms)
            df_results = updated_df


            updated_df["Is Ambiguous"] = [
                False if feedback == "Not Ambiguous" else True
                for feedback in df_results["User Feedback"]
            ]
            st.subheader("✅ Updated Requirements After Feedback")

            for idx, row in updated_df.iterrows():
                st.markdown(f"**Requirement ID {row['ID']}**: {row['Requirement Text']}")
                st.checkbox("Is Ambiguous", value=row["Is Ambiguous"], key=f"updated_checkbox_{idx}", disabled=True)

            # Compare ambiguity status before and after
            updated_df["Was Ambiguous Before"] = df_results["Is Ambiguous"]
            updated_df["Is Ambiguous Now"] = updated_df["Is Ambiguous"]
            st.dataframe(updated_df[["ID", "Requirement Text", "Was Ambiguous Before", "Is Ambiguous Now"]])

            st.success("System has reanalyzed the requirements with your feedback.")
            st.dataframe(updated_df)
            st.subheader("📤 Export Updated Results")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                updated_df.to_csv(tmp.name, index=False)
                st.download_button("📥 Download Updated CSV", data=open(tmp.name, "rb").read(), file_name="updated_ambiguity_results.csv")

else:
    st.warning("Please upload at least one XML or CSV file.")

# ========== EVALUATION SECTION ==========
st.sidebar.header("🧪 Evaluate Your Ambiguity Detection System")
uploaded_labeled_test = st.sidebar.file_uploader("Upload Labeled Test Dataset (.csv)", type="csv", key="labeled_eval")

if uploaded_labeled_test:
    st.subheader("🧪 Evaluation Based on Your Custom Detection System")
    test_df = pd.read_csv(uploaded_labeled_test)
    if 'Requirement' not in test_df.columns or 'Detected as' not in test_df.columns:
        st.error("CSV must contain at least 'Requirement' and 'Detected as' columns.")
    else:
        eval_requirements = [{"id": str(i), "text_body": row['Requirement']} for i, row in test_df.iterrows()]
        results_df = analyze(eval_requirements, ambiguous_terms)
        y_pred = results_df['Is Ambiguous'].astype(int)
        y_true = test_df['Detected as'].map({'NOCUOUS': 1, 'INNOCUOUS': 0}).astype(int)
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_true, y_pred)
        st.markdown(f"### ✅ Custom System Accuracy: `{accuracy:.2f}`")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.subheader("📋 Classification Report ")
        st.dataframe(pd.DataFrame(report).transpose())
        labels = [0, 1]
        label_names = ["Not Ambiguous", "Ambiguous"]
        precision = [report[str(label)]["precision"] for label in labels]
        recall = [report[str(label)]["recall"] for label in labels]
        f1 = [report[str(label)]["f1-score"] for label in labels]
        fig, ax = plt.subplots(figsize=(8, 6))
        x = range(len(label_names))
        ax.bar(x, precision, width=0.2, label='Precision', align='center')
        ax.bar([p + 0.2 for p in x], recall, width=0.2, label='Recall', align='center')
        ax.bar([p + 0.4 for p in x], f1, width=0.2, label='F1 Score', align='center')
        ax.set_xticks([p + 0.2 for p in x])
        ax.set_xticklabels(label_names)
        ax.set_ylim([0, 1])
        ax.set_title(f"Your System Classification Metrics (Accuracy: {accuracy:.2f})")
        ax.legend()
        st.pyplot(fig)
