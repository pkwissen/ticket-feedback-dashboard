import pandas as pd
from bertopic import BERTopic
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import re
from tqdm import tqdm

def clean_label(label):
    return re.sub(r"^\d+_", "", label).replace("_", " ")

def generate_topic_labels(df):
    # Filter valid responses
    df = df[df["User Response"].notna()]
    df = df[~df["User Response"].str.strip().str.lower().isin(["", "no comments from the user"])]

    texts = df["User Response"].tolist()

    # Step 1: Run BERTopic
    topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=False)
    topics, probs = topic_model.fit_transform(texts)

    # Step 2: Set readable topic labels (short phrases)
    new_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, separator=" | ")
    topic_model.set_topic_labels(new_labels)

    df["Topic ID"] = topics
    df["Topic Label"] = [topic_model.topic_labels_.get(t, "Unknown") for t in topics]

    # Optional: drop topic -1 (outliers)
    df = df[df["Topic ID"] != -1]

    # Step 3: Improve topic labels with LLM
    prompt_template = PromptTemplate.from_template(
        "You are a helpful assistant. Rewrite the following topic label to make it human-readable and concise (max 5 words not more than that).\n"
        "Here are some examples:\n\n"
        "Raw: ticket_the_was_closed → Closed Ticket\n"
        "Raw: software_the_to_it → Software Upgrade Process\n"
        "Raw: access_to_this_permissions → Permission Access\n"
        "Raw: password_to_the_phone → Phone Password\n"
        "Raw: problem_not_resolved_issue → Unresolved Issue\n\n"
        "Now refine this:\nRaw: {label}\nImproved:"
    )

    llm = Ollama(model="mistral")

    unique_labels = df[["Topic ID", "Topic Label"]].drop_duplicates()
    final_labels = {}

    for _, row in tqdm(unique_labels.iterrows(), total=len(unique_labels)):
        topic_id = row["Topic ID"]
        raw_label = row["Topic Label"]
        cleaned = clean_label(raw_label)
        prompt = prompt_template.format(label=cleaned)
        improved = llm.invoke(prompt).strip().replace("\n", " ")
        final_labels[topic_id] = improved

    df["Improved Topic Label"] = df["Topic ID"].map(final_labels)
    df = df.drop(columns=["Topic Label"])
    df = df.rename(columns={"Improved Topic Label": "Topic Label"})
    return df
