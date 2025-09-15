import streamlit as st
import requests
import pandas as pd
import uuid
import time

# Backend URL
BASE_URL = "http://localhost:5000"
CSV_FILE = "./Data/ground-truth-retrieval.csv"

# --- Utility functions ---
def get_random_question(file_path):
    df = pd.read_csv(file_path)
    return df.sample(n=1).iloc[0]["question"]

def ask_question(url, question):
    start_time = time.time()
    data = {"question": question}
    response = requests.post(url, json=data)
    elapsed_time = time.time() - start_time
    return response.json(), elapsed_time

def send_feedback(url, conversation_id, feedback):
    feedback_data = {"conversation_id": conversation_id, "feedback": feedback}
    response = requests.post(f"{url}/feedback", json=feedback_data)
    return response.status_code

# --- Streamlit UI ---
st.set_page_config(page_title="Fridge Chef", page_icon="🥗", layout="centered")
st.title("🥗 Fridge Chef – What's Cooking?")
st.write("Tell me what's in your fridge, and I'll suggest a vegetarian/vegan dish you can make!")

# Session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "stats" not in st.session_state:
    st.session_state.stats = {
        "questions": 0,
        "relevant": 0,
        "non_relevant": 0,
        "skipped": 0,
        "times": []
    }

# Sidebar
st.sidebar.header("🔧 Stats & Options")
st.sidebar.write(f"Total questions: {st.session_state.stats['questions']}")
st.sidebar.write(f"Relevant (+1): {st.session_state.stats['relevant']}")
st.sidebar.write(f"Non-relevant (-1): {st.session_state.stats['non_relevant']}")
st.sidebar.write(f"Skipped: {st.session_state.stats['skipped']}")
if st.session_state.stats['times']:
    st.sidebar.write(f"Avg response time: {sum(st.session_state.stats['times'])/len(st.session_state.stats['times']):.2f}s")

use_random = st.sidebar.checkbox("Test with random question (from dataset)")

if st.sidebar.button("Clear conversation"):
    st.session_state.conversation = []
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.stats = {
        "questions": 0,
        "relevant": 0,
        "non_relevant": 0,
        "skipped": 0,
        "times": []
    }
    st.success("Conversation cleared!")

# Main input with buttons on the same line
ingredients = st.text_input("Enter ingredients in your fridge (comma separated) or a question about what you can make")
col1, col2 = st.columns(2)

question = None
if col1.button("Find Recipes"):
    if ingredients.strip():
        question = f"What can I cook with {ingredients}?"
    else:
        st.warning("Please enter at least one ingredient.")

if use_random and col2.button("Get a random question"):
    question = get_random_question(CSV_FILE)

# Ask backend
if question:
    response, elapsed_time = ask_question(f"{BASE_URL}/question", question)
    answer = response.get("answer", "No answer provided")
    conv_id = response.get("conversation_id", st.session_state.conversation_id)

    # Update conversation
    st.session_state.conversation_id = conv_id
    st.session_state.conversation.append((question, answer))
    st.session_state.stats['questions'] += 1
    st.session_state.stats['times'].append(elapsed_time)

# Display conversation
if st.session_state.conversation:
    st.subheader("💬 Conversation")
    for i, (q, a) in enumerate(st.session_state.conversation, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")

    # Feedback for latest answer
    st.subheader("Give Feedback")
    feedback = st.radio(
        "How was the last response?",
        options=["👍 Positive", "👎 Negative", "Skip"],
        index=2,
        horizontal=True
    )
    if st.button("Submit Feedback"):
        if feedback != "Skip":
            feedback_value = 1 if feedback == "👍 Positive" else -1
            status = send_feedback(BASE_URL, st.session_state.conversation_id, feedback_value)
            if feedback_value == 1:
                st.session_state.stats['relevant'] += 1
            else:
                st.session_state.stats['non_relevant'] += 1
            st.success(f"Feedback sent! (status {status})")
        else:
            st.session_state.stats['skipped'] += 1
            st.info("Feedback skipped.")
