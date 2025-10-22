import os
import sqlite3
import streamlit as st # type: ignore 
from langchain_groq import ChatGroq # type: ignore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- SQL Query Generator ---
def get_sql_query(user_query):
    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are an expert in converting English questions to SQL query!
        The SQL database has the name STUDENT and has the following columns - NAME, COURSE, 
        SECTION and MARKS. 
        Example: "How many entries of records are present?" → SELECT COUNT(*) FROM STUDENT;
        Example: "Tell me all the students studying in Data Science COURSE?" → 
                 SELECT * FROM STUDENT where COURSE="Data Science"; 
        IMPORTANT: 
        - Do not include ``` or 'sql'
        - Only return the SQL query
        Now convert the following question to a valid SQL Query: {user_query}.
    """)
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )
    chain = groq_sys_prompt | llm | StrOutputParser()
    return chain.invoke({"user_query": user_query})

# --- Run SQL on SQLite ---
def return_sql_response(sql_query):
    database = "student.db"
    with sqlite3.connect(database) as conn:
        return conn.execute(sql_query).fetchall()

# --- Chatbot for Explanation ---
def explain_query(message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            SystemMessage(content="You are an AI assistant who explains SQL queries in simple language.")
        ]
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )
    st.session_state.chat_history.append(HumanMessage(content=message))
    response = llm.invoke(st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=response.content))
    return response.content

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Text To SQL")
    st.header("Talk to your Database!")

    user_query = st.text_input("Enter your question about the database:")
    if st.button("Generate SQL"):
        sql_query = get_sql_query(user_query)
        retrieved_data = return_sql_response(sql_query)

        st.subheader(f"Generated SQL Query: {sql_query}")
        st.subheader("Query Results:")
        for row in retrieved_data:
            st.write(row)

        # Save last query for chatbot explanation
        st.session_state.last_query = sql_query

    # Chatbot for explaining queries
    st.subheader("Ask the AI to explain")
    chat_input = st.text_input("Type here to get explanation of SQL query:")

    if st.button("Explain"):
        if "last_query" in st.session_state:
            msg = f"My SQL query is: {st.session_state.last_query}. User question: {chat_input}"
        else:
            msg = chat_input
        response = explain_query(msg)
        st.write("AI:", response)

        # Show chat history
        with st.expander("Conversation History"):
            for msg in st.session_state.chat_history:
                st.write(f"{msg.type}: {msg.content}")

if __name__ == "__main__":
    main()
