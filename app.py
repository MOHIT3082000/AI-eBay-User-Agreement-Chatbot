import streamlit as st
from rag_chatbot import RAGChatbot

def main():
    st.title("eBay User Agreement Chatbot")
    st.write("Ask me a question about the eBay User Agreement!")

    # Initialize the chatbot. This will load the LLM and the knowledge base.
    # It might take a moment to load the Gemma model the first time.
    chatbot = RAGChatbot()

    user_input = st.text_input("Enter your question:", "", placeholder="Type your question here...")
    if user_input:
        # Display a spinner while the response is being generated
        with st.spinner("Generating response..."):
            response = chatbot.get_response(user_input)
            if response:
                st.write("Response:", response)
            else:
                st.write("Sorry, I didn't understand your question or encountered an issue generating a response.")

    st.write("---") # Separator for better readability
    st.write("Example Questions:")
    st.write("* What is the eBay User Agreement?")
    st.write("* What are the terms and conditions of using eBay?")
    st.write("* How do I report a problem with a seller?")

    st.write("Note: Please ask questions related to the eBay User Agreement. If I'm unsure or don't have enough knowledge, I'll let you know!")

if __name__ == "__main__":
    main()
