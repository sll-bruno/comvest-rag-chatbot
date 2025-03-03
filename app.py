import streamlit as st
from comvest_bot import ComvestChatbot

def get_answer_from_chatbot(chat, query):
    return chat.ask_question(query)

def main():
    st.title("Q&A Comvest Chatbot")
    comvest_chatbot = ComvestChatbot()

    #Histórico de chat
    if "history" not in st.session_state:
        st.session_state.history = []

    #Entrada do usuário
    user_input = st.text_input("Insira sua pergunta sobre o Vestibular da Unicamp: ")

    if user_input:
        #Simula uma resposta do chatbot
        answer = get_answer_from_chatbot(comvest_chatbot,user_input)

        #Adiciona à lista de histórico
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": answer})

    #Exibe o histórico de chat
    for message in st.session_state.history:
        if message["role"] == "user":
            st.write(f"**Você:** {message['content']}")
        else:
            st.write(f"**ComvestChatbot:** {message['content']}")


if __name__ == "__main__":
    main()
