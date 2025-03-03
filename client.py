from comvest_bot import ComvestChatbot
import sys

comvest_chatbot = ComvestChatbot()

while True:
    question = input("Insira sua pergunta sobre o Vestibular da Unicamp: ")
    
    if question == "exit":
        sys.exit()
        
    answer = comvest_chatbot.ask_question(question)
    print(answer)