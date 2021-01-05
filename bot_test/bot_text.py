import aiml
import os

kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("load aiml b")


messages = []

while True:
    message = input("Enter your message to the bot: ")
    if message in messages:
        print('You already asked me that.')
    elif message == 'add':
        #With a few simple Python File I/O commands, we are able to dynamically edit the AIML file in use.
        #Here we add a new category and then re-load the kernel
        #This can be of tremendous help later when 'dynamizing' conversation and implementing a realistic conversational flow
        a_file = open("basic_chat2.aiml", "r")
        list_of_lines = a_file.readlines()
        list_of_lines[-2] = "<category><pattern>HOW DO YOU FEEL</pattern><template>Very well today</template></category>\n\n"

        a_file = open("basic_chat2.aiml", "w")
        a_file.writelines(list_of_lines)
        a_file.close()

        kernel.respond("load aiml b")
    elif message == "quit":
        exit()
    elif message == "save":
        kernel.saveBrain("bot_brain.brn")
    else:
        messages.append(message)
        bot_response = kernel.respond(message)
        print (bot_response)
        