import aiml
import os

kernel = aiml.Kernel()
#kernel.learn("std-startup.xml")
#kernel.respond("load aiml b")

if os.path.isfile("bot_brain.brn"):
    kernel.bootstrap(brainFile = "bot_brain.brn")
else:
    kernel.bootstrap(learnFiles = "std-startup.xml", commands = "load aiml b")
    kernel.saveBrain("bot_brain.brn")

messages = []

while True:
    message = input("Enter your message to the bot: ")
    if message in messages:
        print('You already asked me that.')
    elif message == "quit":
        exit()
    elif message == "save":
        kernel.saveBrain("bot_brain.brn")
    else:
        messages.append(message)
        bot_response = kernel.respond(message)
        print (bot_response)
        