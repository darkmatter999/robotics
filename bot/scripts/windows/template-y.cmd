@echo off

CLS

python3 -m programy.clients.events.console.client --config ..\..\config\windows\config.yaml --cformat yaml --logging ..\..\config\windows\logging.yaml
