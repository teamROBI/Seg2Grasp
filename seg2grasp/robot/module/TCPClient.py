import socket
import select
import errno
import threading
# from thread import *
import sys, getopt
import os
import datetime
import time

C_HEADER_LENGTH = 2
HEADER_LENGTH = 10
F_HEADER_LENGTH = 20
CLIENT_VERSION = "2.23"

META_HEADER_LENGTH = 2
HEADER_LENGTH = 3

SERVER_IP = "127.0.0.1"
SERVER_PORT = 22021
sent_time = datetime.datetime.now()
received_time = datetime.datetime.now()



def send_msg(msg) :
    # Create client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to server
    try :
        client_socket.connect((SERVER_IP, SERVER_PORT))
    except :
        print("Server is closed")
    if len(msg) < 2 :
        msg = "0" + msg
    client_socket.send(msg.encode('utf-8'))
    client_socket.close()
    return 1

def send_and_receive_msg(msg) :
    # Create client socket
    #print("Init s & m")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to server
    try :
        client_socket.connect((SERVER_IP, SERVER_PORT))
    except :
        return
    if len(msg) < 2 :
        msg = "0" + msg
    #print("client send msg : ",msg)
    client_socket.send(msg.encode('utf-8'))
    
    #print("client wait msg ")

    start_time = time.time()
    while(True) :
        meta_header_raw = client_socket.recv(META_HEADER_LENGTH)
        #print(len(meta_header_raw))
        if len(meta_header_raw) == 0 :
            if (time.time() - start_time) < 0.3 :
                #print("timeout")
                return
            continue
        else :
            break
    
    #print("client got meta_header_raw ", meta_header_raw)

    meta_header = meta_header_raw.decode('utf-8').strip()
    meta_header = int(meta_header)

    if meta_header == 51 :
        header_raw = client_socket.recv(HEADER_LENGTH)
        #print("client got header ", header_raw)
        header = int(header_raw.decode('utf-8').strip())
        content_raw = client_socket.recv(header)
        #print("client got content ", content_raw)
        msg = meta_header_raw + header_raw + content_raw
        msg = msg.decode('utf-8').strip()
        client_socket.close()
        return msg
    else :
        print("Controller client socket received invaild meta header : ",meta_header)
        client_socket.close()


if __name__ == "__main__" :
    send_msg("11")
















