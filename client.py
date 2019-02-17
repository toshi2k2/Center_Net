import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
import pickle
import numpy as np
import socket
import threading

def alphaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    if len(alpha_absolute) < 64:
        alpha_absolute.append([ch1, ch2, ch3, ch4])
    else:
        alpha_absolute.pop(0)
        alpha_absolute.append([ch1, ch2, ch3, ch4])
def betaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    if len(alpha_absolute) < 64:
        beta_absolute.append([ch1, ch2, ch3, ch4])
    else:
        beta_absolute.pop(0)
        beta_absolute.append([ch1, ch2, ch3, ch4])
def deltaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    if len(delta_absolute) < 64:
        delta_absolute.append([ch1, ch2, ch3, ch4])
    else:
        delta_absolute.pop(0)
        delta_absolute.append([ch1, ch2, ch3, ch4])
def gammaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    if len(gamma_absolute) < 64:
        gamma_absolute.append([ch1, ch2, ch3, ch4])
    else:
        gamma_absolute.pop(0)
        gamma_absolute.append([ch1, ch2, ch3, ch4])
def thetaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    if len(theta_absolute) < 64:
        theta_absolute.append([ch1, ch2, ch3, ch4])
    else:
        theta_absolute.pop(0)
        theta_absolute.append([ch1, ch2, ch3, ch4])
        ready = True

def focusClient(clientsocket):
    while True:
        if ready == True:
            dataframe = np.concatenate([alpha_absolute, beta_absolute, delta_absolute, gamma_absolute, theta_absolute], axis=1)
            data = pickle.dump(dataframe)
            clientsocket.send(data.encode())
            prediction = clientsocket.recv().decode()
            if prediction == 1:
                pass
            if prediction == 0:
                print("You don't seem to be concentrating! Stay focused! ")
            ready = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--oschost", default="0.0.0.0", help="IP address of osc server. ")
    parser.add_argument("--oscport", type=int, default=65530, help="Port number of osc server. ")
    parser.add_argument("--host", default="127.0.0.1", help="IP address of focusNet server. ")
    parser.add_argument("--port", type=int, default=65531, help="Port number of focusNet server. ")
    args = parser.parse_args()

    alpha_absolute = []
    beta_absolute = []
    delta_absolute = []
    gamma_absolute = []
    theta_absolute = []
    ready = False
    
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect((args.host, args.port))
    print("Connected to focusNet server {}:{}. ".format(args.host, args.port))

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/elements/alpha_absolute", alphaa_handler, args)
    dispatcher.map("/muse/elements/beta_absolute", betaa_handler, args)
    dispatcher.map("/muse/elements/delta_absolute", deltaa_handler, args)
    dispatcher.map("/muse/elements/gamma_absolute", gammaa_handler, args)
    dispatcher.map("/muse/elements/theta_absolute", thetaa_handler, args)

    oscserver = osc_server.ThreadingOSCUDPServer((args.oschost, args.oscport), dispatcher)
    print("OSC serving on {}".format(oscserver.server_address))
    oscserver_thread = threading.Thread(target=oscserver.serve_forever)
    oscserver_thread.start()

    focusMuseClient = threading.Thread(target=focusClient, args=(clientsocket))
    focusMuseClient.start()
