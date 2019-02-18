import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
import pickle
import numpy as np
import socket
import threading

def alphaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    # print(len(alpha_absolute))
    if len(alpha_absolute) < 64:
        alpha_absolute.append([ch1, ch2, ch3, ch4])
    else:
        alpha_absolute.pop(0)
        alpha_absolute.append([ch1, ch2, ch3, ch4])
        # send()
def betaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    global clientsocket
    if len(beta_absolute) < 64:
        beta_absolute.append([ch1, ch2, ch3, ch4])
    else:
        # print(len(alpha_absolute))
        # print(len(delta_absolute))
        # print(len(beta_absolute))
        # print(len(gamma_absolute))
        # print(len(theta_absolute))
        beta_absolute.pop(0)
        beta_absolute.append([ch1, ch2, ch3, ch4])
        # send()
def deltaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    print(len(delta_absolute))
    if len(delta_absolute) < 64:
        delta_absolute.append([ch1, ch2, ch3, ch4])
    else:
        delta_absolute.pop(0)
        delta_absolute.append([ch1, ch2, ch3, ch4])
        # send()
def gammaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    if len(gamma_absolute) < 64:
        gamma_absolute.append([ch1, ch2, ch3, ch4])
    else:
        gamma_absolute.pop(0)
        gamma_absolute.append([ch1, ch2, ch3, ch4])
        # send()
def thetaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    # global ready
    global clientsocket
    # print(len(theta_absolute), ready)
    if len(theta_absolute) < 64:
        # print("in")
        theta_absolute.append([ch1, ch2, ch3, ch4])
    else:
        theta_absolute.pop(0)
        theta_absolute.append([ch1, ch2, ch3, ch4])
        # send()
        dataframe = np.concatenate([alpha_absolute, beta_absolute, delta_absolute, gamma_absolute, theta_absolute], axis=1).T
        dataframe[dataframe == -np.inf] = 0
        # dataframe = [j for i in dataframe for j in i]
        data = dataframe.tostring()
        clientsocket.send(data)
        print("A data frame sent. ")
        prediction = clientsocket.recv(1024).decode()
        if prediction < 0.4:
            print("You don't seem to be concentrating! Stay focused! Focusing score: {}%".format(prediction*100))
        # ready = True


def send():
    global clientsocket
    # print("alpha", len(alpha_absolute))
    # print("beta", len(beta_absolute))
    # print("delta", len(delta_absolute))
    # print("gamma", len(gamma_absolute))
    # print("theta", len(theta_absolute))
    if len(alpha_absolute) == 64 and len(beta_absolute) == 64 and len(delta_absolute) == 64 and len(gamma_absolute) == 64 and len(theta_absolute) == 64:
        print("in")
        dataframe = np.concatenate([alpha_absolute, beta_absolute, delta_absolute, gamma_absolute, theta_absolute], axis=1)
        dataframe[dataframe == -np.inf] = 0
        dataframe = [j for i in dataframe for j in i]
        data = pickle.dump(dataframe)
        clientsocket.send(data.encode())
        print("A data frame sent. ")
        prediction = clientsocket.recv(1024).decode()
        if prediction < 0.4:
            print("You don't seem to be concentrating! Stay focused! Focusing score: {}%".format(prediction*100))


def focusClient(clientsocket):
    while True:
        global ready
        if ready == True:
            print("Ready")
            dataframe = np.concatenate([alpha_absolute, beta_absolute, delta_absolute, gamma_absolute, theta_absolute], axis=1)
            dataframe[dataframe == -np.inf] = 0
            dataframe = [j for i in dataframe for j in i]
            data = pickle.dump(dataframe)
            clientsocket.send(data.encode())
            print("A data frame sent. ")
            prediction = clientsocket.recv(1024).decode()
            if prediction < 0.4:
                print("You don't seem to be concentrating! Stay focused! Focusing score: {}%".format(prediction*100))
            ready = False

# if __name__ == "__main__":

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

oscserver.serve_forever()

# try:
#     oscserver_thread = threading.Thread(target=oscserver.serve_forever)
#     oscserver_thread.start()

    # print("client threading starting...")
    #
    # focusMuseClient = threading.Thread(target=focusClient, args=(clientsocket, ))
    # focusMuseClient.start()
    # print("client threading started!")
# except KeyboardInterrupt:
#     quit()