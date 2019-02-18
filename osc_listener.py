import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
import pickle
import numpy as np


# def eeg_handler(unused_addr, args, ch1, ch2, ch3, ch4, ch5):
#     # print("EEG (uV) per channel", ch1)
#     pass

def alphar_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    # alpha_relative.append(args)
    # print(unused_addr, ch1, ch2, ch3, ch4)
    # print("[{0}] ~ {1}".format(args[0], ch1))
    pass
def betar_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    # print(unused_addr, ch1, ch2, ch3, ch4)
    pass
def deltar_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    # print(unused_addr, ch1, ch2, ch3, ch4)
    pass
def gammar_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    # print(unused_addr, ch1, ch2, ch3, ch4)
    pass
def thetar_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    # print(unused_addr, ch1, ch2, ch3, ch4)
    pass
def alphaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    alpha_absolute.append([ch1, ch2, ch3, ch4])
    print(unused_addr, ch1, ch2, ch3, ch4)
    # pass
def betaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    beta_absolute.append([ch1, ch2, ch3, ch4])
    print(unused_addr, ch1, ch2, ch3, ch4)
    # pass
def deltaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    delta_absolute.append([ch1, ch2, ch3, ch4])
    print(unused_addr, ch1, ch2, ch3, ch4)
    # pass
def gammaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    gamma_absolute.append([ch1, ch2, ch3, ch4])
    print(unused_addr, ch1, ch2, ch3, ch4)
    # pass
def thetaa_handler(unused_addr, args, ch1, ch2, ch3, ch4):
    theta_absolute.append([ch1, ch2, ch3, ch4])
    print(unused_addr, ch1, ch2, ch3, ch4)
    # pass

def blink_handler(unused_addr, args, value):
    # print(unused_addr, value)
    pass
def jawclench_handler(unused_addr, args, value):
    # print(unused_addr, value)
    pass
def touchingforhead_handler(unused_addr, args, value):
    # print(unused_addr, value)
    pass
def concentration_handler(unused_addr, args, value):
    concentration.append(value)
    # print(unused_addr, value)
    # pass
def mellow_handler(unused_addr, args, value):
    mellow.append(value)
    # print(unused_addr, value)
    # pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="IP address of osc server. ")
    parser.add_argument("--port", type=int, default=65530, help="Port number of osc server. ")
    parser.add_argument("-s", "--subject-number", type=int, help="Experiment number. ")
    parser.add_argument("-c", "--concentrate", type=int, help="If a concentrated experiment use 1, otherwise 0. ")
    args = parser.parse_args()
    if args.concentrate == 1:
        typestr = "concentrated"
    elif args.concentrate == 0:
        typestr = "deviated"
    else:
        print("Argument concentrate should be 0 or 1! ")
        quit()

    alpha_absolute = []
    beta_absolute = []
    delta_absolute = []
    gamma_absolute = []
    theta_absolute = []
    concentration = []
    mellow = []

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/elements/alpha_relative", alphar_handler,  args)
    dispatcher.map("/muse/elements/beta_relative", betar_handler, args)
    dispatcher.map("/muse/elements/delta_relative", deltar_handler, args)
    dispatcher.map("/muse/elements/gamma_relative", gammar_handler, args)
    dispatcher.map("/muse/elements/theta_relative", thetar_handler, args)
    dispatcher.map("/muse/elements/alpha_absolute", alphaa_handler, args)
    dispatcher.map("/muse/elements/beta_absolute", betaa_handler, args)
    dispatcher.map("/muse/elements/delta_absolute", deltaa_handler, args)
    dispatcher.map("/muse/elements/gamma_absolute", gammaa_handler, args)
    dispatcher.map("/muse/elements/theta_absolute", thetaa_handler, args)
    dispatcher.map("/muse/elements/blink", blink_handler, args)
    dispatcher.map("/muse/elements/jaw_clench", jawclench_handler, args)
    dispatcher.map("/muse/elements/touching_forehead", touchingforhead_handler, args)
    dispatcher.map("/muse/elements/experimental/concentration", concentration_handler, args)
    dispatcher.map("/muse/elements/experimental/mellow", mellow_handler, args)

    server = osc_server.ThreadingOSCUDPServer((args.host, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with open("./dataset/subject_"+str(args.subject_number)+"_"+typestr, "wb") as f:
            concentration = [[i] for i in concentration]
            mellow = [[i] for i in mellow]
            pickle.dump(np.array(np.concatenate([alpha_absolute, beta_absolute, delta_absolute, gamma_absolute, theta_absolute, concentration, mellow], axis=1)), f)
        quit()
