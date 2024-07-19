base_url = "http://localhost:8000"
import requests
import time
import matplotlib.pyplot as plt
import numpy as np

def initate():
    #"PATCH /parameters/CREATE_NAS_SESSION_DIR?new_value=false HTTP/1.1" 200 OK
    response = requests.patch(f"{base_url}/parameters/CREATE_NAS_SESSION_DIR?new_value=false")
    response = requests.patch(f"{base_url}/parameters/SESSION_NAME_TEMPLATE?new_value=%Y-%m-%d_%H-%M_mlDataset90onlyUniformeSmallwall")
    
    response = requests.post(f"{base_url}/initiate")
    response = requests.post(f"{base_url}/shm/create_termflag_shm")
    response = requests.post(f"{base_url}/shm/create_unityoutput_shm")
    response = requests.post(f"{base_url}/shm/create_unityinput_shm")
    response = requests.post(f"{base_url}/shm/create_unitycam_shm")
    
    response = requests.post(f"{base_url}/shm/create_ballvelocity_shm")
    response = requests.post(f"{base_url}/shm/create_portentaoutput_shm")
    response = requests.post(f"{base_url}/shm/create_portentainput_shm")
    
def start_unity():
    response = requests.post(f"{base_url}/procs/launch_unity")
    print("Unity stared")
    time.sleep(8)
    
def start_session():
    # Additional requests based on the provided log information
    response = requests.post(f"{base_url}/unityinput/Paradigm%2CP0600_mlGym")
    response = requests.post(f"{base_url}/session/animal/AI_001")
    response = requests.post(f"{base_url}/session/animalweight/1")

    # response = requests.post(f"{base_url}/start_paradigm")
    response = requests.post(f"{base_url}/unityinput/Start")
    print("Session started")
    time.sleep(4)

def start_logging():
    response = requests.post(f"{base_url}/procs/launch_log_unity")
    response = requests.post(f"{base_url}/procs/launch_log_unitycam")

def term_delete():
    response = requests.post(f"{base_url}/unityinput/Stop")
    response = requests.post(f"{base_url}/raise_term_flag/delete")

def term_save():
    # response = requests.post(f"{base_url}/stop_paradigm")
    response = requests.post(f"{base_url}/unityinput/Stop")
    response = requests.post(f"{base_url}/raise_term_flag/post-process")

def sample_locations():
    # sample x,y from guassian distribution with mean 0,0 a std=40
    # x = np.random.normal(0, 40, 100)
    # np.random.normal(0, 40, 100)
    
    n = 3*10**3
    xy_samples = np.random.uniform(-50, 50, (n, 2))
    # xy_samples = np.random.multivariate_normal([0, 0], [[400, 0], [0, 400]], n)
    a_samples = np.random.uniform(0, 360, n)
    
    i = 0
    
    for ((x,y),a) in zip(xy_samples, a_samples):
        x = min(max(-50, x), 50)
        y = min(max(-50, y), 50)
        requests.post(f"{base_url}/unityinput/Teleport%2C{x:.3f}%2C{y:.3f}%2C{90:.3f}")
        time.sleep(.015)
        requests.post(f"{base_url}/unityinput/Teleport%2C{y:.3f}%2C{x:.3f}%2C{90:.3f}")
        time.sleep(.015)
        requests.post(f"{base_url}/unityinput/Teleport%2C{-x:.3f}%2C{-y:.3f}%2C{90:.3f}")
        time.sleep(.015)
        requests.post(f"{base_url}/unityinput/Teleport%2C{-y:.3f}%2C{-x:.3f}%2C{90:.3f}")
        time.sleep(.015)
        
        
        # if i < 10000:
            # plt.scatter(x,y, alpha=.3)    
        print(f"{i:07,}/{n}", end="\r")
        i += 1
        
    # plt.show()
        
    
if __name__ == "__main__":
    initate()
    start_unity()
    start_session()

    start_logging()
    sample_locations()

    # term_delete()
    
    term_save()

