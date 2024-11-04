import socket
import torch
import omni.isaac.lab.utils.math as math_utils
from my_utils import *

class Socket:
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.server_socket = None
        self.client_socket = None
        self.client_address = None
    
    def listen(self)->None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print("Waiting for connection...")
        self.client_socket, self.client_address = self.server_socket.accept()
        print("Connected to:", self.client_address)
    
    def send(self, sent_data: str)->None:
        byte_data = sent_data.encode('utf-8')
        self.client_socket.send(byte_data)
    
    def receive(self)->str:
        received_data = self.client_socket.recv(1024)
        if not received_data:
            return None
        received_data = received_data.decode('utf-8')
        return received_data


class Socket_Manager:
    def __init__(self, **kwargs):
        self.socket = Socket(host=kwargs.get('host', '127.0.0.1'), port=kwargs.get('port', 5000))
    
    def listen(self)->None:
        self.socket.listen()
    
    def send(self):
        self.socket.send(sent_data="STAR PLATINUM | THE WORLD!")
    
    def receive(self)->torch.tensor:
        local_frame_data: str = self.socket.receive()
        return self.decode(local_frame_data)

    def decode(self, frame_data: str)->tuple[torch.tensor, torch.tensor, torch.tensor]:
        root_pos, root_rot, dof = right_hand_decode(frame_data)
        offset_orientation = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
        root_rot = math_utils.quat_mul(root_rot, offset_orientation)
        return root_pos, root_rot, dof