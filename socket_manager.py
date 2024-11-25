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
        received_data = self.client_socket.recv(2048)
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
    
    def receive(self)->tuple[Info, Info]:
        """Receive the data from the socket and decode it.
        The data is in the format of "left_frame_data|right_frame_data".

        left(right)_frame_data: "x y z q1x q1y q1z q1w ... (thumbAngle)"

        thumbAngle: "THJ5 THJ4 THJ3 THJ2 THJ1"

        :return tuple[Info, Info]: _description_
        """
        received_data: str = self.socket.receive()
        left_frame_data, right_frame_data = received_data.split('|')
        left_info = self.__left_decode(left_frame_data)
        right_info = self.__right_decode(right_frame_data)
        return left_info, right_info

    def __right_decode(self, frame_data: str)->Info:
        offset_orientation = torch.tensor([[0.7071068, 0.0, 0.7071068, 0.0]]) # The offset rotation for forearm
        # offset_orientation = torch.tensor([[0.5, 0.5, 0.5, 0.5]]) # The offset rotation for palm
        root_pos, root_rot, dof = right_hand_decode(frame_data)
        root_rot = math_utils.quat_mul(root_rot, offset_orientation)
        return Info(root_pos, root_rot, dof)
    
    def __left_decode(self, frame_data: str)->tuple[torch.tensor, torch.tensor, torch.tensor]:
        offset_orientation = torch.tensor([[0.0, 0.7071068, 0.0, -0.7071068]]) # The offset rotation for forearm
        # offset_orientation = torch.tensor([[0.5, 0.5, -0.5, -0.5]]) # The offset rotation for palm
        root_pos, root_rot, dof = left_hand_decode(frame_data)
        root_rot = math_utils.quat_mul(root_rot, offset_orientation)
        return Info(root_pos, root_rot, dof)