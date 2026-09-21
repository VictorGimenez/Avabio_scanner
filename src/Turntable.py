import serial
from serial.tools import list_ports
# for p in list_ports.comports():
#     print("device:", p.device)
#     print("desc  :", p.description)
#     print("hwid  :", p.hwid)
#     print("vid   :", p.vid)
#     print("pid   :", p.pid)
#     print("-------------------")
import time
import os
import glob
import threading
import traceback

#Válidos somente para a turntable da BKL
commands_bkl = {
        'stop'                 : b'\xFF\xFF\xD1\x04\x00\xD5\xFF\xFE',
        'rot_clockw'           : b'\xFF\xFF\xD1\x04\x01\xD4\xFF\xFE',
        'rot_counterclockw'    : b'\xFF\xFF\xD1\x04\x02\xD7\xFF\xFE',
        'continuous_clockw'    : b'\xFF\xFF\xD1\x04\x03\xD6\xFF\xFE',
        'gear1'                : b'\xFF\xFF\xD2\x04\x00\xD6\xFF\xFE',
        'gear2'                : b'\xFF\xFF\xD2\x04\x01\xD7\xFF\xFE',
        'gear3'                : b'\xFF\xFF\xD2\x04\x02\xD4\xFF\xFE',
        'origin'               : b'\xFF\xFF\xF1\x04\x04\xF1\xFF\xFE',   #esse hexadecimal é retornado assim que a base chega na sua posição de origem
        'reset'                : b'\xFF\xFF\xD3\x04\x02\xD5\xFF\xFE',
        'query_version'        : b'\xFF\xFF\xD3\x04\x01\xD6\xFF\xFE',
        'status_chk'           : b'\xFF\xFF\xD4\x04\x03\xD3\xFF\xFE',
    }

#Válidos somente para a turntable da ComXim
commands_comxim = {
        'stop'                 : b'CT+SETSTOP();',
        'rot_clockw'           : b'CT+TRUNSINGLE(0,360);',  #CR+OK;CR+EVENT=TB_PAUSE;CR+EVENT=TB_END;
        'rot_counterclockw'    : b'CT+TRUNSINGLE(1,360);',    #CR+OK;CR+EVENT=TB_PAUSE;CR+EVENT=TB_END;
        'gear1'                : b'CT+SETSPEED(1);',
        'gear2'                : b'CT+SETSPEED(3);',
        'gear3'                : b'CT+SETSPEED(5);',
        'gear4'                : b'CT+SETSPEED(7);',
        'gear5'                : b'CT+SETSPEED(9);',
        'origin'               : b'CR+EVENT=TB_PAUSE;CR+EVENT=TB_END' #b'CT+TOZERO();',   #diferente da BKL nesse script a base retorna automaticamente a sua posição de origem 
    }
    

class Turntable(serial.Serial):

    TARGET_VID = 6790
    TARGET_PID = 29987

    KW = ["USB", "CH340", "CP210", "FTDI", "Serial"]

    def __init__(self, 
                 port = None,
                 baudrate = 115200, #9600, #115200 é usado pela ComXim e o 9600 pela BKL
                 bytesize = serial.EIGHTBITS, 
                 parity = serial.PARITY_NONE, 
                 stopbits = serial.STOPBITS_ONE, 
                 timeout = 1,
                 arquivo_simulacao=None, 
                 pausa_simulacao=0.5):
        
        self.modo_simulacao = False
        self._arquivo_handle = None
        self.pausa_simulacao = pausa_simulacao
            
        if port is None:
            port = self.detect_port()

        if port is None:
            if arquivo_simulacao is not None:
                self._ativar_modo_simulacao(arquivo_simulacao, baudrate, bytesize, parity, stopbits, timeout)
                return
            else:
                raise serial.SerialException("Não foi possível detectar automaticamente a porta.")

        try:
            super().__init__(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout,
            )
        except Exception as e:    
            if arquivo_simulacao is not None:
                print("[INFO] Falha ao abrir porta física. Entrando em modo simulação.")
                self._ativar_modo_simulacao(arquivo_simulacao, baudrate, bytesize, parity, stopbits, timeout)
            else:
                print(f"[ERRO] Falha ao abrir porta serial: {e}")
                traceback.print_exc()
                raise
    
    
    def _ativar_modo_simulacao(self, arquivo_simulacao, baudrate, bytesize, parity, stopbits, timeout):
        self.modo_simulacao = True
        self.port = None
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self._arquivo_handle = open(arquivo_simulacao, "r", encoding="utf-8")
    
    
    @classmethod
    def detect_port(cls):

        ports = list_ports.comports()

        # 1) VID / PID
        if cls.TARGET_VID is not None and cls.TARGET_PID is not None:
            for p in ports:
                if p.vid == cls.TARGET_VID and p.pid == cls.TARGET_PID:
                    return p.device

        # 2) keywords
        for p in ports:
            text = f"{p.description} {p.manufacturer} {p.hwid}".lower()
            if any(k.lower() in text for k in cls.KW):
                return p.device

        # 3) OS fallback
        if os.name == "nt":
            for p in ports:
                if p.device.startswith("COM"):
                    return p.device
        else:
            for p in ports:
                if p.device.startswith("/dev/tty"):
                    return p.device

        # 4) final fallback
        if ports:
            return ports[0].device

        return None

    
    def open_turntable_connection(self):
        print("Abrindo")
        if self == None or self.is_open:
            pass
        else:
            self.open()


    def close_turntable_connection(self):
        print("Fechando")
        if self == None or self.closed:
            pass
        else:
            self.close()
            
            
    def send_message(self, message):
        self.write(message)

    
    def rotate(self):
        self.send_message(commands_comxim['rot_clockw'])
        return self.readline()
    
    
    def readline(self):
        print(f"[DEBUG readline] modo_simulacao = {self.modo_simulacao}")
        if self.modo_simulacao:
            linha = self._arquivo_handle.readline()
            if not linha:
                return ""
            time.sleep(self.pausa_simulacao)
            return linha.strip()
        return super().readline()

    
    def waiting_origin(self, target = commands_comxim['origin']):
        while True:
            linha = self.readline().strip()
            if not linha:
                break
            
            #linha_bytes = string_para_bytes(linha)
            
            if target in linha:
            #if linha == target:
                return True
            

def string_para_bytes(s):
    """
    Converte uma string como '\\xF2\\x6B' (texto literal)
    para o objeto bytes real b'\\xf2\\x6b'.
    """
    s = s.strip()
    hex_str = s.replace("\\x", "")  # remove todos os '\x', sobra só 'F26B'
    return bytes.fromhex(hex_str) 