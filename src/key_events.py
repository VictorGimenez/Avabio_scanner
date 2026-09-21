from pynput import keyboard as kb

# Flags compartilhadas
teclas = {
    'esc': False,
    's':   False
}

def start_keyboard():
    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    

def on_press(key):
    if key == kb.Key.esc:
        teclas['esc'] = True
    try:
        if key.char == 's':
            teclas['s'] = True
    except:
        pass

def on_release(key):
    try:
        if key.char == 's':
            teclas['s'] = False
    except:
        pass