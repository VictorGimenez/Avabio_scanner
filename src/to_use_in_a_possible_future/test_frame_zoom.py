from pyorbbecsdk import *
import cv2
import numpy as np

def color_frame_to_bgr(color_frame) -> np.ndarray:
    """
    Converte um ColorFrame do pyorbbecsdk em imagem BGR (np.ndarray)
    tratando formatos RGB, BGR e MJPG.
    """
    fmt = color_frame.get_format()
    h = color_frame.get_height()
    w = color_frame.get_width()
    data = color_frame.get_data()
    buf = np.frombuffer(data, dtype=np.uint8)

    if fmt == OBFormat.RGB:
        img = buf.reshape((h, w, 3))
        # SDK costuma fornecer RGB, convertendo para BGR pro OpenCV
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif fmt == OBFormat.BGR:
        img = buf.reshape((h, w, 3))
        return img

    elif fmt == OBFormat.MJPG:
        # frame comprimido em JPEG → precisa decodificar
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Falha ao decodificar frame MJPG com cv2.imdecode")
        return img

    else:
        raise RuntimeError(f"Formato de cor não suportado: {fmt}")


def main():
    pipeline = Pipeline()
    config = Config()

    # Pega o perfil de stream padrão de color
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)

    # Fator de escala inicial (0 < scale <= 1.0)
    # 1.0 = tamanho original, 0.7 = "zoom out" leve
    scale = 0.7

    print("Controles:")
    print("  +  → zoom in (aumenta escala até 1.0)")
    print("  -  → zoom out (diminui escala até 0.2)")
    print("  ESC → sair")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color = frames.get_color_frame()
            if color is None:
                continue

            # Converte ColorFrame para BGR (np.ndarray)
            color_img = color_frame_to_bgr(color)

            # Rotaciona a imagem (ajuste conforme sua necessidade)
            rotated = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
            # Se quiser testar outra:
            # rotated = cv2.rotate(color_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # rotated = cv2.rotate(color_img, cv2.ROTATE_180)

            h, w = rotated.shape[:2]

            # Canvas do mesmo tamanho da imagem rotacionada
            canvas = np.zeros_like(rotated)

            # Garante limites razoáveis de escala
            if scale < 0.2:
                scale = 0.2
            if scale > 1.0:
                scale = 1.0

            # Redimensiona o frame conforme o scale (zoom out/in)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(
                rotated,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            # Centraliza a imagem redimensionada no canvas
            y0 = (h - new_h) // 2
            x0 = (w - new_w) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

            cv2.imshow("Color Rotated with Zoom-Out Style", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('+'):
                scale *= 1.05  # aumenta levemente a escala
            elif key == ord('-'):
                scale /= 1.05  # diminui levemente a escala

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
