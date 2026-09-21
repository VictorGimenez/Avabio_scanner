
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
from typing import Optional
import time
import numpy as np

try:
    import mediapipe as mp
    import cv2 as cv
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tipos públicos
# ---------------------------------------------------------------------------

class Orientation(IntEnum):
    UNKNOWN         = -1
    FRONT           = 0
    SEMI_PROFILE_R  = 1
    PROFILE_R       = 2
    SEMI_PROFILE_L  = 3
    FRONT_BACK      = 4   # pessoa de costas — detectada como FRONT pelo MediaPipe
    SEMI_PROFILE_BL = 5  # semi-perfil de costas lado esquerdo
    PROFILE_L       = 6
    SEMI_PROFILE_BR = 7  # semi-perfil de costas lado direito

    def label_pt(self) -> str:
        return {
            Orientation.UNKNOWN:         "Desconhecido",
            Orientation.FRONT:           "Frente",
            Orientation.SEMI_PROFILE_R:  "Semi-perfil D",
            Orientation.PROFILE_R:       "Perfil D",
            Orientation.SEMI_PROFILE_L:  "Semi-perfil E",
            Orientation.FRONT_BACK:      "Frente (costas)",
            Orientation.SEMI_PROFILE_BL: "Semi-perfil costas E",
            Orientation.PROFILE_L:       "Perfil E",
            Orientation.SEMI_PROFILE_BR: "Semi-perfil costas D",
        }[self]

    def is_pure(self) -> bool:
        """Estados puros exigem parada; semi-perfis capturam em movimento."""
        return self in (Orientation.FRONT, Orientation.PROFILE_R,
                        Orientation.PROFILE_L, Orientation.FRONT_BACK)
 
    def captures_total(self) -> int:
        return 1 if self.is_pure() else 3


class Direction(IntEnum):
    UNKNOWN       = 0
    CLOCKWISE     = 1   # horário   (do ponto de vista da câmera)
    COUNTER       = -1  # anti-horário


@dataclass
class DetectionResult:
    state: Orientation = Orientation.UNKNOWN
    transition: bool = False          # True quando o estado mudou neste frame
    direction: Direction = Direction.UNKNOWN
    full_rotation: bool = False       # True quando volta completa detectada
    shoulder_width: float = 0.0       # razão de aspecto dos ombros [0..1]
    nose_visible: bool = False
    landmarks_found: bool = False
    # --- StabilityGate ---
    capture_ready: bool = False       # True no frame exato em que a captura deve ser feita
    capture_index: int = 0          # qual captura dentro do estado: 1, 2 ou 3
    captures_total: int = 1         # total de capturas esperadas para este estado
    stability_progress: float = 0.0   # 0.0 → 1.0: progresso até liberar captura
    angular_velocity: float = 0.0     # variação de shoulder_width por segundo (proxy de velocidade)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    # Limiares de largura dos ombros (fração da largura do frame)
    front_threshold: float   = 0.28   # dx > threshold → frente/costas
    profile_threshold: float = 0.10   # dx < threshold → perfil puro
    # Histerese: quantos frames consecutivos para confirmar transição
    hysteresis_frames: int = 5
    # Visibilidade mínima para considerar landmark válido
    min_visibility: float = 0.5
    # Mínimo de estados distintos para aceitar volta completa
    # (evita falsas detecções por oscilação rápida)
    min_states_for_full_rotation: int = 3
    # --- StabilityGate ---
    # Tempo mínimo que a pessoa deve ficar estável antes de liberar captura (segundos)
    stable_duration: float = 0.8
    # Velocidade angular máxima para considerar estável (variação de shoulder_width/s)
    max_angular_velocity: float = 0.015
    # Janela de suavização da velocidade angular (frames)
    velocity_window: int = 8
    # Semi-perfis: número de sub-faixas de dx (= número de capturas por semi-perfil)
    semi_profile_captures: int = 3
    # Histerese do sinal facing_camera (frames consecutivos para mudar True↔False)
    facing_hysteresis_frames: int = 6


class FaceDetector:
    """
    Usa MediaPipe Face Detection para determinar se o rosto está visível.
    Não alucina landmarks de costas — simplesmente não retorna detecção.
    """
 
    def __init__(self, min_confidence: float = 0.5, hysteresis_frames: int = 6):
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence,
        )
        self.raw_result = None
        self._confirmed: bool = False
        self._candidate: bool = False
        self._count: int = 0
        self._hysteresis = hysteresis_frames
 
    def is_face_visible(self, rgb_frame: np.ndarray) -> bool:
        self.raw_result = self._detector.process(rgb_frame)
        raw = bool(self.raw_result.detections)
 
        # Histerese: só muda confirmed após N frames consecutivos do mesmo valor
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count = 1
 
        if self._count >= self._hysteresis:
            self._confirmed = self._candidate
 
        return self._confirmed
 
    def close(self):
        self._detector.close()
        

# ---------------------------------------------------------------------------
# Núcleo: estimativa de orientação a partir de landmarks
# ---------------------------------------------------------------------------

class OrientationEstimator:
    """
    Estima Orientation a partir dos landmarks do MediaPipe Pose.
    Entrada: objeto landmarks do MediaPipe (ou None).
    """

    def __init__(self, config: DetectorConfig):
        self.cfg = config

    def estimate(self, landmarks) -> tuple[Orientation, float, bool]:
        """
        Retorna (Orientation, shoulder_width, nose_visible).
        shoulder_width é a distância horizontal normalizada [0..1].
        """
        if landmarks is None:
            return Orientation.UNKNOWN, 0.0, False

        lm = landmarks.landmark
        PL = mp.solutions.pose.PoseLandmark

        ls = lm[PL.LEFT_SHOULDER]
        rs = lm[PL.RIGHT_SHOULDER]
        # nose      = lm[PL.NOSE]
        # left_ear  = lm[PL.LEFT_EAR]
        # right_ear = lm[PL.RIGHT_EAR]
        # left_eye  = lm[PL.LEFT_EYE]
        # right_eye = lm[PL.RIGHT_EYE]

        # Visibilidade mínima nos ombros
        if ls.visibility < self.cfg.min_visibility or rs.visibility < self.cfg.min_visibility:
            return Orientation.UNKNOWN, 0.0, False

        dx = abs(ls.x - rs.x)
        
        # ear_visible  = (left_ear.visibility  >= self.cfg.min_visibility or right_ear.visibility >= self.cfg.min_visibility)
        # eye_visible  = (left_eye.visibility  >= self.cfg.min_visibility or right_eye.visibility >= self.cfg.min_visibility)
        # nose_visible = nose.visibility >= self.cfg.min_visibility

        # face_votes = sum([nose_visible, ear_visible, eye_visible])
        facing_camera = face_votes >= 2
                

        # Qual ombro está "mais à frente" (z menor = mais próximo da câmera)
        # No MediaPipe, z é relativo ao quadril: negativo = mais próximo
        left_closer = ls.z > rs.z #ls.z < rs.z

        orientation = self._classify(dx, facing_camera, left_closer)
        return orientation, dx, facing_camera


    def _classify(self, dx: float, facing_camera: bool, left_closer: bool) -> Orientation:
        ft = self.cfg.front_threshold
        pt = self.cfg.profile_threshold
        mid_f = (ft + pt) / 2   # fronteira semi-perfil / perfil

        if dx >= ft:
            return Orientation.FRONT if facing_camera else Orientation.BACK
 
        if dx <= pt:
            return Orientation.PROFILE_L if left_closer else Orientation.PROFILE_R
 
        # Zona intermediária → semi-perfil ou semi-costas
        if dx > mid_f:
            if facing_camera:
                return Orientation.SEMI_PROFILE_L if left_closer else Orientation.SEMI_PROFILE_R
            else:
                return Orientation.SEMI_BACK_L if left_closer else Orientation.SEMI_BACK_R
        else:
            if facing_camera:
                return Orientation.SEMI_PROFILE_L if left_closer else Orientation.SEMI_PROFILE_R
            else:
                return Orientation.SEMI_BACK_L if left_closer else Orientation.SEMI_BACK_R

# ---------------------------------------------------------------------------
# Máquina de estados com histerese
# ---------------------------------------------------------------------------

class StateMachine:
    """
    Consome orientações brutas frame a frame e emite transições confirmadas
    com histerese, além de detectar volta completa.
    """

    # Sequências de volta completa (horária e anti-horária)
    # Como o MediaPipe não detecta orientações com a pessoa totalmente de costas e posições semi perfil de costas substitui esse bloco pelo bloco abaixo
    _CW  = [Orientation.SEMI_PROFILE_R, Orientation.PROFILE_R,
        Orientation.SEMI_PROFILE_L, Orientation.FRONT]
    _CCW = [Orientation.SEMI_PROFILE_L, Orientation.PROFILE_L,
        Orientation.SEMI_PROFILE_R, Orientation.FRONT]
    # _CW  = [Orientation.SEMI_PROFILE_R, Orientation.PROFILE_R,
    #         Orientation.SEMI_PROFILE_L, Orientation.PROFILE_L, Orientation.FRONT]
    # _CCW = [Orientation.SEMI_PROFILE_L, Orientation.PROFILE_L,
    #         Orientation.SEMI_PROFILE_R, Orientation.PROFILE_R, Orientation.FRONT]
    # _CW = [
    #     Orientation.FRONT,
    #     Orientation.SEMI_PROFILE_R,
    #     Orientation.PROFILE_R,
    #     Orientation.SEMI_PROFILE_R,
    #     Orientation.FRONT,           # costas
    #     Orientation.SEMI_PROFILE_L,
    #     Orientation.PROFILE_L,
    #     Orientation.SEMI_PROFILE_L,
    #     Orientation.FRONT,           # volta completa
    # ]
    # _CCW = [
    #     Orientation.FRONT,
    #     Orientation.SEMI_PROFILE_L,
    #     Orientation.PROFILE_L,
    #     Orientation.SEMI_PROFILE_L,
    #     Orientation.FRONT,           # costas
    #     Orientation.SEMI_PROFILE_R,
    #     Orientation.PROFILE_R,
    #     Orientation.SEMI_PROFILE_R,
    #     Orientation.FRONT,           # volta completa
    # ]

    def __init__(self, config: DetectorConfig):
        self.cfg = config
        self.confirmed_state = Orientation.UNKNOWN
        self.direction = Direction.UNKNOWN
        self._candidate = Orientation.UNKNOWN
        self._candidate_count = 0
        self._history: deque[Orientation] = deque(maxlen=len(self._CW) + 2)

    def update(self, raw: Orientation) -> tuple[bool, bool, Direction]:
        """
        Retorna (transition_happened, full_rotation, direction).
        """
        transition = False
        full_rotation = False

        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1

        if self._candidate_count >= self.cfg.hysteresis_frames:
            if self._candidate != self.confirmed_state:
                self.confirmed_state = self._candidate
                transition = True
                if self.confirmed_state != Orientation.UNKNOWN:
                    self._history.append(self.confirmed_state)
                    full_rotation, detected_dir = self._check_full_rotation()
                    if full_rotation:
                        self.direction = detected_dir

        return transition, full_rotation, self.direction

    def _check_full_rotation(self) -> tuple[bool, Direction]:
        h = list(self._history)
        print(f"  [history] {[o.label_pt() for o in h]}") 
        n = len(self._CW)
        if len(h) < self.cfg.min_states_for_full_rotation:
            return False, Direction.UNKNOWN

        # Testa subsequência dos últimos n estados
        tail = h[-n:]
        if self._matches_sequence(tail, self._CW):
            self._history.clear()
            return True, Direction.CLOCKWISE
        if self._matches_sequence(tail, self._CCW):
            self._history.clear()
            return True, Direction.COUNTER
        return False, Direction.UNKNOWN

    @staticmethod
    def _matches_sequence(observed: list, expected: list) -> bool:
        """
        Verifica se 'observed' cobre todos os estados de 'expected' em ordem,
        tolerando estados duplicados e estados intermediários ausentes.
        """
        if not observed or not expected:
            return False
        ei = 0
        for state in observed:
            if state == expected[ei]:
                ei += 1
                if ei == len(expected):
                    return True
        return False


# ---------------------------------------------------------------------------
# StabilityGate
# ---------------------------------------------------------------------------

class StabilityGate:
    """
    Decide se o momento atual é adequado para disparar a captura.

    Critérios simultâneos:
      1. Velocidade angular baixa  — a pessoa parou (ou quase) de girar.
      2. Estabilidade temporal     — ficou nesse estado por >= stable_duration segundos.

    O gate é por estado: ao entrar em um novo estado confirmado, o contador
    zera e o gate só abre depois que ambos os critérios forem satisfeitos.
    O gate fecha imediatamente se a velocidade angular subir novamente
    (a pessoa recomeçou a girar antes de capturar).

    Emite capture_ready=True apenas UMA VEZ por estado (no frame exato em que
    a condição é satisfeita pela primeira vez). Após isso, fica silencioso até
    uma nova transição de estado.
    """

    def __init__(self, config: DetectorConfig):
        self.cfg = config
        self._current_state: Orientation = Orientation.UNKNOWN
        self._stable_since: float = 0.0          # timestamp em que ficou estável
        self._fired: bool = False                 # captura já emitida para este estado
        self._dx_window: deque[float] = deque(maxlen=config.velocity_window)
        self._last_time: float = time.monotonic()
        self._angular_velocity: float = 0.0

    def update(self, state: Orientation, transition: bool, dx: float) -> tuple[bool, float, float]:
        """
        Recebe o estado confirmado atual, se houve transição e o shoulder_width (dx).
        Retorna (capture_ready, stability_progress, angular_velocity).

        capture_ready     — True apenas no frame em que a captura deve ser disparada.
        stability_progress — [0, 1]: quanto do stable_duration já foi cumprido.
        angular_velocity  — velocidade angular suavizada (para debug/UI).
        """
        now = time.monotonic()
        dt = max(now - self._last_time, 1e-6)
        self._last_time = now

        # --- Velocidade angular (variação de dx suavizada) ---
        self._dx_window.append(dx)
        if len(self._dx_window) >= 2:
            deltas = [abs(self._dx_window[i] - self._dx_window[i - 1])
                      for i in range(1, len(self._dx_window))]
            # converte para por-segundo usando dt médio
            avg_delta_per_frame = sum(deltas) / len(deltas)
            self._angular_velocity = avg_delta_per_frame / dt
        else:
            self._angular_velocity = 0.0

        # --- Novo estado: zera o gate ---
        if transition and state != Orientation.UNKNOWN:
            self._current_state = state
            self._stable_since = 0.0   # ainda não confirmamos estabilidade
            self._fired = False
            self._dx_window.clear()

        # Já disparou para este estado — nada a fazer
        if self._fired:
            return False, 1.0, self._angular_velocity

        # Estado inválido
        if self._current_state == Orientation.UNKNOWN:
            return False, 0.0, self._angular_velocity

        moving = self._angular_velocity > self.cfg.max_angular_velocity

        if moving:
            # Pessoa ainda girando — reseta o contador de tempo estável
            self._stable_since = 0.0
            return False, 0.0, self._angular_velocity

        # Pessoa parada: inicia ou continua contagem
        if self._stable_since == 0.0:
            self._stable_since = now

        elapsed = now - self._stable_since
        progress = min(elapsed / self.cfg.stable_duration, 1.0)

        if elapsed >= self.cfg.stable_duration:
            self._fired = True
            return True, 1.0, self._angular_velocity

        return False, progress, self._angular_velocity


# ---------------------------------------------------------------------------
# Interface pública principal
# ---------------------------------------------------------------------------

class RotationDetector:
    """
    Detector de rotação completa do corpo humano via câmera frontal.

    Uso:
        detector = RotationDetector()
        result = detector.update(frame)  # frame: np.ndarray BGR
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe não encontrado. Instale com: pip install mediapipe")

        self.cfg = config or DetectorConfig()
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._estimator = OrientationEstimator(self.cfg)
        self._sm = StateMachine(self.cfg)
        self._gate = StabilityGate(self.cfg)

    def update(self, frame: np.ndarray) -> DetectionResult:
        """
        Processa um frame BGR e retorna DetectionResult.
        Chame uma vez por frame no seu loop de captura.

        O campo result.capture_ready é True APENAS no frame em que a pessoa
        ficou estável tempo suficiente — esse é o momento certo para registrar
        a nuvem de pontos.
        """
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        landmarks = results.pose_landmarks if results.pose_landmarks else None
        self._pose_landmarks_raw = results.pose_landmarks
        orientation, shoulder_width, nose_visible = self._estimator.estimate(landmarks)
        transition, full_rotation, direction = self._sm.update(orientation)
        capture_ready, stability_progress, angular_velocity = self._gate.update(self._sm.confirmed_state, transition, shoulder_width)

        return DetectionResult(
            state=self._sm.confirmed_state,
            transition=transition,
            direction=direction,
            full_rotation=full_rotation,
            shoulder_width=shoulder_width,
            nose_visible=nose_visible,
            landmarks_found=landmarks is not None,
            capture_ready=capture_ready,
            stability_progress=stability_progress,
            angular_velocity=angular_velocity,
        )

    def reset(self):
        """Reinicia o estado interno (use entre sessões de scanning)."""
        self._sm = StateMachine(self.cfg)
        self._gate = StabilityGate(self.cfg)

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Demo standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    cap = cv.VideoCapture(0)
    detector = RotationDetector()

    print("Gire lentamente 360°. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.update(frame)

        # Overlay simples
        label = result.state.label_pt() if result.landmarks_found else "Sem pose"
        color = (0, 255, 0) if result.landmarks_found else (0, 0, 255)
        cv.putText(frame, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        if result.transition:
            print(f"→ {result.state.label_pt()}")

        if result.full_rotation:
            dir_label = "HORÁRIA" if result.direction == Direction.CLOCKWISE else "ANTI-HORÁRIA"
            print(f"✓ Volta completa {dir_label}!")
            cv.putText(frame, f"VOLTA COMPLETA {dir_label}", (20, 90),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv.imshow("Rotation Detector", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    detector.close()