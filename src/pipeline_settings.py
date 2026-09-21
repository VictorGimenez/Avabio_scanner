"""Ainda em fase de melhorias e testes"""

from pyorbbecsdk import *

import cv2 as cv


class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame.copy()
        else:
            if frame.dtype != self.previous_frame.dtype:
                frame = frame.astype(self.previous_frame.dtype)
                
            result = cv.addWeighted(
                frame, 
                self.alpha, 
                self.previous_frame, 
                1 - self.alpha, 
                0)
        self.previous_frame = result.copy()
        return result
    
    def reset(self):
        self.previous_frame = None


CAPTURE_PROFILES = {
    'low_light': {
        'name': 'Pouca Luz (< 400 lux)',
        'exposure': 15000,
        'gain': 200, 
        'laser_power': 200,
    },
    'normal': {
        'name': 'Iluminação Normal (400-800 lux)',
        'exposure': 12000,
        'gain': 180,      
        'laser_power': 180,
    },
    'bright': {
        'name': 'Muita Luz (> 800 lux)',
        'exposure': 8000,
        'gain': 150,
        'laser_power': 150,
    }
}

#############################################################  
############### Implementar em outro momento ################      
#############################################################
# def configure_depth_sensor(pipeline, profile_name='normal'):
    
#     if profile_name not in CAPTURE_PROFILES:
#         print(f"[WARN] Perfil '{profile_name}' inválido. Usando 'normal'.")
#         profile_name = 'normal'
    
#     profile = CAPTURE_PROFILES[profile_name]
    
#     print(f"CONFIGURANDO SENSOR: {profile['name']}")
    
#     try:
#         # Obter device e sensor de profundidade
#         device = pipeline.get_device()
#         depth_sensor = device.get_sensor_list().get_sensor_by_type(OBSensorType.DEPTH_SENSOR)
        
#         if depth_sensor is None:
#             print("[Error] Sensor de profundidade não encontrado!")
#             return False
        
#         # 1. DESABILITAR AUTO-EXPOSURE (CRÍTICO!)
#         print("\n[WARN] 1. Desabilitando auto-exposure...")
#         try:
#             depth_sensor.set_bool_property(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, False)
#             print(f"[WARN] Auto-exposure: OFF")
#         except AttributeError:
#             # print(f"[WARN] Não foi possível desabilitar auto-exposure: {e}")
#             try:
#                 ctrl_list = depth_sensor.get_supported_property_list()
#                 for i in range(ctrl_list.count()):
#                     prop = ctrl_list.get_property_at(i)
#                     if prop.id == OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL:
#                         depth_sensor.set_property_value_bool(prop.id, False)
#                         print("[OK] Auto-exposure: OFF (método alternativo)")
#                         break
#             except Exception as e2:
#                 print(f"[WARN] Auto-exposure: usando padrão ({e2})")
        
#         # 2. CONFIGURAR EXPOSURE MANUAL
#         print("\n2. Configurando exposure manual...")
#         try:
#             depth_sensor.set_int_property(
#                 OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, 
#                 profile['exposure']
#             )
#             print(f"[OK] Exposure: {profile['exposure']} μs")
#         except AttributeError:
#             # print(f"[WARN] Não foi possível configurar exposure: {e}")
#             try:
#                 ctrl_list = depth_sensor.get_supported_property_list()
#                 for i in range(ctrl_list.count()):
#                     prop = ctrl_list.get_property_at(i)
#                     if prop.id == OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT:
#                         depth_sensor.set_property_value_int(prop.id, profile['exposure'])
#                         print(f"[OK] Exposure: {profile['exposure']} μs (método alternativo)")
#                         break
#             except Exception as e2:
#                 print(f"[WARN] Exposure: usando padrão ({e2})")
        
#         try:
#             depth_sensor.set_int_property(
#                 OBPropertyID.OB_PROP_DEPTH_GAIN_INT, 
#                 profile['gain']
#             )
#             print(f"[OK] Gain: {profile['gain']}")
#         except AttributeError:
#             try:
#                 ctrl_list = depth_sensor.get_supported_property_list()
#                 for i in range(ctrl_list.count()):
#                     prop = ctrl_list.get_property_at(i)
#                     if prop.id == OBPropertyID.OB_PROP_DEPTH_GAIN_INT:
#                         depth_sensor.set_property_value_int(prop.id, profile['gain'])
#                         print(f"[OK] Gain: {profile['gain']}")
#                         break
#             except Exception as e2:
#                 print(f"[WARN] Gain: usando padrão ({e2})")
        
#         print("\n4. Configurando laser power...")
#         try:
#             try:
#                 depth_sensor.set_int_property(
#                     OBPropertyID.OB_PROP_LASER_POWER_INT, 
#                     profile['laser_power']
#                 )
#                 print(f"[OK] Laser Power: {profile['laser_power']}")
#             except:
#                 depth_sensor.set_int_property(
#                     OBPropertyID.OB_PROP_LDP_INT, 
#                     profile['laser_power']
#                 )
#                 print(f"[OK] Laser Power: {profile['laser_power']}")
#         except:
#             try:
#                 ctrl_list = depth_sensor.get_supported_property_list()
#                 for i in range(ctrl_list.count()):
#                     prop = ctrl_list.get_property_at(i)
#                     if prop.id in [OBPropertyID.OB_PROP_LASER_POWER_INT, OBPropertyID.OB_PROP_LDP_INT]:
#                         depth_sensor.set_property_value_int(prop.id, profile['laser_power'])
#                         print(f"[OK] Laser Power: {profile['laser_power']} (método alternativo)")
#                         break
#             except Exception as e2:
#                 print(f"[WARN]  Laser Power: usando padrão ({e2})")
        
#         print("[OK] SENSOR CONFIGURADO COM SUCESSO!")
        
#         return True
        
#     except Exception as e:
#         print(f"\n [ERROR] Erro ao configurar sensor: {e}\n")
#         return False


def setup_filters():
    """
    Cria e configura filtros para melhorar qualidade do depth.
    """
    print("\n Configurando filtros...")
    
    filters = {}
    
    # 1. Hole Filling Filter (preenche buracos pequenos)
    try:
        hole_filling = HoleFillingFilter()
        #hole_filling.set_filter_params(OBHoleFillingMode.OB_HOLE_FILL_FAREST)
        filters['hole_filling'] = hole_filling
        print("[OK] Hole Filling Filter: Ativado")
    except Exception as e:
        print(f"[WARN] Hole Filling não disponível: {e}")
    
    try:
        spatial = SpatialAdvancedFilter()
        params = OBSpatialAdvancedFilterParams()
        params.alpha = 0.5
        params.magnitude = 2
        params.disp_diff = 20
        params.radius = 2
        # spatial.set_filter_params(
        #     alpha=0.5,           # Força da suavização (0-1)
        #     diff_threshold=20,   # Threshold de diferença
        #     radius=2             # Raio do filtro
        # )
        spatial.set_filter_params(params)
        filters['spatial'] = spatial
        print("[OK] Spatial Filter: Ativado")
    except Exception as e:
        # Tentar versão simples
        try:
            spatial = SpatialModerateFilter()
            filters['spatial'] = spatial
            print("  Spatial Filter (Moderate): Ativado")
        except:
            print(f"  Spatial Filter não disponível: {e}")
    
    print(f"  Total de filtros SDK ativos: {len(filters)}\n")
    
    return filters


def apply_sdk_filters(depth_frame, filters):
    """
    Aplica filtros em sequência no frame de profundidade.
    """
    filtered_frame = depth_frame
    
    filter_order = ['hole_filling', 'spatial']  #filter_order = ['hole_filling', 'spatial', 'temporal']
    
    for filter_name in filter_order:
        if filter_name in filters:
            try:
                filtered_frame = filters[filter_name].process(filtered_frame)
            except Exception as e:
                print(f"[WARN]  Erro ao aplicar {filter_name}: {e}")
    
    return filtered_frame


def create_align_and_point_cloud_filter(has_color_sensor):
    align_to = OBStreamType.COLOR_STREAM if has_color_sensor else OBStreamType.DEPTH_STREAM
    align_filter = AlignFilter(align_to_stream=align_to)
    point_cloud_filter = PointCloudFilter()
    return align_filter, point_cloud_filter


def generate_pipeline(): #substituir cbp = None por cbp = CURRENT_BRIGHTNESS_PROFILE para futuros ajustes de iluminação da câmera
    try:
        pipeline = Pipeline()
        config = Config()

        color_enable = True
        depth_enable = True
        
        # print(f"color_enable = True\ndepth_enable = True")
        
        if depth_enable:
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profiles is None:
                print("No proper depth profile, cannot generate point cloud")
                return
            depth_profile = depth_profiles.get_default_video_stream_profile()
            config.enable_stream(depth_profile)

        if color_enable:
            has_color_sensor = False
            try:
                color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                if color_profiles is not None:
                    color_profile = color_profiles.get_default_video_stream_profile()
                    config.enable_stream(color_profile)
                    has_color_sensor = True
            except OBError as e:
                print(e)
                
        # color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # try:
        #     color_profile = color_profiles.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        # except:
        #     color_profile = color_profiles.get_default_video_stream_profile()
        # config.enable_stream(color_profile)
        
        pipeline.enable_frame_sync() 
        pipeline.start(config)
        
        # configure_depth_sensor(pipeline, profile_name=CURRENT_BRIGHTNESS_PROFILE)
        
        sdk_filters = setup_filters()
        
        align_filter, point_cloud_filter = create_align_and_point_cloud_filter(has_color_sensor)
        
        return pipeline, has_color_sensor, sdk_filters, align_filter, point_cloud_filter
    except RuntimeError as runex:
        print(f"[WARN] Ligue a câmera no computador {runex}")