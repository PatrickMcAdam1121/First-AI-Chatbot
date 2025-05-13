
import random
import time

class GenerativeManager:
    def __init__(self):
        self.last_popup_time = 0
        self.popup_interval = 900  # 15 minutes in seconds
        self.premium_features = {
            "image_generation": {
                "enabled": False,
                "max_daily": 10,
                "cost": 5.99
            },
            "video_generation": {
                "enabled": False,
                "max_daily": 5,
                "cost": 9.99
            },
            "audio_generation": {
                "enabled": False,
                "max_daily": 15,
                "cost": 7.99
            }
        }
        
    def should_show_popup(self):
        current_time = time.time()
        if current_time - self.last_popup_time > self.popup_interval:
            self.last_popup_time = current_time
            return True
        return False
    
    def get_feature_status(self, feature_name):
        return self.premium_features.get(feature_name, {})
    
    def enable_feature(self, feature_name):
        if feature_name in self.premium_features:
            self.premium_features[feature_name]["enabled"] = True
            
    def disable_feature(self, feature_name):
        if feature_name in self.premium_features:
            self.premium_features[feature_name]["enabled"] = False
