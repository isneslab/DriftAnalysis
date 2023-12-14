import random
import string
import pickle
import os

class ColorMap():
    # This class handles all colors
    def __init__(self):
        if not os.path.isfile('pkl_files/color_family_dict.pkl'):
            self.colormap = {
                'GOODWARE' : 'green',
                'DROIDKUNGFU' : 'blue',
                'AIRPUSH' : 'dodgerblue',
                'DOWGIN' : 'orange',
                'DNOTUA' : '#48C9B0', 
                'KUGUO' : '#E74C3C',
                'REVMOB' : '#9B59B6',
                'INMOBI' : '#F1C40F',
                'FEIWO' : '#3498DB',
                'FAKEINST' : '#FF9F9B',
                'MECOR' : 'brown',
                'SMSREG' :'#8FEB34',
                'LEADBOLT' : '#204A3B',
                'ANYDOWN' : '#783878',
                'MALWARE' : '#DB7093',
                'PREC': '#FFE7A3',
                'RECALL' :'#FF5757',
                'F1' :'#57FFFC',
                'others' : '#800000'
            }
            with open('pkl_files/color_family_dict.pkl','wb') as f:
                pickle.dump(self.colormap, f)
        else:
            with open('pkl_files/color_family_dict.pkl','rb') as f:
                self.colormap = pickle.load(f)
        
        
    def get_color(self, family_name):
        if family_name in self.colormap:
            return self.colormap[family_name]
        else:
            return self.add_color(family_name)
    
    
    def add_color(self, family_name):
        listed_colors = list(self.colormap.values())        
        while True:
            new_color = self.make_new_color()
            if new_color not in listed_colors:
                break
            
        self.colormap[family_name] = new_color
        return new_color
                   
        
    def make_new_color(self):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        return f'#{red:02x}{green:02x}{blue:02x}'
