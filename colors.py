from enum import Enum

class ColorMap(Enum):
    # This class handles all colors
    GOODWARE = 'green'
    DROIDKUNGFU = 'blue'
    AIRPUSH = 'dodgerblue'
    DOWGIN = 'orange'
    DNOTUA = '#48C9B0' 
    KUGUO = '#E74C3C'
    REVMOB = '#9B59B6'
    # INMOBI = '#F1C40F'
    # FEIWO = '#3498DB'
    # FAKEINST = '#FF9F9B'
    # MECOR = '#brown'
    # SMSREG ='#8FEB34'
    # LEADBOLT = '#204A3B'
    ANYDOWN = '#783878'
    MALWARE = '#DB7093'
    PREC= '#FFE7A3'
    RECALL='#FF5757'
    F1 ='#57FFFC'
    others = '#800000'
    
    @classmethod
    def _missing_(cls,value):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return cls.others  # Return a default color if not found
        