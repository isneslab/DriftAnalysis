from enum import Enum

class ColorMap(Enum):
    # This class handles all colors
    GOODWARE = '#00FF00'
    DROIDKUNGFU = '#FFFFCC'
    AIRPUSH = '#FFCCFF'
    DOWGIN = '#CCFFFF'
    DNOTUA = '#FFDAB9' 
    KUGUO = '#E6E6FA'
    REVMOB = '#98FB98'
    INMOBI = '#FFB6C1'
    FEIWO = '#AFEEEE'
    FAKEINST = '#FFDAB9'
    MECOR = '#FFC0CB'
    SMSREG ='#008080'
    LEADBOLT = '#FFB6C1'
    ANYDOWN = '#F0E68C'
    MALWARE = 'DB7093'
    PREC= '#E0FFFF'
    RECALL='#FFD700'
    F1 ='#ADD8E6'
    
    @classmethod
    def _missing_(cls,value):
        value = value.lower()
        for member in cls:
            if member in cls:
                if member.value == value:
                    return member
            else:
                return '#800000'
        