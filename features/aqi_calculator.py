import math

def calc_aqi(cp, breakpoints):
    for bp_low, bp_high, i_low, i_high in breakpoints:
        if bp_low <= cp <= bp_high:
            return ((i_high - i_low) / (bp_high - bp_low)) * (cp - bp_low) + i_low
    return None

#  PM2.5 
def aqi_pm25(pm25):
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    return calc_aqi(pm25, bps)

#  PM10 
def aqi_pm10(pm10):
    bps = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ]
    return calc_aqi(pm10, bps)

#  NO2 
def aqi_no2(no2):
    bps = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ]
    return calc_aqi(no2, bps)

def ugm3_to_ppm_o3(o3_ugm3):
    return o3_ugm3 / 1960

#  O3 (8-hour) 
def aqi_o3(o3):
    if o3 is None:
        return None

    o3_ppm = ugm3_to_ppm_o3(o3)

    bps = [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ]
    return calc_aqi(o3_ppm, bps)

def aqi_so2(so2):
    bps = [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ]
    return calc_aqi(so2, bps)

#  CO 
def aqi_co(co):
    bps = [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ]
    return calc_aqi(co, bps)

#  OVERALL AQI 
def compute_overall_aqi(row):
    values = [
        aqi_pm25(row.get("pm2_5")),
        aqi_pm10(row.get("pm10")),
        aqi_no2(row.get("no2")),
        aqi_o3(row.get("o3")),
        aqi_co(row.get("co")),
        aqi_so2(row.get("so2")),
    ]
    values = [v for v in values if v is not None]
    return max(values) if values else None
