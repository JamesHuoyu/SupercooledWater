data = [
    [185.205, 160.813],
    [188.013, 153],
    [190.737, 144.967],
    [193.206, 137.843],
    [195.943, 129.56],
    [198.443, 122.376],
    [200.973, 114.416],
    [204.107, 104.037],
    [206.774, 95.761],
    [209.252, 87.929],
    [211.65, 80.12],
    [213.995, 71.692],
    [216.044, 64.762],
    [218.068, 57.762],
    [220.325, 49.803],
    [222.462, 42.148],
    [224.363, 35.171],
    [226.461, 27.446],
    [228.577, 19.202],
    [230.552, 11.727],
    [232.336, 4.995],
    [234.031, -1.115],
    [235.595, -7.647],
    [237.446, -15.337],
    [239.421, -23.355],
    [241.272, -31.315],
    [243.124, -39.158],
    [245.099, -47.351],
    [246.95, -55.545],
    [248.802, -63.622],
    [250.53, -71.64],
    [252.381, -79.775],
    [254.232, -88.203],
    [255.96, -96.396],
    [257.688, -104.649],
]
import numpy as np
import pandas as pd

T, P = np.array(data).T
df_widom = pd.DataFrame({"T": T, "P": P})
# 做一个插值函数，用于根据温度获取对应的压力
from scipy.interpolate import interp1d

interp_func = interp1d(T, P, kind="linear", fill_value="extrapolate")


def get_widom_pressure(temperature: float) -> float:
    return float(interp_func(temperature))


def get_widom_temperature(pressure: float) -> float:
    # 反向插值
    reverse_interp_func = interp1d(P, T, kind="linear", fill_value="extrapolate")
    return float(reverse_interp_func(pressure))


if __name__ == "__main__":
    test_temps = [190, 210, 225, 255]
    for temp in test_temps:
        pressure = get_widom_pressure(temp)
        print(f"Temperature: {temp} K -> Widom Pressure: {pressure:.2f} MPa")
    test_pressures = [0.1, -20, -50, -80]
    for pressure in test_pressures:
        temperature = get_widom_temperature(pressure)
        print(f"Pressure: {pressure} MPa -> Widom Temperature: {temperature:.2f} K")
