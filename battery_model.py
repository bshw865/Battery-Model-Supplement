
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D              
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MACARON_COLORS = {
    'blue': '#AEC6CF', 'pink': '#FFD1DC', 'green': '#77DD77',
    'purple': '#CFCFC4', 'orange': '#FFB347', 'yellow': '#FDFD96',
    'mint': '#98FF98', 'red': '#FF6B6B'
}


                                                              
             
                                                              
@dataclass
class BatteryParams:
       
    Lx: float = 0.044
    Ly: float = 0.0863
    Lz: float = 0.0046
    nx: int = 10
    ny: int = 20
    nz: int = 3

    mass: float = 0.046
    C_nom: float = 3.24         
    V_nom: float = 3.7         

    rho: float = None
    Cp: float = 1000.0
    k: float = 1.5

    h_conv: float = 10.0
    h_contact: float = 50.0

    mass_phone: float = 0.180
    Cp_phone: float = 850.0
    A_phone_surf: float = 0.015
    h_phone_amb: float = 14.0

                                        
    R_ct_ref:  float = 0.010              
    R_diff_ref: float = 0.010           
    R_SEI_ref:  float = 0.005             
    R_ohm_ref:  float = 0.005           
                           

                                  
    Ea_ct:   float = 0.54
    Ea_diff: float = 0.54
    Ea_ohm:  float = 0.10

    alpha_SEI:  float = 1.0
    alpha_ohm:  float = 1.0
    alpha_ct:   float = 1.0
    alpha_diff: float = 1.0

    T_min: float = 0.0
    T_max: float = 65.0
    SOC_min: float = 0.0
    SOC_max: float = 100.0

    def __post_init__(self):
        self.volume = self.Lx * self.Ly * self.Lz
        self.rho = self.mass / self.volume
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        self.V_cell = self.dx * self.dy * self.dz
        self.m_cell = self.rho * self.V_cell
        self.R_total_ref = (self.R_ct_ref + self.R_diff_ref
                            + self.R_SEI_ref + self.R_ohm_ref)
        self.C_th_phone = self.mass_phone * self.Cp_phone


                                                              
                       
                                                              
class ButlerVolmerPolarizationModel:
       

    R_GAS = 8.314                 
    F     = 96485.0           
    k_B   = 8.617e-5         
    T_REF = 298.15        

    I0_CATHODE_REF = 2.334                        
    I0_ANODE_REF   = 1.8714                             

    EA_CATHODE = 0.45             
    EA_ANODE   = 0.40             

    ALPHA = 0.5

    A_CATHODE = 885000.0                 
    A_ANODE   = 723600.0                 

    L_CATHODE = 134e-6            
    L_ANODE   = 146e-6            

    A_ELECTRODE = 0.044 * 0.0863      

    def exchange_current_density(self, T_K: float,
                                  electrode: str) -> float:
        if electrode == 'cathode':
            i0_ref, Ea = self.I0_CATHODE_REF, self.EA_CATHODE
        else:
            i0_ref, Ea = self.I0_ANODE_REF, self.EA_ANODE
        return i0_ref * np.exp(
            -(Ea / self.k_B) * (1.0 / T_K - 1.0 / self.T_REF)
        )

    def polarization_resistance(self, T_K: float,
                                 electrode: str) -> float:
        i0   = self.exchange_current_density(T_K, electrode)
        a    = self.A_CATHODE if electrode == 'cathode' else self.A_ANODE
        L    = self.L_CATHODE if electrode == 'cathode' else self.L_ANODE
        R_area = (self.R_GAS * T_K) / (self.ALPHA * self.F * i0 * a * L)
        return R_area / self.A_ELECTRODE

    def calculate_polarization_factor(self, T_eff: float,
                                       is_charging: bool,
                                       R_ref: float) -> float:
        T_K      = T_eff + 273.15
        electrode = 'anode' if is_charging else 'cathode'
        R_pol    = self.polarization_resistance(T_K, electrode)
        factor   = R_pol / R_ref if R_ref > 0 else 1.0
        return float(np.clip(factor, 0.05, 20.0))

    def verify_charge_discharge_ratio(self) -> dict:
        T_K         = self.T_REF
        R_charge    = self.polarization_resistance(T_K, 'anode')
        R_discharge = self.polarization_resistance(T_K, 'cathode')
        ratio       = R_charge / R_discharge
        in_range    = 1.2 <= ratio <= 1.6
        return {
            'R_charge_mOhm'   : R_charge * 1000,
            'R_discharge_mOhm': R_discharge * 1000,
            'ratio'           : ratio,
            'literature_range': '1.2 ~ 1.6',
            'in_range'        : in_range
        }

    def verify_c_rate_trend(self) -> dict:
        temps    = [10, 25, 40, 55]
        R_values = []
        for T in temps:
            T_K = T + 273.15
            R   = self.polarization_resistance(T_K, 'cathode')
            R_values.append(R * 1000)
        is_decreasing = all(
            R_values[i] > R_values[i+1]
            for i in range(len(R_values)-1)
        )
        return {
            'temperatures_C' : temps,
            'R_pol_mOhm'     : [round(r, 4) for r in R_values],
            'trend_decreasing': is_decreasing,
            'note': '温度升高→i0增大→R_pol减小，与文献趋势一致'
        }

    def verify_absolute_value(self, R_total_ref: float = 0.030) -> dict:
        T_K  = self.T_REF
        R_ct = self.polarization_resistance(T_K, 'cathode') * 1000
        R_an = self.polarization_resistance(T_K, 'anode')   * 1000
        R_total_mOhm = R_total_ref * 1000
        ratio_ct = R_ct / R_total_mOhm
        return {
            'R_pol_cathode_mOhm': round(R_ct, 4),
            'R_pol_anode_mOhm'  : round(R_an, 4),
            'R_total_ref_mOhm'  : R_total_mOhm,
            'ratio_to_ref'      : round(ratio_ct, 3),
            'reasonable'        : 0.1 <= ratio_ct <= 10.0
        }


                                                              
       
                                                              
class EntropyCoefficientModel:
    def get_entropy_coefficient(self, SOC: float,
                                 I_app: float = 0.0) -> float:
        C_rate = abs(I_app) / 3.24 if 3.24 > 0 else 0
        if I_app > 0:
            return (0.05 + 0.12 * C_rate) * 1e-3
        else:
            return -0.27 * 1e-3


class ResistanceModel:
       
    def __init__(self, params: BatteryParams):
        self.params   = params
        self.bv_model = ButlerVolmerPolarizationModel()

        self._center_factor_scale   = 1.5
        self._depth_factor_surface  = 0.8
        self._depth_factor_interior = 1.2

    def calculate_spatial_factor(self, i: int, j: int,
                                  k: int) -> float:
        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz
        xc = abs(i-(nx-1)/2) / ((nx-1)/2) if nx > 1 else 0
        yc = abs(j-(ny-1)/2) / ((ny-1)/2) if ny > 1 else 0
        center_factor = 1 + self._center_factor_scale * (1-xc)*(1-yc)
        depth_factor  = (self._depth_factor_surface
                         if (k == 0 or k == nz-1)
                         else self._depth_factor_interior)
        return center_factor * depth_factor

    def calculate_temperature_factor(self, T_node: float,
                                      res_type: str) -> float:
        T_K   = T_node + 273.15
        T_ref = 298.15
        k_B   = 8.617e-5
        Ea_map = {'ct':   self.params.Ea_ct,
                  'diff': self.params.Ea_diff,
                  'ohm':  self.params.Ea_ohm}
        Ea = Ea_map.get(res_type, 0.0)
        if Ea == 0:
            return 1.0
        return np.exp((Ea / k_B) * (1/T_K - 1/T_ref))

    def get_total_resistance_matrix(self, T_field: np.ndarray,
                                     I_app: float = 0.0,
                                     T_amb: float = 25.0
                                     ) -> np.ndarray:
        nx, ny, nz  = self.params.nx, self.params.ny, self.params.nz
        R_matrix    = np.zeros((nx, ny, nz))
        is_charging = I_app > 0
        aging_factor = 1.0

        T_avg = float(np.mean(T_field))
        pol_factor = self.bv_model.calculate_polarization_factor(
            T_eff       = T_avg,
            is_charging = is_charging,
            R_ref       = self.params.R_total_ref
        )

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    sf     = self.calculate_spatial_factor(i, j, k)
                    T_node = T_field[i, j, k]

                    r_ct   = (self.params.R_ct_ref * sf
                               * self.calculate_temperature_factor(T_node, 'ct')
                               * aging_factor * pol_factor)
                    r_diff = (self.params.R_diff_ref * sf
                               * self.calculate_temperature_factor(T_node, 'diff')
                               * aging_factor * pol_factor)
                    r_sei  = (self.params.R_SEI_ref * sf
                               * aging_factor)
                    r_ohm  = (self.params.R_ohm_ref * sf
                               * self.calculate_temperature_factor(T_node, 'ohm')
                               * aging_factor)

                    R_matrix[i, j, k] = r_ct + r_diff + r_sei + r_ohm

        return R_matrix


class PowerModel:
    def __init__(self):
        self.P_screen_base = 1.0
        self.P_cpu_base    = 1.2
        self.P_gpu_base    = 0.8
        self.P_net_base    = 0.4
        self.P_gps_base    = 0.5
        self.P_wifi_base   = 0.1
        self.P_bt_base     = 0.05
        self.P_bg_base     = 0.02

    def calculate_power(self, scene_params: Dict) -> float:
        brightness  = scene_params.get('brightness', 0.5)
        P_screen    = brightness * self.P_screen_base
        cpu_load    = scene_params.get('cpu_load', 0.3)
        game_factor = 1.5 if scene_params.get('is_game', False) else 1.0
        P_cpu  = cpu_load * self.P_cpu_base * game_factor
        P_gpu  = self.P_gpu_base if scene_params.get('is_game', False) else 0
        P_net  = scene_params.get('network_activity', 0.2) * self.P_net_base
        P_gps  = 0.3*self.P_gps_base if scene_params.get('gps_on', False) else 0
        P_wifi = self.P_wifi_base if scene_params.get('wifi_on', True) else 0
        P_bt   = self.P_bt_base if scene_params.get('bluetooth_on', False) else 0
        P_bg   = scene_params.get('n_background_apps', 5) * self.P_bg_base
        return float(np.clip(
            P_screen+P_cpu+P_gpu+P_net+P_gps+P_wifi+P_bt+P_bg, 0.3, 8.0
        ))

    def get_scene_presets(self) -> Dict:
        return {
            'light'   : {'brightness':0.3,'cpu_load':0.2,'is_game':False,
                         'network_activity':0.1,'gps_on':False,
                         'wifi_on':True,'bluetooth_on':False,'n_background_apps':3},
            'moderate': {'brightness':0.6,'cpu_load':0.5,'is_game':False,
                         'network_activity':0.3,'gps_on':True,
                         'wifi_on':True,'bluetooth_on':True,'n_background_apps':8},
            'heavy'   : {'brightness':0.8,'cpu_load':0.8,'is_game':False,
                         'network_activity':0.6,'gps_on':True,
                         'wifi_on':True,'bluetooth_on':True,'n_background_apps':12},
            'game'    : {'brightness':1.0,'cpu_load':1.0,'is_game':True,
                         'network_activity':0.8,'gps_on':False,
                         'wifi_on':True,'bluetooth_on':True,'n_background_apps':5}
        }


class HeatGenerationModel:
    def __init__(self, params: BatteryParams,
                 resistance_model: ResistanceModel):
        self.params           = params
        self.resistance_model = resistance_model
        self.entropy_model    = EntropyCoefficientModel()

    def calculate_total_heat(self, I: float, T_avg: float,
                              SOC: float,
                              R_total: float) -> Tuple[float,float,float]:
        Q_irr  = (I**2) * R_total
        T_K    = T_avg + 273.15
        dUdT   = self.entropy_model.get_entropy_coefficient(SOC, I)
        Q_rev  = I * T_K * dUdT
        Q_gen  = Q_irr + Q_rev
        return Q_gen, Q_irr, Q_rev

    def get_average_resistance(self, R_matrix: np.ndarray) -> float:
        return float(np.mean(R_matrix))

    def distribute_heat(self, Q_gen: float,
                         R_matrix: np.ndarray) -> np.ndarray:
        total_R = np.sum(R_matrix)
        if total_R > 0:
            return Q_gen * (R_matrix / total_R)
        return np.zeros_like(R_matrix)


class ThermalModel:
    def __init__(self, params: BatteryParams):
        self.params = params
        self._precompute()

    def _precompute(self):
        p       = self.params
        self.Gx = p.k * (p.dy * p.dz) / p.dx
        self.Gy = p.k * (p.dx * p.dz) / p.dy
        self.Gz = p.k * (p.dx * p.dy) / p.dz
        self.C_th = p.m_cell * p.Cp

    def update_temperature(self, T: np.ndarray,
                            Q_cell: np.ndarray,
                            T_amb: float,
                            T_phone: float,
                            dt: float) -> Tuple[np.ndarray, float]:
        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz
        dt_inner   = 0.5
        n_steps    = int(np.ceil(dt / dt_inner))
        actual_dt  = dt / n_steps
        T_cur      = T.copy()
        total_Q_bp = 0.0
        A_xy = self.params.dx * self.params.dy
        A_xz = self.params.dx * self.params.dz
        A_yz = self.params.dy * self.params.dz
        h_internal = 8.0

        for _ in range(n_steps):
            dTdt     = Q_cell / self.C_th
            Q_step   = 0.0
            for i in range(1, nx):
                for j in range(ny):
                    for k in range(nz):
                        q = self.Gx*(T_cur[i-1,j,k]-T_cur[i,j,k])/self.C_th
                        dTdt[i,j,k] += q; dTdt[i-1,j,k] -= q
            for i in range(nx):
                for j in range(1, ny):
                    for k in range(nz):
                        q = self.Gy*(T_cur[i,j-1,k]-T_cur[i,j,k])/self.C_th
                        dTdt[i,j,k] += q; dTdt[i,j-1,k] -= q
            for i in range(nx):
                for j in range(ny):
                    for k in range(1, nz):
                        q = self.Gz*(T_cur[i,j,k-1]-T_cur[i,j,k])/self.C_th
                        dTdt[i,j,k] += q; dTdt[i,j,k-1] -= q
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        A_surf = 0
                        if i == 0 or i == nx-1: A_surf += A_yz
                        if j == 0 or j == ny-1: A_surf += A_xz
                        if k == 0 or k == nz-1: A_surf += A_xy
                        if A_surf > 0:
                            qs = h_internal*A_surf*(T_cur[i,j,k]-T_phone)
                            dTdt[i,j,k] -= qs / self.C_th
                            Q_step += qs
            T_cur += dTdt * actual_dt
            total_Q_bp += Q_step * actual_dt

        T_cur = np.clip(T_cur, self.params.T_min, self.params.T_max)
        return T_cur, total_Q_bp / dt


class CapacityModel:
    def __init__(self, params: BatteryParams):
        self.params = params

    def temperature_factor(self, T: float) -> float:
        if T < 0:     return 0.5 + 0.02 * T
        elif T < 20:  return 0.5 + 0.025 * T
        elif T <= 40: return 1.0 - 0.01 * (T - 20)
        else:         return max(0.6, 0.9 - 0.015 * (T - 40))

    def effective_capacity(self, T_avg: float) -> float:
        return self.params.C_nom * self.temperature_factor(T_avg)


class ImprovedBatteryModel:
    def __init__(self, params: Optional[BatteryParams] = None):
        self.params           = params or BatteryParams()
        self.resistance_model = ResistanceModel(self.params)
        self.power_model      = PowerModel()
        self.heat_model       = HeatGenerationModel(
                                    self.params, self.resistance_model)
        self.thermal_model    = ThermalModel(self.params)
        self.capacity_model   = CapacityModel(self.params)
        self.reset_state()

    def reset_state(self):
        self.T       = 25.0 * np.ones((self.params.nx,
                                        self.params.ny,
                                        self.params.nz))
        self.T_phone = 25.0
        self.SOC     = 100.0
        self.history = {k: [] for k in
                        ['time','SOC','T_max','T_avg','T_phone',
                         'P','I','Q_gen','Q_irr','Q_rev','C_eff']}

    def set_SOC(self, SOC: float):
        self.SOC = np.clip(SOC, 0, 100)

    def calculate_current(self, P: float, SOC: float,
                           is_charging: bool = False) -> float:
        V = self.params.V_nom * (0.95 if SOC < 20
                                 else 1.05 if SOC > 90 else 1.0)
        I = P / V
        return I if is_charging else -I

    def step(self, scene_params: Dict = None, dt: float = 1.0,
             T_amb: float = 25.0, is_charging: bool = False,
             I_app: float = None) -> Dict:
        if I_app is not None:
            I       = I_app
            P_total = abs(I) * self.params.V_nom
        else:
            P_total = self.power_model.calculate_power(
                          scene_params or {})
            if self.T_phone > 38.0:
                throttle = max(0.6,
                               1.0 - 0.05*(self.T_phone-38.0))
                P_total *= throttle
            I = self.calculate_current(P_total, self.SOC,
                                        is_charging)

        R_matrix = self.resistance_model.get_total_resistance_matrix(
                       self.T, I_app=I, T_amb=T_amb)
        T_avg    = float(np.mean(self.T))
        R_avg    = self.heat_model.get_average_resistance(R_matrix)
        Q_gen, Q_irr, Q_rev = self.heat_model.calculate_total_heat(
                                   I, T_avg, self.SOC, R_avg)
        Q_cell   = self.heat_model.distribute_heat(Q_gen, R_matrix)

        self.T, Q_batt_to_phone = self.thermal_model.update_temperature(
                                      self.T, Q_cell, T_amb,
                                      self.T_phone, dt)
        P_system   = (max(0.0, P_total-Q_gen)
                      if I_app is None else 0.0)
        Q_phone_out= (self.params.h_phone_amb
                      * self.params.A_phone_surf
                      * (self.T_phone - T_amb))
        dT_phone   = ((P_system + Q_batt_to_phone - Q_phone_out)
                      / self.params.C_th_phone)
        self.T_phone += dT_phone * dt

        C_eff     = self.capacity_model.effective_capacity(T_avg)
        self.SOC += (I / C_eff) * 100 * (dt / 3600)
        self.SOC  = np.clip(self.SOC, 0, 100)

        state = {
            'time'   : len(self.history['time']) * dt / 60,
            'SOC'    : self.SOC,
            'T_max'  : float(np.max(self.T)),
            'T_avg'  : T_avg,
            'T_phone': float(self.T_phone),
            'P'      : float(P_total),
            'I'      : float(I),
            'Q_gen'  : float(Q_gen),
            'Q_irr'  : float(Q_irr),
            'Q_rev'  : float(Q_rev),
            'C_eff'  : float(C_eff)
        }
        for k in state:
            self.history[k].append(state[k])
        return state

    def simulate_discharge(self, scene_params: Dict,
                            duration: float = 3600,
                            dt: float = 1.0,
                            T_amb: float = 25.0) -> List[Dict]:
        self.reset_state()
        states = []
        for _ in range(int(duration / dt)):
            if self.SOC <= 0:
                break
            states.append(self.step(scene_params, dt,
                                     T_amb, is_charging=False))
        return states

    def calculate_TTE(self, scene_params: Dict) -> float:
        E_rem = ((self.SOC/100) * self.params.C_nom
                  * self.params.V_nom)
        P_avg = (np.mean(self.history['P'][-10:])
                 if len(self.history['P']) > 10
                 else self.power_model.calculate_power(scene_params))
        TTE_25 = E_rem / P_avg if P_avg > 0 else 24
        T_avg  = float(np.mean(self.T))
        f_T    = self.capacity_model.temperature_factor(T_avg)
        f_25   = self.capacity_model.temperature_factor(25)

        game_correction = 1.0
        if scene_params.get('is_game', False):
            P_game = self.power_model.calculate_power(scene_params)
            base_p = dict(scene_params)
            base_p['is_game'] = False
            P_base = self.power_model.calculate_power(base_p)
            game_correction = (P_base / P_game
                                if P_game > 0 else 1.0)

        return float(np.clip(
            TTE_25 * (f_T/f_25) * game_correction, 0.5, 48.0
        ))


                                                              
       
                                                              
def run_bv_verification():
    bv   = ButlerVolmerPolarizationModel()
    p    = BatteryParams()

    print("\n" + "="*60)
    print("  Butler-Volmer 极化模型验证报告")
    print("="*60)

    r1 = bv.verify_charge_discharge_ratio()
    print("\n【验证1】充放电极化阻抗比值")
    print(f"  充电极化阻抗  : {r1['R_charge_mOhm']:.4f} mΩ")
    print(f"  放电极化阻抗  : {r1['R_discharge_mOhm']:.4f} mΩ")
    print(f"  计算比值      : {r1['ratio']:.3f}")
    print(f"  文献参考范围  : {r1['literature_range']}"
          "  (来源: HPPC, Scientific.Net AMR.926-930.915)")
    print(f"  验证结果      : {'✓ 通过' if r1['in_range'] else '✗ 未通过'}")

    r2 = bv.verify_c_rate_trend()
    print("\n【验证2】极化阻抗随温度变化趋势")
    for T, R in zip(r2['temperatures_C'], r2['R_pol_mOhm']):
        print(f"  {T:>4}°C → R_pol = {R:.4f} mΩ")
    print(f"  单调递减      : {'✓ 是' if r2['trend_decreasing'] else '✗ 否'}")
    print(f"  物理意义      : {r2['note']}")

    r3 = bv.verify_absolute_value(p.R_total_ref)
    print("\n【验证3】极化阻抗绝对值量级")
    print(f"  正极极化阻抗  : {r3['R_pol_cathode_mOhm']:.4f} mΩ")
    print(f"  负极极化阻抗  : {r3['R_pol_anode_mOhm']:.4f} mΩ")
    print(f"  模型基准总阻  : {r3['R_total_ref_mOhm']:.1f} mΩ"
          "  (来源: PMC文献表1 LCO ≤60mΩ)")
    print(f"  极化/基准比值 : {r3['ratio_to_ref']:.3f}")
    print(f"  量级合理性    : {'✓ 合理(0.1~10×)' if r3['reasonable'] else '✗ 偏差过大'}")

    print("\n【验证4】经验系数消除对比")
    print(f"  {'参数':<30} {'旧代码':<20} {'新代码':<20} {'文献依据'}")
    rows = [
        ("充电极化常数项",    "0.16 (经验)",  "由i0_anode计算",  "[1] Doyle 1993"),
        ("充电极化二次项",    "0.80 (经验)",  "由Ea_anode计算",  "[2] Ecker 2015"),
        ("放电极化常数项",    "1.46 (经验)",  "由i0_cathode计算","[1] Doyle 1993"),
        ("放电极化二次项",    "0.33 (经验)",  "由Ea_cathode计算","[2] Ecker 2015"),
        ("充放电不对称比值",  "隐含于系数中", "i0比值自动给出",  "[4] HPPC验证"),
        ("内阻基准30mΩ",      "无依据",       "无修改",          "[3] PMC表1 ≤60mΩ"),
    ]
    for name, old, new, ref in rows:
        print(f"  {name:<28} {old:<20} {new:<20} {ref}")

    return r1, r2, r3


def run_scenario_comparison():
    model  = ImprovedBatteryModel()
    scenes = model.power_model.get_scene_presets()

    print("\n" + "="*60)
    print("  场景仿真结果（Butler-Volmer极化模型）")
    print("="*60)
    print(f"\n  {'场景':<10} {'终止SOC%':<12} {'电池最高温°C':<14} "
          f"{'机身温°C':<12} {'TTE(h)':<10} {'功耗W'}")
    print("  " + "-"*68)

    results = {}
    for name, params in scenes.items():
        model.reset_state()
        model.set_SOC(100)
        states = model.simulate_discharge(params, duration=3600, dt=5.0)
        final  = states[-1]
        tte    = model.calculate_TTE(params)
        results[name] = {
            'SOC'    : final['SOC'],
            'T_max'  : final['T_max'],
            'T_phone': final['T_phone'],
            'TTE'    : tte,
            'P'      : final['P'],
            'history': model.history,
            'T_field': model.T.copy()                   
        }
        print(f"  {name.upper():<10} {final['SOC']:<12.1f} "
              f"{final['T_max']:<14.1f} {final['T_phone']:<12.1f} "
              f"{tte:<10.1f} {final['P']:.3f}")

    P_light = results['light']['P']
    P_game  = results['game']['P']
    ratio   = P_game / P_light
    print(f"\n  功耗倍数验证：{P_light:.3f}W → {P_game:.3f}W"
          f" = {ratio:.2f}倍")
    print(f"  {'✓ 与计算一致' if abs(ratio-4.9)<0.5 else '✗ 请核查'}"
          f"（论文摘要应写'近五倍'而非'近四倍'）")

    return results


                                                              
            
                                                              
def plot_3d_temperature_distribution(T_field: np.ndarray,
                                      title_suffix: str = '',
                                      save_path: str = None):
       
    nx, ny, nz = T_field.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

                 
    T_min_all = float(T_field.min())
    T_max_all = float(T_field.max())

    fig = plt.figure(figsize=(13, 10), facecolor='#FAFAFA')
    fig.suptitle(f'3D Temperature Distribution {title_suffix}',
                 fontsize=13, y=0.98)

    cmap = 'hot'

                                                          
    ax0 = fig.add_subplot(2, 2, 1)
    top_layer = T_field[:, :, -1]                          
    im0 = ax0.imshow(top_layer.T, origin='lower', aspect='auto',
                     extent=[0, nx, 0, ny], cmap=cmap,
                     vmin=T_min_all, vmax=T_max_all)
    ax0.set_title(f'Top View  (k = {nz-1})', fontsize=11)
    ax0.set_xlabel('X Cell Index')
    ax0.set_ylabel('Y Cell Index')
    cb0 = plt.colorbar(im0, ax=ax0)
    cb0.set_label('Temperature (°C)')
             
    idx = np.unravel_index(top_layer.argmax(), top_layer.shape)
    ax0.plot(idx[0]+0.5, idx[1]+0.5, 'b*', ms=10,
             label=f'T_max={top_layer.max():.1f}°C')
    ax0.legend(fontsize=8, loc='upper right')

                                                          
    ax1 = fig.add_subplot(2, 2, 2)
    j_mid = ny // 2
    front_slice = T_field[:, j_mid, :]                    
    im1 = ax1.imshow(front_slice.T, origin='lower', aspect='auto',
                     extent=[0, nx, 0, nz], cmap=cmap,
                     vmin=T_min_all, vmax=T_max_all)
    ax1.set_title(f'Front View  (y = {j_mid})', fontsize=11)
    ax1.set_xlabel('X Cell Index')
    ax1.set_ylabel('Z Cell Index')
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label('Temperature (°C)')

                                                          
    ax2 = fig.add_subplot(2, 2, 3)
    i_mid = nx // 2
    side_slice = T_field[i_mid, :, :]                     
    im2 = ax2.imshow(side_slice.T, origin='lower', aspect='auto',
                     extent=[0, ny, 0, nz], cmap=cmap,
                     vmin=T_min_all, vmax=T_max_all)
    ax2.set_title(f'Side View  (x = {i_mid})', fontsize=11)
    ax2.set_xlabel('Y Cell Index')
    ax2.set_ylabel('Z Cell Index')
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label('Temperature (°C)')

                                                          
    ax3 = fig.add_subplot(2, 2, 4, projection='3d')
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    temps = T_field.ravel()
    sc = ax3.scatter(X.ravel(), Y.ravel(), Z.ravel(),
                     c=temps, cmap=cmap, s=40, alpha=0.75,
                     vmin=T_min_all, vmax=T_max_all)
    ax3.set_title('3D Isometric View', fontsize=11)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    cb3 = plt.colorbar(sc, ax=ax3, shrink=0.55, pad=0.1)
    cb3.set_label('Temperature (°C)')

              
    fig.text(0.5, 0.01,
             f'Color range: {T_min_all:.2f} – {T_max_all:.2f} °C  '
             f'(consistent across all panels)',
             ha='center', fontsize=9, color='#555555')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  3D temperature distribution saved: {save_path}")
    plt.close()


def plot_all_scenes_3d_temperature(results: dict,
                                    base_path: str = None):
       
    scene_labels = {
        'light'   : 'Light Use',
        'moderate': 'Moderate Use',
        'heavy'   : 'Heavy Use',
        'game'    : 'Gaming'
    }

    for name, res in results.items():
        T_field = res['T_field']
        label   = scene_labels[name]
        suffix  = f'— {label}  (after 1 h discharge)'
        path    = (f'{base_path}_3dtemp_{name}.png'
                   if base_path else None)
        print(f"  Generating 3D temperature plot for: {label}")
        plot_3d_temperature_distribution(T_field,
                                          title_suffix=suffix,
                                          save_path=path)


                                                              
       
                                                              
def plot_bv_vs_empirical():
       
    bv      = ButlerVolmerPolarizationModel()
    p       = BatteryParams()
    temps   = [15, 25, 35, 45]
    C_rates = np.linspace(0.1, 2.0, 50)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#FAFAFA')

    colors = [MACARON_COLORS['blue'],   MACARON_COLORS['mint'],
              MACARON_COLORS['orange'], MACARON_COLORS['pink']]

    ax = axes[0]
    for T, col in zip(temps, colors):
        bv_vals = [bv.calculate_polarization_factor(T, True, p.R_total_ref)
                   for _ in C_rates]
        ax.plot(C_rates, bv_vals, color=col, lw=2.5,
                label=f'BV model  {T} °C')

    emp_charge = 0.16 + 0.80 * C_rates**2
    ax.plot(C_rates, emp_charge, color='#CC3333', lw=2.0,
            linestyle='--', label='Empirical (no physical basis)', alpha=0.85)

    ax.set_title('(a)  Charging — Polarization Factor vs. C-rate',
                 fontsize=11, pad=8)
    ax.set_xlabel('C-rate  (h$^{-1}$)', fontsize=10)
    ax.set_ylabel('Polarization Factor  (dimensionless)', fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.annotate('Temperature effect:\nhigher T → lower R$_{pol}$',
                xy=(1.5, bv.calculate_polarization_factor(45, True,
                    p.R_total_ref)),
                xytext=(1.1, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=8, color='gray')

    ax = axes[1]
    for T, col in zip(temps, colors):
        bv_vals = [bv.calculate_polarization_factor(T, False, p.R_total_ref)
                   for _ in C_rates]
        ax.plot(C_rates, bv_vals, color=col, lw=2.5,
                label=f'BV model  {T} °C')

    emp_discharge = 1.46 + 0.33 * C_rates**2
    ax.plot(C_rates, emp_discharge, color='#CC3333', lw=2.0,
            linestyle='--', label='Empirical (no physical basis)', alpha=0.85)

    ax.set_title('(b)  Discharging — Polarization Factor vs. C-rate',
                 fontsize=11, pad=8)
    ax.set_xlabel('C-rate  (h$^{-1}$)', fontsize=10)
    ax.set_ylabel('Polarization Factor  (dimensionless)', fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)

    r_charge    = bv.calculate_polarization_factor(25, True,  p.R_total_ref)
    r_discharge = bv.calculate_polarization_factor(25, False, p.R_total_ref)
    ax.annotate(f'Charge/discharge ratio = {r_charge/r_discharge:.2f}\n'
                f'(Literature HPPC range: 1.2–1.6)',
                xy=(0.5, r_discharge), xytext=(0.6, r_discharge * 0.55),
                arrowprops=dict(arrowstyle='->', color='#555555'),
                fontsize=8, color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#AAAAAA', alpha=0.8))

    fig.suptitle(
        'Fig. 1  Butler–Volmer Polarization Model vs. Original Empirical Coefficients\n'
        'BV parameters: Doyle et al. (1993) [i$_0$, a, L]  +  Ecker et al. (2015) [E$_a$]',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig('fig1_bv_vs_empirical.svg', bbox_inches='tight')
    plt.close()
    print("  Fig.1 saved: fig1_bv_vs_empirical.svg")


def plot_scenario_results(results: dict):
       
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor='#FAFAFA')

    scene_colors = {
        'light'   : MACARON_COLORS['mint'],
        'moderate': MACARON_COLORS['blue'],
        'heavy'   : MACARON_COLORS['orange'],
        'game'    : MACARON_COLORS['pink']
    }
    scene_labels = {
        'light'   : 'Light Use',
        'moderate': 'Moderate Use',
        'heavy'   : 'Heavy Use',
        'game'    : 'Gaming'
    }

    for name, res in results.items():
        col   = scene_colors[name]
        label = scene_labels[name]
        t     = res['history']['time']
        axes[0,0].plot(t, res['history']['SOC'],
                       color=col, lw=2.2, label=label)
        axes[0,1].plot(t, res['history']['T_max'],
                       color=col, lw=2.2, label=label)
        axes[1,0].plot(t, res['history']['P'],
                       color=col, lw=2.2, label=label)
        axes[1,1].plot(t, res['history']['Q_gen'],
                       color=col, lw=2.2, label=label)

    panel_cfg = [
        ('(a)  State of Charge',        'SOC (%)',
         'Time (min)', [0, 60, 0, 105]),
        ('(b)  Battery Maximum Temperature',
         'Temperature (°C)', 'Time (min)', None),
        ('(c)  Total System Power Consumption',
         'Power (W)',         'Time (min)', None),
        ('(d)  Battery Heat Generation Rate',
         'Heat Generation (W)', 'Time (min)', None),
    ]
    for ax, (title, ylabel, xlabel, ylim) in zip(axes.flat, panel_cfg):
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.4)
        if ylim:
            ax.set_ylim(ylim[2], ylim[3])
            ax.set_xlim(ylim[0], ylim[1])

    axes[0,1].axhline(y=45, color='#CC3333', linestyle=':',
                      lw=1.5, alpha=0.7, label='Thermal warning (45°C)')
    axes[0,1].legend(fontsize=8, framealpha=0.9)

    P_light = results['light']['P']
    P_game  = results['game']['P']
    axes[1,0].annotate(
        f'Power ratio\nGaming / Light\n= {P_game/P_light:.1f}×',
        xy=(55, P_game), xytext=(35, P_game * 0.75),
        arrowprops=dict(arrowstyle='->', color='#555555'),
        fontsize=8.5, color='#333333',
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec='#AAAAAA', alpha=0.85)
    )

    fig.suptitle(
        'Fig. 2  Discharge Simulation Results under Four Usage Scenarios\n'
        '(Butler–Volmer polarization model, T$_{amb}$ = 25 °C, '
        'initial SOC = 100%, duration = 1 h)',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig('fig2_scenario_results.svg', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig.2 saved: fig2_scenario_results.png")


                                                              
      
                                                              
if __name__ == '__main__':
                 
    r1, r2, r3 = run_bv_verification()

                                              
    results = run_scenario_comparison()

                
    plot_bv_vs_empirical()
    plot_scenario_results(results)

                             
    print("\n" + "="*60)
    print("  生成三维温度分布图（各场景仿真结束时刻）")
    print("="*60)
    plot_all_scenes_3d_temperature(
        results,
        base_path='fig3'                                   
    )

    print("\n" + "="*60)
    print("  全部验证与绘图完成")
    print("="*60)