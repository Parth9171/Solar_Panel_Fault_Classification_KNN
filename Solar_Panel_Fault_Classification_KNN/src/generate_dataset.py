"""Generate the educational synthetic solar PV fault dataset.
This is synthetic, engineering-inspired data; it is NOT field measurement data.
Run: python src/generate_dataset.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_PER_CLASS = 1000
CLASSES = ["Normal", "Partial_Shading", "Soiling", "Open_Circuit", "Short_Circuit", "Overheating"]
MODES = ["MPPT", "FixedVoltage", "Diagnostic"]
SPEC = {
    "Normal": ([700,42,360,8.0,2800,.75,10.0,1.5,2.0,4.0], [100,8,22,1.0,550,.055,2.5,.7,2.0,2.0]),
    "Partial_Shading": ([650,45,320,4.5,1450,.58,9.0,2.7,18.0,6.0], [120,9,28,1.0,450,.075,2.8,1.0,7.0,3.0]),
    "Soiling": ([600,43,350,5.8,2000,.68,9.5,1.8,5.0,5.0], [110,8,24,.9,480,.065,2.5,.8,3.5,2.5]),
    "Open_Circuit": ([700,40,420,.25,25,.08,11.0,.8,92.0,1.0], [110,8,25,.18,30,.035,2.0,.45,8.0,1.0]),
    "Short_Circuit": ([700,52,18,10.0,40,.06,1.4,7.0,78.0,19.0], [100,10,10,1.1,35,.035,.7,2.0,12.0,6.0]),
    "Overheating": ([740,78,300,7.0,2100,.64,5.0,3.2,6.0,30.0], [100,7,25,1.0,500,.065,1.8,1.0,4.0,6.0]),
}
COLS = ["irradiance_W_m2","module_temp_C","dc_voltage_V","dc_current_A","dc_power_W",
        "fill_factor","insulation_resistance_MOhm","voltage_ripple_V",
        "current_imbalance_pct","temperature_rise_C","operating_mode","fault_label"]

def generate_dataset(output_path=None, add_quality_issues=True):
    rng = np.random.default_rng(SEED)
    rows = []
    for label in CLASSES:
        mean, sd = SPEC[label]
        arr = rng.normal(mean, sd, size=(N_PER_CLASS, len(mean)))
        irr,temp,volt,curr,power,ff,insulation,ripple,imbalance,rise = arr.T
        irr=np.clip(irr,80,1100); temp=np.clip(temp,10,100)
        volt=np.clip(volt,0,500); curr=np.clip(curr,0,15)
        power=np.clip(power+rng.normal(0,.035*np.maximum(np.abs(power),50),size=N_PER_CLASS),0,6000)
        ff=np.clip(ff,.01,.95); insulation=np.clip(insulation,.1,20)
        ripple=np.clip(ripple,.1,15); imbalance=np.clip(imbalance,0,100); rise=np.clip(rise,0,60)
        mode=rng.choice(MODES,size=N_PER_CLASS,p=[.55,.25,.20])
        for i in range(N_PER_CLASS):
            rows.append([irr[i],temp[i],volt[i],curr[i],power[i],ff[i],insulation[i],
                         ripple[i],imbalance[i],rise[i],mode[i],label])
    df=pd.DataFrame(rows,columns=COLS)
    if add_quality_issues:
        numeric=COLS[:-2]
        for c in numeric:
            idx=rng.choice(df.index,size=max(1,int(.008*len(df))),replace=False)
            df.loc[idx,c]=np.nan
        for c,lo,hi in [("dc_voltage_V",-80,800),("module_temp_C",-20,150),
                        ("dc_current_A",-3,30),("voltage_ripple_V",0,35)]:
            idx=rng.choice(df.index,size=10,replace=False)
            df.loc[idx,c]=rng.choice([lo,hi],size=len(idx))
        df=pd.concat([df,df.sample(25,random_state=SEED)],ignore_index=True)
    if output_path is None:
        output_path=Path(__file__).resolve().parents[1]/"data"/"solar_pv_faults_synthetic.csv"
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_path,index=False)
    return df

if __name__=="__main__":
    out=Path(__file__).resolve().parents[1]/"data"/"solar_pv_faults_synthetic.csv"
    generated=generate_dataset(out)
    print(f"Wrote {len(generated)} rows to {out}")
