import pandas as pd
import json

metadata_keys = {
    "sample_number": "Sample Number",
    "photocatalyst": "PC",
    "solvent": "Solvent",
    "concentration of photocatalyst (M)": "Conc of PC (M)",
    "volume of photocatalyst (mL)": "Vol of PC (mL)",
    "quencher": "Quencher?",
    "concentration of quencher (M)": "Conc. Of Quencher (M)",
    "volume of quencher (mL)": "Vol. of Quencher (mL)",
    "atmosphere": "Atmosphere",
    "sparge_duration (s)": "Sparge Duration (s)",
    "spectrometer power (kW)": "Spectrometer Power (kW)",
    "excitation wavelength start (nm)": "excitation wavelength start (nm)",
    "excitation wavelength end (nm)": "excitation wavelength end (nm)",
    "emission wavelength start (nm)": "emission wavelength start (nm)",
    "emission wavelength end (nm)": "emission wavelength end (nm)",
    "LED_power (%)": "LED_power_%",
    "instrument": "instrument",
}


def csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=0, how="all")
    experiments: dict = {}
    for i, row in df.iterrows():
        experiments[row["Sample Number"]] = {}
        for metadata_key, metadata_value in metadata_keys.items():
            experiments[row["Sample Number"]][metadata_key] = row[metadata_value]
    with open(json_path, "w") as f:
        json.dump(experiments, f)
