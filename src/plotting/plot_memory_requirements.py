import json
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    data_dir = ROOT_DIR / "memory_analysis"

    files = sorted(list(data_dir.glob("*.json")), key=lambda p: int(p.stem))
    outputs = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            outputs.append(data)

    df = pd.DataFrame(outputs)

    def name_fn(x):
        if "base" in x:
            return "Base"
        else:
            return x.split("(")[1].split(")")[0]

    df["setup"] = df["model"].apply(name_fn)
    df["model"] = df["model"].apply(lambda x: x.split("(")[0])

    df.to_csv(data_dir / "memory.csv", index=False)
