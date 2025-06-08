import pandas as pd
import sys

def validate(path):
    df = pd.read_csv(path)
    required_vars = ['EDAD', 'ANTIGUEDAD', 'FLG_BANCARIZADO',
                     'FLG_NOMINA', 'SDO_ACTIVO_MENOS0', 'FLG_SEGURO_MENOS0']
    missing_vars = [
        var for var in required_vars if var not in df.columns]

    if missing_vars:
        print(f"❌ Missing values in {path}")
        sys.exit(1)
    print(f"✅ Validated {path}")

# validate("data/train_requerimientos_sample.csv")
validate("data/train_clientes_sample.csv")
