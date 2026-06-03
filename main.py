import src.loading as ld
import src.cleaning as cl
from constants import ORDER_COLUMN

def main():
    result_saber_11 = ld.load_data()
    result_saber_11_clean = cl.cleaning_pipeline(
        result_saber_11,
        'data/raw/feature_standarizacion_school.csv',
        ORDER_COLUMN
    )
    print(f"Hello from trabajo-modulo-programacion! {result_saber_11_clean.size}")
    print(result_saber_11.describe())


if __name__ == "__main__":
    main()
