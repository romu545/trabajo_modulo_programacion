import src.loading_rafa as ld

def main():
    saber11 = ld.load_data()
    print(f"Hello from trabajo-modulo-programacion! {saber11.size}")
    print(saber11.describe())


if __name__ == "__main__":
    main()
