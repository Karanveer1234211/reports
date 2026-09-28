import argparse
import pandas as pd


def print_panel_summary(df):
    print("\n" + "=" * 110)
    print("PANEL SUMMARY")
    print("=" * 110)

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns):,}")

    if "symbol" in df.columns:
        print(f"Symbols    : {df['symbol'].nunique():,}")

    if "timestamp" in df.columns:
        print(f"First date : {df['timestamp'].min()}")
        print(f"Last date  : {df['timestamp'].max()}")

    print("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:4d}. {col}")


def show_rows(df, rows=100, tail=False):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 40)

    result = df.tail(rows) if tail else df.head(rows)

    print("\n" + "=" * 110)
    print(f"{'LAST' if tail else 'FIRST'} {rows} ROWS")
    print("=" * 110)
    print(result.to_string(index=False))


def inspect_stock(df, stock, rows=100, tail=False):
    if "symbol" not in df.columns:
        print("\nERROR: Panel does not contain a 'symbol' column.")
        return

    stock = stock.strip().upper()

    s = df[
        df["symbol"].astype(str).str.upper() == stock
    ].copy()

    if s.empty:
        print(f"\nERROR: Stock '{stock}' was not found in the panel.")
        return

    if "timestamp" in s.columns:
        s = s.sort_values("timestamp")

    print("\n" + "=" * 110)
    print(f"STOCK: {stock}")
    print("=" * 110)
    print(f"Rows       : {len(s):,}")
    print(f"Columns    : {len(s.columns):,}")

    if "timestamp" in s.columns:
        print(f"First date : {s['timestamp'].min()}")
        print(f"Last date  : {s['timestamp'].max()}")

    show_rows(s, rows, tail)


def interactive_mode(df):
    while True:
        print("\n" + "=" * 110)
        print("PANEL INSPECTOR")
        print("=" * 110)
        print("1 = First 100 panel rows")
        print("2 = Last 100 panel rows")
        print("3 = Inspect stock")
        print("4 = List stocks")
        print("5 = Panel summary")
        print("Q = Quit")

        choice = input("\nChoice: ").strip().upper()

        if choice == "1":
            show_rows(df, 100, False)

        elif choice == "2":
            show_rows(df, 100, True)

        elif choice == "3":
            stock = input("Enter stock symbol: ").strip()
            if not stock:
                print("No stock entered.")
                continue

            rows_input = input("Rows [100]: ").strip()
            rows = int(rows_input) if rows_input else 100

            mode = input("F = first rows, L = last rows [F]: ").strip().upper()
            tail = mode == "L"

            inspect_stock(df, stock, rows, tail)

        elif choice == "4":
            if "symbol" not in df.columns:
                print("\nERROR: No 'symbol' column found.")
                continue

            symbols = sorted(
                df["symbol"].dropna().astype(str).str.upper().unique()
            )

            print(f"\nTotal symbols: {len(symbols):,}")
            print("\n".join(symbols))

        elif choice == "5":
            print_panel_summary(df)

        elif choice == "Q":
            break

        else:
            print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an existing panel parquet without modifying it."
    )

    parser.add_argument(
        "--panel",
        required=True,
        help="Path to panel.parquet"
    )

    parser.add_argument(
        "--stock",
        default=None,
        help="Inspect a specific stock symbol"
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of rows to display"
    )

    parser.add_argument(
        "--tail",
        action="store_true",
        help="Display last N rows instead of first N"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Display panel summary and exit"
    )

    args = parser.parse_args()

    panel_path = args.panel

    print(f"\nLoading panel:")
    print(panel_path)

    try:
        df = pd.read_parquet(panel_path)
    except Exception as e:
        print("\nERROR loading panel:")
        print(e)
        return

    print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns.")

    if args.summary:
        print_panel_summary(df)
        return

    if args.stock:
        inspect_stock(
            df,
            args.stock,
            args.rows,
            args.tail
        )
        return

    # No command-line stock supplied -> interactive inspector
    interactive_mode(df)


if __name__ == "__main__":
    main()
