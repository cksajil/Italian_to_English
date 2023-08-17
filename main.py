import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Translated text in Italian to English"
    )
    parser.add_argument("--it", type=str, help="Input text in Italian")
    args = parser.parse_args()

    if args.it:
        print(args.it)
    else:
        print(
            "Please provide a valid text in Italian language to translate as argument"
        )
        return


if __name__ == "__main__":
    main()
