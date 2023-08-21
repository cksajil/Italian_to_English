import argparse
from src import create_models_folder
from src import download_model, predict


def main():
    create_models_folder()
    download_model()

    parser = argparse.ArgumentParser(
        description="Translated text in Italian to English"
    )
    parser.add_argument("--it", type=str, help="Input text in Italian")
    args = parser.parse_args()

    if args.it:
        output_text = predict(args.it)
    else:
        print(
            "Please provide a valid text in Italian language to translate as argument"
        )
    print(output_text)


if __name__ == "__main__":
    main()
