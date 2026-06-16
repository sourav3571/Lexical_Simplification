import sys
import traceback

with open("test_train.log", "w", encoding="utf-8") as log_file:
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        print("Starting test train...")
        import train_cwi
        print("Running train_cwi.main()...")
        train_cwi.main()
        print("Finished successfully!")
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc(file=log_file)
